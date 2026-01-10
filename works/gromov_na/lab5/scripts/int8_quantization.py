import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pathlib import Path
import os
from glob import glob
from PIL import Image
import torchvision.transforms as T


class RetinaNetCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file="retinanet_int8.cache", calib_batch_size=8, calib_images_dir=None):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.batch_size = calib_batch_size
        self.current = 0
        
        self.files = sorted(glob(os.path.join(calib_images_dir, "*.jpg")) + 
                           glob(os.path.join(calib_images_dir, "*.png")))
        
        self.num_batches = (len(self.files) + self.batch_size - 1) // self.batch_size

        self.device_input = cuda.mem_alloc(trt.volume((1, 3, 640, 640)) * trt.float32.itemsize * self.batch_size)

    def preprocess_image(self, img_path):

        img = Image.open(img_path).convert("RGB")
        transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(img).unsqueeze(0)
        return tensor.numpy()

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current >= self.num_batches:
            return None

        batch_files = self.files[self.current * self.batch_size : (self.current + 1) * self.batch_size]
        batch_imgs = []
        for f in batch_files:
            img_np = self.preprocess_image(f)
            batch_imgs.append(img_np)

        while len(batch_imgs) < self.batch_size:
            batch_imgs.append(batch_imgs[-1])

        batch = np.concatenate(batch_imgs, axis=0).astype(np.float32)

        cuda.memcpy_htod(self.device_input, batch)
        self.current += 1
        return [self.device_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_int8_engine(onnx_path: str, engine_path: str, calib_images_dir: str):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    config = builder.create_builder_config()
    
    config.set_flag(trt.BuilderFlag.INT8)
    
    calibrator = RetinaNetCalibrator(
        cache_file="retinanet_r50_fpn_int8.cache",
        calib_batch_size=8,
        calib_images_dir=calib_images_dir
    )
    config.int8_calibrator = calibrator
    
    print(f"Парсинг ONNX модели {onnx_path}...")
    with open(onnx_path, 'rb') as model:
        success = parser.parse(model.read())
        if not success:
            print("ОШИБКА парсинга ONNX:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))
    config.add_optimization_profile(profile)
    
    engine = builder.build_serialized_network(network, config)
    
    if engine is None:
        raise RuntimeError("Не удалось собрать engine")
    
    with open(engine_path, "wb") as f:
        f.write(engine)
    
    return engine_path

if __name__ == "__main__":
    onnx_path = "models/retinanet_raw_heads.onnx"
    engine_path = "models/retinanet_int8_raw.trt"
    calib_dir = "/l5/calibration_images" 
    
    Path(engine_path).parent.mkdir(exist_ok=True)
    
    build_int8_engine(
        onnx_path,
        engine_path,
        calib_dir
    )