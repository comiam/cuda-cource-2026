import torch
import torchvision
import onnx
import numpy as np
import os
from pathlib import Path
import onnxruntime as ort


def export_retinanet_onnx():

    weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=weights, progress=True)
    
    model.eval()

    class RawRetinaNet(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images):
            features = self.model.backbone(images)
            features = list(features.values())
            head_outputs = self.model.head(features)
            return head_outputs['cls_logits'], head_outputs['bbox_regression']

    raw_model = RawRetinaNet(model)
    
    categories = weights.meta.get("categories", {})
    if categories:
        if isinstance(categories, dict):
            labels_path = Path("models/labels.txt")
            labels_path.parent.mkdir(exist_ok=True)
            with open(labels_path, 'w', encoding='utf-8') as f:
                for idx, name in categories.items():
                    f.write(f"{name}\n")
        elif isinstance(categories, list):
            labels_path = Path("models/labels.txt")
            labels_path.parent.mkdir(exist_ok=True)
            with open(labels_path, 'w', encoding='utf-8') as f:
                for name in categories:
                    f.write(f"{name}\n")
    
    dummy_input = torch.randn(1, 3, 640, 640, requires_grad=True)
    
    with torch.no_grad():
        dummy_output = model(dummy_input)
        print(f"Выход модели: {type(dummy_output)}, количество обнаружений: {len(dummy_output)}")
    
    onnx_path = Path("models/retinanet_r50_fpn.onnx")
    onnx_path.parent.mkdir(exist_ok=True)
    
    torch.onnx.export(
        raw_model,
        dummy_input,
        "models/retinanet_raw_heads.onnx",
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['cls_logits', 'bbox_regression'],
        dynamo=False,
    )
    
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    return onnx_path


def validate_onnx_model(onnx_path):
    ort_session = None

    ort_session = ort.InferenceSession(onnx_path)
    
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
    ort_outputs = ort_session.run(None, ort_inputs)

    for i, output in enumerate(ort_outputs):
        print(f"  Выход {i}: {output.shape}")


if __name__ == "__main__":
    onnx_path = export_retinanet_onnx()
    validate_onnx_model(onnx_path)