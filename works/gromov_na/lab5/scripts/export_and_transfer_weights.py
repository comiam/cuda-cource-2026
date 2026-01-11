import onnx
import torch
import torchvision
import numpy as np
from pathlib import Path
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection.backbone_utils import BackboneWithFPN


def create_mobilenet_with_resnet_head_weights():

    mobilenet = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
    backbone_features = mobilenet.features

    return_layers = {
        '4': '0',
        '7': '1',
        '13': '2',
    }

    body = IntermediateLayerGetter(backbone_features, return_layers)

    out_channels = 256
    base_fpn = BackboneWithFPN(
        body,
        return_layers=return_layers,
        in_channels_list=[40, 80, 160],
        out_channels=out_channels,
        extra_blocks=None,
    )

    class ExtendedMobileNetFPN(torch.nn.Module):
        def __init__(self, base_fpn):
            super().__init__()
            self.base_fpn = base_fpn

            self.p6_conv = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )

            self.p7_conv = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )

        def forward(self, x):
            base_features = self.base_fpn(x)

            p3 = base_features['0']
            p4 = base_features['1']
            p5 = base_features['2']

            p6 = self.p6_conv(p5)

            p7 = self.p7_conv(torch.nn.functional.relu(p6))

            return [p3, p4, p5, p6, p7]

    extended_fpn = ExtendedMobileNetFPN(base_fpn)

    
    head = RetinaNetHead(
        in_channels=out_channels,
        num_anchors=9,
        num_classes=91
    )

    class RawMobileNetRetinaNet(torch.nn.Module):
        def __init__(self, fpn, head):
            super().__init__()
            self.backbone = fpn
            self.head = head

        def forward(self, images):
            features = self.backbone(images)

            head_outputs = self.head(features)
            return head_outputs['cls_logits'], head_outputs['bbox_regression']

    mobilenet_model = RawMobileNetRetinaNet(extended_fpn, head)

    resnet_weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
    resnet_model = torchvision.models.detection.retinanet_resnet50_fpn(weights=resnet_weights, progress=True)
    resnet_model.eval()

    class RawResNet(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.backbone = model.backbone
            self.head = model.head

        def forward(self, images):
            features = self.backbone(images)
            if isinstance(features, dict):
                features = list(features.values())
            head_outputs = self.head(features)
            return head_outputs['cls_logits'], head_outputs['bbox_regression']

    raw_resnet = RawResNet(resnet_model)
    raw_resnet.eval()

    resnet_head_state = raw_resnet.head.state_dict()

    mobilenet_head_dict = mobilenet_model.head.state_dict()

    transferred = 0
    for name, resnet_param in resnet_head_state.items():
        if name in mobilenet_head_dict:
            if resnet_param.shape == mobilenet_head_dict[name].shape:
                mobilenet_head_dict[name].copy_(resnet_param)
                transferred += 1
            else:
                pass
        else:
            pass

    mobilenet_model.head.load_state_dict(mobilenet_head_dict)

    return mobilenet_model


def export_model_with_transferred_weights():

    model = create_mobilenet_with_resnet_head_weights()
    model.eval()

    dummy_input = torch.randn(1, 3, 640, 640, requires_grad=False)

    onnx_path = Path("models/retinanet_mobilenet_int8_transferred_weights.onnx")
    onnx_path.parent.mkdir(exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path.as_posix(),
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


if __name__ == "__main__":
    export_model_with_transferred_weights()