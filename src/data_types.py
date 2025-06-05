from typing import NotRequired, TypedDict

from torch import Tensor

ModelInfo = TypedDict('ModelInfo', {
    'file': str,
    'name': str,
    'version': str,
    'prefix': str
})
ControlNetInfo = TypedDict('ControlNetInfo', {
    'file': str,
    'name': str,
    'version': str,
    'modifier': str,
    'type': str
})
LoRAInfo = TypedDict('LoRAInfo', {
    'file': str,
    'name': str,
    'version': str,
    'prefix': str
})
UpscalerInfo = TypedDict('UpscalerInfo', {
    'file': str,
    'name': str,
})
ModelsInfo = TypedDict('ModelsInfo', {
    'models': list[ModelInfo],
    'controlNets': list[ControlNetInfo],
    'loras': list[LoRAInfo],
    'upscalers': list[UpscalerInfo]
})

_LoraStackItem = TypedDict('_LoraStackItem', {
    'model': LoRAInfo,
    'weight': float,
    'control_image': NotRequired[Tensor]
})
LoraStack = list[_LoraStackItem]

_ControlStackItem = TypedDict('_ControlStackItem', {
    "model": ControlNetInfo,
    "input_type": str,
    "mode": str,
    "weight": float,
    "start": float,
    "end": float,
    "image": NotRequired[Tensor]
})
ControlStack = list[_ControlStackItem]
