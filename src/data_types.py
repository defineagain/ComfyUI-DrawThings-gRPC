from typing import NotRequired, TypedDict
from torch import Tensor

ModelInfo = TypedDict(
    "ModelInfo", {"file": str, "name": str, "version": str, "prefix": str}
)
ControlNetInfo = TypedDict(
    "ControlNetInfo",
    {"file": str, "name": str, "version": str, "modifier": str, "type": str},
)
LoRAInfo = TypedDict(
    "LoRAInfo", {"file": str, "name": str, "version": str, "prefix": str}
)
UpscalerInfo = TypedDict(
    "UpscalerInfo",
    {
        "file": str,
        "name": str,
    },
)
ModelsInfo = TypedDict(
    "ModelsInfo",
    {
        "models": list[ModelInfo],
        "controlNets": list[ControlNetInfo],
        "loras": list[LoRAInfo],
        "upscalers": list[UpscalerInfo],
    },
)

_LoraStackItem = TypedDict(
    "_LoraStackItem",
    {"model": LoRAInfo, "weight": float, "control_image": NotRequired[Tensor]},
)
LoraStack = list[_LoraStackItem]

_ControlStackItem = TypedDict(
    "_ControlStackItem",
    {
        "model": ControlNetInfo,
        "input_type": str,
        "mode": str,
        "weight": float,
        "start": float,
        "end": float,
        "image": NotRequired[Tensor],
    },
)
ControlStack = list[_ControlStackItem]

# this should match kwargs in the sampler node method
class Config(TypedDict, total=False):
    settings: str
    server: str
    port: str
    use_tls: bool
    width: int
    height: int
    seed: int
    seed_mode: str
    steps: int
    cfg: float
    strength: float
    sampler_name: str
    batch_count: int
    batch_size: int
    clip_skip: int
    mask_blur: float
    mask_blur_outset: int
    sharpness: float
    shift: float
    preserve_original: bool
    res_dpt_shift: bool
    image_guidance_scale: float

    model: str
    version: str
    control_net: ControlStack
    lora: LoraStack
    upscaler: dict
    # upscaler_scale_factor: int
    # refiner_model: str
    # refiner_start: float
    refiner: dict

    num_frames: int
    fps: int
    motion_scale: int
    guiding_frame_noise: float
    start_frame_guidance: float
    causal_inference: int

    # conditional
    high_res_fix: bool
    high_res_fix_start_width: int
    high_res_fix_start_height: int
    high_res_fix_strength: float

    tiled_decoding: bool
    decoding_tile_width: int
    decoding_tile_height: int
    decoding_tile_overlap: int

    tiled_diffusion: bool
    diffusion_tile_width: int
    diffusion_tile_height: int
    diffusion_tile_overlap: int

    separate_clip_l: bool
    clip_l_text: str

    separate_open_clip_g: bool
    open_clip_g_text: str

    speed_up: bool
    guidance_embed: float

    tea_cache_start: int
    tea_cache_end: int
    tea_cache_threshold: float
    tea_cache: bool
    tea_cache_max_skip_steps: int

    # face_restoration: str
    # decode_with_attention: bool
    # hires_fix_decode_with_attention: int
    # clip_weight: int
    # negative_prompt_for_image_prior: int
    # image_prior_steps: int
    # original_image_height: int
    # original_image_width: int
    # crop_top: int
    # crop_left: int
    # target_image_height: int
    # target_image_width: int
    # aesthetic_score: int
    # negative_aesthetic_score: int
    # zero_negative_prompt: int
    # negative_original_image_height: int
    # negative_original_image_width: int
    # name: int

    # stage_2_steps: int
    # stage_2_cfg: int
    # stage_2_shift: int
    # stochastic_sampling_gamma: int
    # t5_text_encoder: int
    # separate_t5: int
    # t5_text: int
    # causal_inference_enabled: bool


class DrawThingsLists:
    dtserver = "localhost"
    dtport = "7859"

    sampler_list = [
        "DPM++ 2M Karras",
        "Euler A",
        "DDIM",
        "PLMS",
        "DPM++ SDE Karras",
        "UniPC",
        "LCM",
        "Euler A Substep",
        "DPM++ SDE Substep",
        "TCD",
        "Euler A Trailing",
        "DPM++ SDE Trailing",
        "DPM++ 2M AYS",
        "Euler A AYS",
        "DPM++ SDE AYS",
        "DPM++ 2M Trailing",
        "DDIM Trailing",
    ]

    seed_mode = [
        "Legacy",
        "TorchCpuCompatible",
        "ScaleAlike",
        "NvidiaGpuCompatible",
    ]

    control_mode = [
        "Balanced",
        "Prompt",
        "Control",
    ]

    control_input_type = [
        "Unspecified",
        "Custom",  # -> Slot
        "Depth",  # -> Slot
        "Canny",
        "Scribble",  # -> Slot
        "Pose",  # -> Slot
        "Normalbae",
        "Color",  # -> Slot
        "Lineart",
        "Softedge",
        "Seg",
        "Inpaint",
        "Ip2p",
        "Shuffle",
        "Mlsd",
        "Tile",
        "Blur",
        "Lowquality",
        "Gray",
    ]
