import numpy as np

from .generated.LoRA import LoRAT
from .generated.GenerationConfiguration import GenerationConfigurationT
from .data_types import Config, ModelInfo
from .data_types import DrawThingsLists


def round_by_64(x):
    return round((x if np.isfinite(x) else 0) / 64) * 64


def clamp(x, min_val=64, max_val=2048):
    return int(max(min(x if np.isfinite(x) else 0, max_val), min_val))


def clamp_f(x, min_val, max_val):
    return float(max(min(x if np.isfinite(x) else 0, max_val), min_val))


class Model:
    def __init__(self, model_option):
        info = model_option["value"]
        if not info:
            self.file = "unknown"
            self.name = "unknown"
            self.version = "unknown"
            self.prefix = "unknown"
            return
        self.file = info["file"]
        self.name = info["name"]
        self.version = info["version"]
        self.prefix = info["prefix"]

    @property
    def res_dpt_shift(self):
        return self.version in ["flux1", "sd3", "hidream_i1"]

    @property
    def video(self):
        return self.version in [
            "hunyuan_video",
            "wan_v2.1_1.3b",
            "wan_v2.1_14b",
            "svd_i2v",
        ]

    @property
    def tea_cache(self):
        return self.version in [
            "flux1",
            "hidream_i1",
            "wan_v2.1_1.3b",
            "wan_v2.1_14b",
            "hunyuan_video",
        ]

    @property
    def speed_up(self):
        return self.version in ["flux1", "hidream_i1", "hunyuan_video"]

    @property
    def clip_l(self):
        return self.version in ["flux1", "hidream_i1", "sd3"]

    @property
    def open_clip_g(self):
        return self.version in ["sd3"]

    @property
    def svd(self):
        return self.version in ["svd_i2v"]

    @property
    def causal_inference(self):
        return self.version in ["wan_v2.1_1.3b", "wan_v2.1_14b"]

    @property
    def sdxl(self):
        return self.version in [
            "sdxl_base_v0.9",
            "sdxl_refiner_v0.9",
        ]


def build_config(config: Config):
    configT = GenerationConfigurationT()
    apply_common(config, configT)
    apply_conditional(config, configT)
    apply_control(config, configT)
    apply_lora(config, configT)
    model = Model(config.get("model"))
    return (configT, model.version)


def apply_control(config: Config, configT: GenerationConfigurationT):
    configT.controls = []


def apply_lora(config: Config, configT: GenerationConfigurationT):
    configT.loras = []

    if "lora" not in config or config["lora"] is None or len(config["lora"]) == 0:
        return

    for lora in config["lora"]:
        if "model" not in lora or "weight" not in lora:
            continue
        loraT = LoRAT()
        loraT.file = lora["model"]["file"]
        loraT.weight = lora["weight"]

        configT.loras.append(loraT)


def apply_common(config: Config, configT: GenerationConfigurationT):
    if "model" in config:
        configT.model = Model(config["model"]).file
    if "width" in config:
        configT.startWidth = config["width"] // 64
    if "height" in config:
        configT.startHeight = config["height"] // 64
    if "seed" in config:
        configT.seed = int(config["seed"] % 4294967295)
    if "seed_mode" in config:
        configT.seedMode = DrawThingsLists.seed_mode.index(config["seed_mode"]) or 0
    if "steps" in config:
        configT.steps = config["steps"]
    if "cfg" in config:
        configT.guidanceScale = config["cfg"]
    if "strength" in config:
        configT.strength = config["strength"]
    if "sampler_name" in config:
        configT.sampler = (
            DrawThingsLists.sampler_list.index(config["sampler_name"]) or 0
        )
    if "batch_count" in config:
        configT.batchCount = config["batch_count"]
    if "batch_size" in config:
        configT.batchSize = config["batch_size"]
    if "clip_skip" in config:
        configT.clipSkip = config["clip_skip"]
    if "mask_blur" in config:
        configT.maskBlur = config["mask_blur"]
    if "mask_blur_outset" in config:
        configT.maskBlurOutset = config["mask_blur_outset"]
    if "sharpness" in config:
        configT.sharpness = config["sharpness"]
    if "shift" in config:
        configT.shift = config["shift"]
    if "preserve_original" in config:
        configT.preserveOriginalAfterInpaint = bool(config["preserve_original"])
    # if "image_guidance_scale" in config:
    #     configT.imageGuidanceScale = config["image_guidance_scale"]


def apply_conditional(config: Config, configT: GenerationConfigurationT):
    model = Model(config.get("model"))

    if config.get("high_res_fix"):
        configT.hiresFix = True
        if "high_res_fix_start_width" in config:
            configT.hiresFixStartWidth = config["high_res_fix_start_width"] // 64
        if "high_res_fix_start_height" in config:
            configT.hiresFixStartHeight = config["high_res_fix_start_height"] // 64
        if "high_res_fix_strength" in config:
            configT.hiresFixStrength = config["high_res_fix_strength"]

    if config.get("tiled_decoding"):
        configT.tiledDecoding = True
        if "decoding_tile_width" in config:
            configT.decodingTileWidth = config["decoding_tile_width"] // 64
        if "decoding_tile_height" in config:
            configT.decodingTileHeight = config["decoding_tile_height"] // 64
        if "decoding_tile_overlap" in config:
            configT.decodingTileOverlap = config["decoding_tile_overlap"] // 64

    if config.get("tiled_diffusion"):
        configT.tiledDiffusion = True
        if "diffusion_tile_width" in config:
            configT.diffusionTileWidth = config["diffusion_tile_width"] // 64
        if "diffusion_tile_height" in config:
            configT.diffusionTileHeight = config["diffusion_tile_height"] // 64
        if "diffusion_tile_overlap" in config:
            configT.diffusionTileOverlap = config["diffusion_tile_overlap"] // 64

    if model.res_dpt_shift and config.get("res_dpt_shift"):
        configT.resolutionDependentShift = True

    # separate_clip_l
    # clip_l_text
    if model.clip_l and config.get("separate_clip_l"):
        configT.separateClipL = True
        if "clip_l_text" in config:
            configT.clipLText = config["clip_l_text"]

    # separate_open_clip_g
    # open_clip_g_text
    if model.open_clip_g and config.get("separate_open_clip_g"):
        configT.separateOpenClipG = True
        if "open_clip_g_text" in config:
            configT.openClipGText = config["open_clip_g_text"]

    # speed_up_with_guidance_embed
    if model.speed_up and config.get("speed_up_with_guidance_embed") == False:
        configT.speedUpWithGuidanceEmbed = False
        if "guidance_embed" in config:
            configT.guidanceEmbed = config["guidance_embed"]

    # tea_cache_start
    # tea_cache_end
    # tea_cache_threshold
    # tea_cache
    # tea_cache_max_skip_steps
    if model.tea_cache and config.get("tea_cache"):
        configT.teaCache = True
        if "tea_cache_start" in config:
            configT.teaCacheStart = config["tea_cache_start"]
        if "tea_cache_end" in config:
            configT.teaCacheEnd = config["tea_cache_end"]
        if "tea_cache_threshold" in config:
            configT.teaCacheThreshold = config["tea_cache_threshold"]
        if "tea_cache_max_skip_steps" in config:
            configT.teaCacheMaxSkipSteps = config["tea_cache_max_skip_steps"]

    if model.speed_up and config.get("speed_up_with_guidance_embed") == False:
        configT.speedUpWithGuidanceEmbed = False
        if "guidance_embed" in config:
            configT.guidanceEmbed = config["guidance_embed"]

    if model.video and "num_frames" in config:
        configT.numFrames = config["num_frames"]

    if model.svd:
        if "fps" in config:
            configT.fpsId = config["fps"]
        if "motion_scale" in config:
            configT.motionBucketId = config["motion_scale"]
        if "guiding_frame_noise" in config:
            configT.condAug = config["guiding_frame_noise"]
        if "start_frame_guidance" in config:
            configT.startFrameCfg = config["start_frame_guidance"]

    if model.causal_inference:
        if "causal_inference" in config:
            if config["causal_inference"] > 0:
                configT.causalInferenceEnabled = True
                configT.causalInference = config["causal_inference"]
            else:
                configT.causalInferenceEnabled = False

    if model.sdxl:
        if "height" in config:
            configT.originalImageHeight = config['height']
            configT.targetImageHeight = config['height']
            configT.negativeOriginalImageHeight = config['height'] // 2
        if "width" in config:
            configT.originalImageWidth = config['width']
            configT.targetImageWidth = config['width']
            configT.negativeOriginalImageWidth = config['width'] // 2

    # stochastic_sampling_gamma


def coerce_common(config: Config, configT: GenerationConfigurationT):
    if "start_width" in config:
        configT.startWidth = round_by_64(clamp(config["start_width"]))
    if "start_height" in config:
        configT.startHeight = round_by_64(clamp(config["start_height"]))
    if "seed" in config:
        configT.seed = int(config["seed"] % 4294967295)
    if "seed_mode" in config:
        configT.seedMode = DrawThingsLists.seed_mode.index(config["seed_mode"]) or 0
    if "steps" in config:
        configT.steps = clamp(config["steps"], 1, 150)
    if "guidance_scale" in config:
        configT.guidanceScale = clamp_f(config["guidance_scale"], 0, 50)
    if "strength" in config:
        configT.strength = clamp_f(config["strength"], 0, 1)
    if "sampler" in config:
        configT.sampler = DrawThingsLists.sampler_list.index(config["sampler"]) or 0
    if "batch_count" in config:
        configT.batchCount = clamp(config["batch_count"], 1, 4)
    if "batch_size" in config:
        configT.batchSize = clamp(config["batch_size"], 1, 1)
    if "clip_skip" in config:
        configT.clipSkip = clamp(config["clip_skip"], 1, 23)
    if "mask_blur" in config:
        configT.maskBlur = clamp_f(config["mask_blur"], 0, 15)
    if "mask_blur_outset" in config:
        configT.maskBlurOutset = clamp(config["mask_blur_outset"], -100, 100)
    if "sharpness" in config:
        configT.sharpness = clamp_f(config["sharpness"], 0, 30)
    if "shift" in config:
        configT.shift = clamp_f(config["shift"], 0.1, 8)
    if "preserve_original_after_inpaint" in config:
        configT.preserveOriginalAfterInpaint = bool(
            config["preserve_original_after_inpaint"]
        )
    if "resolution_dependent_shift" in config:
        configT.resolutionDependentShift = bool(config["resolution_dependent_shift"])
    if "image_guidance_scale" in config:
        configT.imageGuidanceScale = clamp_f(config["image_guidance_scale"], 0, 50)


def coerce_conditional(config: Config, configT: GenerationConfigurationT):
    if config.get("hires_fix"):
        configT.hiresFix = True
        if "hires_fix_start_width" in config:
            configT.hiresFixStartWidth = round_by_64(
                clamp(config["hires_fix_start_width"])
            )
        if "hires_fix_start_height" in config:
            configT.hiresFixStartHeight = round_by_64(
                clamp(config["hires_fix_start_height"])
            )
        if "hires_fix_strength" in config:
            configT.hiresFixStrength = clamp_f(config["hires_fix_strength"], 0, 1)

    if config.get("tiled_decoding"):
        configT.tiledDecoding = True
        if "decoding_tile_width" in config:
            configT.decodingTileWidth = round_by_64(
                clamp(config["decoding_tile_width"])
            )
        if "decoding_tile_height" in config:
            configT.decodingTileHeight = round_by_64(
                clamp(config["decoding_tile_height"])
            )
        if "decoding_tile_overlap" in config:
            configT.decodingTileOverlap = round_by_64(
                clamp(config["decoding_tile_overlap"], max_val=1024)
            )

    if config.get("tiled_diffusion"):
        configT.tiledDiffusion = True
        if "diffusion_tile_width" in config:
            configT.diffusionTileWidth = round_by_64(
                clamp(config["diffusion_tile_width"])
            )
        if "diffusion_tile_height" in config:
            configT.diffusionTileHeight = round_by_64(
                clamp(config["diffusion_tile_height"])
            )
        if "diffusion_tile_overlap" in config:
            configT.diffusionTileOverlap = round_by_64(
                clamp(config["diffusion_tile_overlap"], max_val=1024)
            )

    # separate_clip_l
    # clip_l_text
    if config.get("separate_clip_l"):
        configT.separateClipL = True
        if "clip_l_text" in config:
            configT.clipLText = str(config["clip_l_text"])

    # separate_open_clip_g
    # open_clip_g_text
    if config.get("separate_open_clip_g"):
        configT.separateOpenClipG = True
        if "open_clip_g_text" in config:
            configT.openClipGText = str(config["open_clip_g_text"])

    # speed_up_with_guidance_embed
    if not config.get("speed_up_with_guidance_embed"):
        configT.speedUpWithGuidanceEmbed = False
        if "guidance_embed" in config:
            configT.guidanceEmbed = clamp_f(config["guidance_embed"], 0, 50)

    # tea_cache_start
    # tea_cache_end
    # tea_cache_threshold
    # tea_cache
    # tea_cache_max_skip_steps
    if config.get("tea_cache"):
        configT.teaCache = True
        if (
            "tea_cache_start" in config
            and "tea_cache_end" in config
            and "steps" in config
        ):
            steps = config["steps"]
            a = clamp(config["tea_cache_start"], 0, steps)
            b = clamp(config["tea_cache_end"], 0, steps)
            configT.teaCacheStart = min(a, b)
            configT.teaCacheEnd = max(a, b)
        if "tea_cache_threshold" in config:
            configT.teaCacheThreshold = config["tea_cache_threshold"]
        if "tea_cache_max_skip_steps" in config:
            configT.teaCacheMaxSkipSteps = config["tea_cache_max_skip_steps"]

    # stochastic_sampling_gamma
