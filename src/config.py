import numpy as np
from .generated.GenerationConfiguration import GenerationConfigurationT
from .data_types import Config
from .data_types import DrawThingsLists
from .config_properties import property_data


def round_by_64(x):
    return round((x if np.isfinite(x) else 0) / 64) * 64


def clamp(x, min_val=64, max_val=2048):
    return int(max(min(x if np.isfinite(x) else 0, max_val), min_val))


def clamp_f(x, min_val, max_val):
    return float(max(min(x if np.isfinite(x) else 0, max_val), min_val))


def build_config(config: Config):
    configT = GenerationConfigurationT()
    apply_standard(config, configT)
    apply_conditional(config, configT)


def apply_standard(config: Config, configT: GenerationConfigurationT):
    if "start_width" in config:
        configT.startWidth = config["start_width"]
    if "start_height" in config:
        configT.startHeight = config["start_height"]
    if "seed" in config:
        configT.seed = int(config["seed"] % 4294967295)
    if "seed_mode" in config:
        configT.seedMode = DrawThingsLists.seed_mode.index(config["seed_mode"]) or 0
    if "steps" in config:
        configT.steps = config["steps"]
    if "guidance_scale" in config:
        configT.guidanceScale = config["guidance_scale"]
    if "strength" in config:
        configT.strength = config["strength"]
    if "sampler" in config:
        configT.sampler = DrawThingsLists.sampler_list.index(config["sampler"]) or 0
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
    if "preserve_original_after_inpaint" in config:
        configT.preserveOriginalAfterInpaint = bool(
            config["preserve_original_after_inpaint"]
        )
    if "resolution_dependent_shift" in config:
        configT.resolutionDependentShift = bool(config["resolution_dependent_shift"])
    if "image_guidance_scale" in config:
        configT.imageGuidanceScale = config["image_guidance_scale"]


def coerce_standard(config: Config, configT: GenerationConfigurationT):
    if 'start_width' in config:
        configT.startWidth = round_by_64(clamp(config['start_width']))
    if 'start_height' in config:
        configT.startHeight = round_by_64(clamp(config['start_height']))
    if 'seed' in config:
        configT.seed = int(config["seed"] % 4294967295)
    if 'seed_mode' in config:
        configT.seedMode = DrawThingsLists.seed_mode.index(config['seed_mode']) or 0
    if 'steps' in config:
        configT.steps = clamp(config['steps'], 1, 150)
    if 'guidance_scale' in config:
        configT.guidanceScale = clamp_f(config['guidance_scale'], 0, 50)
    if 'strength' in config:
        configT.strength = clamp_f(config['strength'], 0, 1)
    if 'sampler' in config:
        configT.sampler = DrawThingsLists.sampler_list.index(config['sampler']) or 0
    if 'batch_count' in config:
        configT.batchCount = clamp(config['batch_count'], 1, 4)
    if 'batch_size' in config:
        configT.batchSize = clamp(config['batch_size'], 1, 1)
    if 'clip_skip' in config:
        configT.clipSkip = clamp(config['clip_skip'], 1, 23)
    if 'mask_blur' in config:
        configT.maskBlur = clamp_f(config['mask_blur'], 0, 15)
    if 'mask_blur_outset' in config:
        configT.maskBlurOutset = clamp(config['mask_blur_outset'], -100, 100)
    if 'sharpness' in config:
        configT.sharpness = clamp_f(config['sharpness'], 0, 30)
    if 'shift' in config:
        configT.shift = clamp_f(config['shift'], 0.1, 8)
    if 'preserve_original_after_inpaint' in config:
        configT.preserveOriginalAfterInpaint = bool(config['preserve_original_after_inpaint'])
    if 'resolution_dependent_shift' in config:
        configT.resolutionDependentShift = bool(config['resolution_dependent_shift'])
    if 'image_guidance_scale' in config:
        configT.imageGuidanceScale = clamp_f(config['image_guidance_scale'], 0, 50)


def apply_conditional(config: Config, configT: GenerationConfigurationT):
    if config.get("hires_fix"):
        configT.hiresFix = True
        if 'hires_fix_start_width' in config:
            configT.hiresFixStartWidth = config["hires_fix_start_width"]
        if 'hires_fix_start_height' in config:
            configT.hiresFixStartHeight = config['hires_fix_start_height']
        if 'hires_fix_strength' in config:
            configT.hiresFixStrength = config['hires_fix_strength']

    if config.get("tiled_decoding"):
        configT.tiledDecoding = True
        if 'decoding_tile_width' in config:
            configT.decodingTileWidth = config["decoding_tile_width"]
        if 'decoding_tile_height' in config:
            configT.decodingTileHeight = config["decoding_tile_height"]
        if 'decoding_tile_overlap' in config:
            configT.decodingTileOverlap = config["decoding_tile_overlap"]

    if config.get("tiled_diffusion"):
        configT.tiledDiffusion = True
        if 'diffusion_tile_width' in config:
            configT.diffusionTileWidth = config["diffusion_tile_width"]
        if 'diffusion_tile_height' in config:
            configT.diffusionTileHeight = config["diffusion_tile_height"]
        if 'diffusion_tile_overlap' in config:
            configT.diffusionTileOverlap = config["diffusion_tile_overlap"]

    # separate_clip_l
    # clip_l_text
    if config.get("separate_clip_l"):
        configT.separateClipL = True
        if 'clip_l_text' in config:
            configT.clipLText = config["clip_l_text"]

    # separate_open_clip_g
    # open_clip_g_text
    if config.get("separate_open_clip_g"):
        configT.separateOpenClipG = True
        if 'open_clip_g_text' in config:
            configT.openClipGText = config["open_clip_g_text"]

    # speed_up_with_guidance_embed
    if not config.get("speed_up_with_guidance_embed"):
        configT.speedUpWithGuidanceEmbed = False
        if 'guidance_embed' in config:
            configT.guidanceEmbed = config["guidance_embed"]

    # tea_cache_start
    # tea_cache_end
    # tea_cache_threshold
    # tea_cache
    # tea_cache_max_skip_steps
    if config.get("tea_cache"):
        configT.teaCache = True
        if 'tea_cache_start' in config and 'tea_cache_end' in config and 'steps' in config:
            steps = config["steps"]
            a = config["tea_cache_start"]
            b = config["tea_cache_end"]
            configT.teaCacheStart = min(a, b)
            configT.teaCacheEnd = max(a, b)
        if 'tea_cache_threshold' in config:
            configT.teaCacheThreshold = config["tea_cache_threshold"]
        if 'tea_cache_max_skip_steps' in config:
            configT.teaCacheMaxSkipSteps = config["tea_cache_max_skip_steps"]

    # stochastic_sampling_gamma


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


class Property:
    def __init__(self, prop_data):
        self.fbs = prop_data[0]
        self.py = prop_data[1]
        self.node = prop_data[2]
        self.json = prop_data[3]
        self.type = prop_data[4]
        self.default = prop_data[5]

        if type == "int" or type == "float":
            self.vmin = prop_data[6]
            self.vmax = prop_data[7]
            self.step = prop_data[8]
            self.spec = prop_data[9]

        if type == "bool" or type == "string":
            self.spec = prop_data[7]

    def validate(self, v):
        if self.type == 'int' or self.type == 'float':
            return True


def find_by_fbs(fbs_name):
    for p in property_data:
        if p[0] == fbs_name and None not in p[0:3]:
            return Property(*p)
    return None


def find_by_py(fbs_name):
    for p in property_data:
        if p[1] == fbs_name and None not in p[0:3]:
            return Property(p)
    return None
