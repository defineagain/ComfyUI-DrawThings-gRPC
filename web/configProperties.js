import { updateNodeModels } from './models.js'
import { samplers, seedModes } from './ComfyUI-DrawThings-gRPC.js'
import { calcShift } from './widgets.js'

/** [ config.fbs name, comfy widget name, the node it belongs to, the property name in DT's json config] */
const propertyData = [
    ['start_width', 'width', 'DrawThingsSampler', 'width'],
    ['start_height', 'height', 'DrawThingsSampler', 'height'],
    ['seed', 'seed', 'DrawThingsSampler', 'seed'],
    ['steps', 'steps', 'DrawThingsSampler', 'steps'],
    ['guidance_scale', 'cfg', 'DrawThingsSampler', 'guidanceScale'],
    ['strength', 'strength', 'DrawThingsSampler', 'strength'],
    ['model', 'model', 'DrawThingsSampler', 'model'],
    ['sampler', 'sampler_name', 'DrawThingsSampler', 'sampler'],
    ['batch_count', 'batch_count', 'DrawThingsSampler', 'batchCount'],
    ['batch_size', 'batch_size', 'DrawThingsSampler', 'batchSize'],
    ['hires_fix', 'high_res_fix', 'DrawThingsSampler', 'hiresFix'],
    ['hires_fix_start_width', 'high_res_fix_start_width', 'DrawThingsSampler', 'hiresFixWidth'],
    ['hires_fix_start_height', 'high_res_fix_start_height', 'DrawThingsSampler', 'hiresFixHeight'],
    ['hires_fix_strength', 'high_res_fix_strength', 'DrawThingsSampler', 'hiresFixStrength'],

    // in upscaler node
    ['upscaler', 'upscaler', 'DrawThingsUpscaler', 'upscaler'],

    ['image_guidance_scale', 'image_guidance_scale', 'DrawThingsSampler', 'imageGuidanceScale'],
    ['seed_mode', 'seed_mode', 'DrawThingsSampler', 'seedMode'],
    ['clip_skip', 'clip_skip', 'DrawThingsSampler', 'clipSkip'],
    ['controls', null, 'DrawThingsControlNet', 'controls'],
    ['loras', null, 'DrawThingsLoRA', 'loras'],
    ['mask_blur', 'mask_blur', 'DrawThingsSampler', 'maskBlur'],
    ['face_restoration', null, null, 'faceRestoration'],
    ['decode_with_attention', null, null],
    ['hires_fix_decode_with_attention', null, null],
    ['clip_weight', null, null],
    ['negative_prompt_for_image_prior', null, null],
    ['image_prior_steps', null, null],

    // in refiner node
    ['refiner_model', 'refiner_model', 'DrawThingsRefiner', 'refinerModel'],

    ['original_image_height', null, null, 'originalImageHeight'],
    ['original_image_width', null, null, 'originalImageWidth'],
    ['crop_top', null, null, 'cropTop'],
    ['crop_left', null, null, 'cropLeft'],
    ['target_image_height', null, null, 'targetImageHeight'],
    ['target_image_width', null, null, 'targetImageWidth'],
    ['aesthetic_score', null, null, 'aestheticScore'],
    ['negative_aesthetic_score', null, null, 'negativeAestheticScore'],
    ['zero_negative_prompt', null, null, 'zeroNegativePrompt'],

    // in refiner node
    ['refiner_start', 'refiner_start', 'DrawThingsRefiner', 'refinerStart'],

    ['negative_original_image_height', null, null, 'negativeOriginalImageHeight'],
    ['negative_original_image_width', null, null, 'negativeOriginalImageWidth'],
    ['name', null, null, null],

    // check these
    ['fps_id', 'fps', 'DrawThingsSampler', 'fps'],
    ['motion_bucket_id', 'motion_scale', 'DrawThingsSampler', 'motionScale'],
    ['cond_aug', 'guiding_frame_noise', 'DrawThingsSampler', 'guidingFrameNoise'],
    ['start_frame_cfg', 'start_frame_guidance', 'DrawThingsSampler', 'startFrameGuidance'],

    ['num_frames', 'num_frames', 'DrawThingsSampler', 'numFrames'],
    ['mask_blur_outset', 'mask_blur_outset', 'DrawThingsSampler', 'maskBlurOutset'],
    ['sharpness', 'sharpness', 'DrawThingsSampler', 'sharpness'],
    ['shift', 'shift', 'DrawThingsSampler', 'shift'],
    ['stage_2_steps', null, null, 'stage2Steps'],
    ['stage_2_cfg', null, null, 'stage2Guidance'],
    ['stage_2_shift', null, null, 'stage2Shift'],
    ['tiled_decoding', 'tiled_decoding', 'DrawThingsSampler', 'tiledDecoding'],
    ['decoding_tile_width', 'decoding_tile_width', 'DrawThingsSampler', 'decodingTileWidth'],
    ['decoding_tile_height', 'decoding_tile_height', 'DrawThingsSampler', 'decodingTileHeight'],
    ['decoding_tile_overlap', 'decoding_tile_overlap', 'DrawThingsSampler', 'decodingTileOverlap'],
    ['stochastic_sampling_gamma', 'stochastic_sampling_gamma', 'stochasticSamplingGamma'],
    ['preserve_original_after_inpaint', 'preserve_original', 'DrawThingsSampler', 'preserveOriginalAfterInpaint'],
    ['tiled_diffusion', 'tiled_diffusion', 'DrawThingsSampler', 'tiledDiffusion'],
    ['diffusion_tile_width', 'diffusion_tile_width', 'DrawThingsSampler', 'diffusionTileWidth'],
    ['diffusion_tile_height', 'diffusion_tile_height', 'DrawThingsSampler', 'diffusionTileHeight'],
    ['diffusion_tile_overlap', 'diffusion_tile_overlap', 'DrawThingsSampler', 'diffusionTileOverlap'],

    // in upscaler node
    ['upscaler_scale_factor', 'upscaler_scale_factor', 'DrawThingsUpscaler', 'upscalerScaleFactor'],

    ['t5_text_encoder', null, null, 't5TextEncoder'],
    ['separate_clip_l', 'separate_clip_l', 'DrawThingsSampler', 'separateClipL'],
    ['clip_l_text', 'clip_l_text', 'DrawThingsSampler', null],
    ['separate_open_clip_g', 'separate_open_clip_g', 'DrawThingsSampler', 'separateOpenClipG'],
    ['open_clip_g_text', 'open_clip_g_text', 'DrawThingsSampler', null],
    ['speed_up_with_guidance_embed', 'speed_up', 'DrawThingsSampler', 'speedUpWithGuidanceEmbed'],
    ['guidance_embed', 'guidance_embed', 'DrawThingsSampler', 'guidanceEmbed'],
    ['resolution_dependent_shift', 'res_dpt_shift', 'DrawThingsSampler', 'resolutionDependentShift'],
    ['tea_cache_start', 'tea_cache_start', 'DrawThingsSampler', 'teaCacheStart'],
    ['tea_cache_end', 'tea_cache_end', 'DrawThingsSampler', 'teaCacheEnd'],
    ['tea_cache_threshold', 'tea_cache_threshold', 'DrawThingsSampler', 'teaCacheThreshold'],
    ['tea_cache', 'tea_cache', 'DrawThingsSampler', 'teaCache'],
    ['separate_t5', null, null, 'separateT5'],
    ['t5_text', null, null, null],
    ['tea_cache_max_skip_steps', 'tea_cache_max_skip_steps', 'DrawThingsSampler', 'teaCacheMaxSkipSteps'],

    // causal_inference_enabled is implied by causal_inference==0
    ['causal_inference_enabled', null, null, null],
    ['causal_inference', 'causal_inference', 'DrawThingsSampler', 'causalInference'],
]

/** @typedef {{ fbs: string, python: string, node: string, json: string }} DTProperty */

function roundBy64(value) {
    return Math.round(value / 64) * 64
}

/** @type {Record<string, (key, value, widget: import('@comfyorg/litegraph').IWidget, node, config) => void>} */
const importers = {
    model: async (k, v, w, n, c) => {
        await updateNodeModels(n)
        const matchingOption = w?.options?.values?.find(ov => ov.value?.file === v)
        if (matchingOption) w.value = matchingOption
    },
    start_width: (k, v, w) => {
        if (w) w.value = roundBy64(v)
    },
    start_height: (k, v, w) => {
        if (w) w.value = roundBy64(v)
    },
    sampler: (k, v, w) => {
        w.value = samplers[v]
    },
    seed: (k, v, w, n) => {
        if (typeof v === "number" && v >= 0) {
            w.value = v
            const controlWidget = n.widgets.find((w) => w.name === "control_after_generate")
            if (controlWidget) controlWidget.value = "fixed"
        }
    },
    hires_fix_start_width: (k, v, w) => {
        if (w) w.value = roundBy64(v)
    },
    hires_fix_start_height: (k, v, w) => {
        if (w) w.value = roundBy64(v)
    },
    seed_mode: (k, v, w) => {
        if (w && seedModes[v]) w.value = seedModes[v]
    },
    decoding_tile_width: (k, v, w) => {
        if (w) w.value = roundBy64(v)
    },
    decoding_tile_height: (k, v, w) => {
        if (w) w.value = roundBy64(v)
    },
    decoding_tile_overlap: (k, v, w) => {
        if (w) w.value = roundBy64(v)
    },
    diffusion_tile_width: (k, v, w) => {
        if (w) w.value = roundBy64(v)
    },
    diffusion_tile_height: (k, v, w) => {
        if (w) w.value = roundBy64(v)
    },
    diffusion_tile_overlap: (k, v, w) => {
        if (w) w.value = roundBy64(v)
    },
    guidance_embed: (k, v, w, n, c) => {
        if (w) w.value = v
    },
    resolution_dependent_shift: (k, v, w, n, c) => {
        if (w) w.value = v
        if (v) {
            const shiftWidget = n.widgets.find((w) => w.name === "shift")
            const width = c.width || n.widgets.find((w) => w.name === "width")?.value
            const height = c.height || n.widgets.find((w) => w.name === "height")?.value
            if (shiftWidget && width && height) shiftWidget.value = calcShift(width, height)
        }
    },
    causal_inference: (k, v, w, n, c) => {
        // only set if enabled in the config
        if (w && typeof v === 'number') w.value = v * 4 - 3
    },
}

const exporters = {
    // start_width: {},
    // start_height: {},
    model: async (w, n, c) => {
      if (w && w.value && w.value.value) c.model = w.value.value.file ?? w.value.value.name
    },
    sampler: (w, n, c) => {
        if (w && typeof w.value === 'string') c.sampler = samplers.indexOf(w.value)
    },
    // hires_fix_start_width: {},
    // hires_fix_start_height: {},
    seed_mode: (w, n, c) => {
        if (w && typeof w.value === 'string') c.seed_mode = seedModes.indexOf(w.value)
    },
    // decoding_tile_width: {},
    // decoding_tile_height: {},
    // decoding_tile_overlap: {},
    // diffusion_tile_width: {},
    // diffusion_tile_height: {},
    // diffusion_tile_overlap: {},
    // guidance_embed: (w, n, c) => {
    // },
    // resolution_dependent_shift: (w, n, c) => {
    // },
    // shift: (w, n, c) => {
    // },
    causal_inference: (w, n, c) => {
        if (w && typeof w.value === 'number') c.causal_inference = (w.value + 3) / 4
    },
}

class DTProperty {
    constructor(fbs, python, node, json) {
        this.fbs = fbs
        this.python = python
        this.node = node
        this.json = json
    }

    customImport = undefined
    customExport = undefined

    async import(jsonKey, jsonValue, widget, node, config) {
        if (this.customImport)
            return await this.customImport(jsonKey, jsonValue, widget, node, config)
        else {
            if (widget)
                widget.value = jsonValue
        }
    }

    async export(widget, node, config) {
        if (this.customExport)
            return await this.customExport(widget, node, config)
        else {
            if (this.json && widget && widget.value !== undefined) config[this.json] = widget.value
        }
    }
}

/** @type {DTProperty[]} */
export const properties = propertyData.map(([fbs, python, node, json]) => {
    const prop = new DTProperty(fbs, python, node, json)
    prop.customImport = importers[fbs]
    prop.customExport = exporters[fbs]
    return prop
})

export function findPropertyJson(name) {
    return properties.find(p => p.json === name)
}

export function findPropertyPython(name) {
    return properties.find(p => p.python === name)
}
