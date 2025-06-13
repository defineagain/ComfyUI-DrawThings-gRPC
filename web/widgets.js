/** @import { INodeInputSlot, LGraphNode } from '@comfyorg/litegraph' */
import { app } from "../../scripts/app.js";
import { updateProto } from "./util.js";

const basicWidgets = [
    "server",
    "port",
    "use_tls",
    // "model",
    "strength",
    "seed",
    "control_after_generate",
    "width",
    "height",
    "steps",
    "cfg",
    "sampler_name",
    "res_dpt_shift",
    "shift",
    "batch_size",
    "num_frames",
];
const advancedWidgets = [
    "seed_mode",
    "speed_up",
    "guidance_embed",
    "fps",
    "motion_scale",
    "guiding_frame_noise",
    "start_frame_guidance",
    "causal_inference",
    // zero neg
    // sep clip
    "clip_skip",
    "sharpness",
    "mask_blur",
    "mask_blur_outset",
    "preserve_original",
    "separate_clip_l",
    "clip_l_text",
    "separate_open_clip_g",
    "open_clip_g_text",
    "high_res_fix",
    "high_res_fix_start_width",
    "high_res_fix_start_height",
    "high_res_fix_strength",
    "upscaler",
    "image_guidance_scale",
    "tiled_decoding",
    "decoding_tile_width",
    "decoding_tile_height",
    "decoding_tile_overlap",
    "tiled_diffusion",
    "diffusion_tile_width",
    "diffusion_tile_height",
    "diffusion_tile_overlap",
    "tea_cache",
    "tea_cache_start",
    "tea_cache_end",
    "tea_cache_threshold",
    "tea_cache_max_skip_steps",
];

let origProps = {};

// From flux-auto-workflow.js
export function calcShift(h, w) {
    const step1 = (h * w) / 256;
    const step2 = (1.15 - 0.5) / (4096 - 256);
    const step3 = (step1 - 256) * step2;
    const step4 = step3 + 0.5;
    const result = Math.exp(step4);
    return Math.round(result * 100) / 100;
}

function findWidgetByName(node, name) {
    return node.widgets.find((w) => w.name === name);
}

function doesInputWithNameExist(node, name) {
    return node.inputs ? node.inputs.some((input) => input.name === name) : false;
}

/**
 * this replace the .pos property on input slots with a getter that returns its
 * normal position only if its associated widget is not hidden. if its widget is
 * hidden, it returns the collapsedPos instead
 * @param {INodeInputSlot?} input
 * @param {LGraphNode} node
 */
function updateInput(input, node) {
    if (!input || !input.widget || !node) return;
    if (input._origPos === undefined) {
        input._origPos = input.pos;
        // input._isHidden = false;

        const widget = node.getWidgetFromSlot(input)

        Object.defineProperty(input, "pos", {
            get() {
                if (widget?.type?.startsWith("hidden")) {
                    return this.collapsedPos;
                } else {
                    return input._origPos;
                }
            },
            set(value) {
                input._origPos = value;
            },
        });
    }
}

/** @param node {LGraphNode} */
function showWidget(node, widgetName, show = false, suffix = "") {
    const widget = findWidgetByName(node, widgetName);
    if (!widget) return;
    if (!origProps[widget.name]) {
        origProps[widget.name] = {
            origType: widget.type,
            origComputeSize: widget.computeSize,
            origComputedHeight: widget.computedHeight,
        };
    }

    const isVisible = !widget.type.startsWith("hidden");
    if (isVisible === show) return;

    widget.type = show ? origProps[widget.name].origType : "hidden" + suffix;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];
    widget.computedHeight = show ? origProps[widget.name].origComputedHeight : 0;

    widget.linkedWidgets?.forEach((w) => showWidget(node, w, ":" + widget.name, show));

    const minHeight = node.computeSize()[1];
    if (minHeight > node.size[1]) node.setSize([node.size[0], minHeight]);

    if (app.extensionManager.setting.get("drawthings.node.keep_shrunk") && minHeight < node.size[1])
        node.setSize([node.size[0], minHeight]);

    setTimeout(() => app.canvas.setDirty(true, true), 10);
}

function showWidgets(node, show, ...widgetNames) {
    widgetNames.forEach((w) => showWidget(node, w, show));
}

function updateSamplerWidgets(node) {
    const selectedModel = findWidgetByName(node, "model")?.value;
    const version = selectedModel?.value?.version ?? "none";

    const settings = findWidgetByName(node, "settings")?.value;
    const isBasic = settings === "Basic" || settings === "All";
    const isAdvanced = settings === "Advanced" || settings === "All";

    showWidgets(node, isBasic, ...basicWidgets);
    showWidgets(node, isAdvanced, ...advancedWidgets);

    /**
     * A list of versions can be found here:
     * https://github.com/drawthingsai/draw-things-community/blob/6f03f7d4a200ffeb6fdc6022a6ee579e4e534831/Libraries/SwiftDiffusion/Sources/Samplers/Sampler.swift#L4
     * "v1", "v2", "kandinsky2.1", "sdxl_base_v0.9", "sdxl_refiner_v0.9", "ssd_1b", "svd_i2v",
     * "wurstchen_v3.0_stage_c", "wurstchen_v3.0_stage_b", "sd3", "pixart", "auraflow", "flux1",
     * "sd3_large", "hunyuan_video", "wan_v2.1_1.3b", "wan_v2.1_14b", "hidream_i1"
     */

    if (isBasic) {
        // res_dpt_shift (flux, sd3, hidream)
        const resDPTShiftAvailable = ["flux1", "sd3", "hidream_i1"].includes(version);
        showWidget(node, "res_dpt_shift", resDPTShiftAvailable);
        const shiftDisabled = resDPTShiftAvailable && findWidgetByName(node, "res_dpt_shift")?.value;
        findWidgetByName(node, "shift").disabled = shiftDisabled;

        // num_frames (wan, hunyuan, svd)
        const isVideo = ["hunyuan_video", "wan_v2.1_1.3b", "wan_v2.1_14b", "svd_i2v"].includes(version);
        showWidget(node, "num_frames", isVideo);
    }

    if (isAdvanced) {
        // tea cache (for hidream, wan, hunyuan)
        const teaCacheAvailable = ["flux1", "hidream_i1", "wan_v2.1_1.3b", "wan_v2.1_14b", "hunyuan_video"].includes(
            version
        );
        const teaCacheEnabled = teaCacheAvailable && findWidgetByName(node, "tea_cache")?.value;
        showWidget(node, "tea_cache", teaCacheAvailable);
        showWidgets(
            node,
            teaCacheEnabled,
            "tea_cache_start",
            "tea_cache_end",
            "tea_cache_threshold",
            "tea_cache_max_skip_steps"
        );

        // speed up (flux, hidream, hunyuan)
        const speedUpAvailable = ["flux1", "hidream_i1", "hunyuan_video"].includes(version);
        showWidget(node, "speed_up", speedUpAvailable);
        const speedUpEnabled = findWidgetByName(node, "speed_up")?.value;
        showWidget(node, "guidance_embed", speedUpAvailable && !speedUpEnabled);

        // separate clip l (flux, hidream, sd3)
        const separateClipLAvailable = ["flux1", "hidream_i1", "sd3"].includes(version);
        showWidget(node, "separate_clip_l", separateClipLAvailable);
        const separateClipLEnabled = separateClipLAvailable && findWidgetByName(node, "separate_clip_l")?.value;
        showWidget(node, "clip_l_text", separateClipLEnabled);

        // separate open clip g (sd3)
        const separateOpenClipGAvailable = ["sd3"].includes(version);
        showWidget(node, "separate_open_clip_g", separateOpenClipGAvailable);
        const separateOpenClipGEnabled =
            separateOpenClipGAvailable && findWidgetByName(node, "separate_open_clip_g")?.value;
        showWidget(node, "open_clip_g_text", separateOpenClipGEnabled);

        // svd
        const isSvd = ["svd_i2v"].includes(version);
        showWidgets(node, isSvd, "fps", "motion_scale", "guiding_frame_noise", "start_frame_guidance");

        // causal_inference (just for wan I think)
        const causalInferenceAvailable = ["wan_v2.1_1.3b", "wan_v2.1_14b"].includes(version);
        showWidget(node, "causal_inference", causalInferenceAvailable);

        // high res fix
        const hiResFixEnabled = findWidgetByName(node, "high_res_fix")?.value;
        showWidgets(
            node,
            hiResFixEnabled,
            "high_res_fix_start_width",
            "high_res_fix_start_height",
            "high_res_fix_strength"
        );

        // tiled decoding
        const tiledDecodingEnabled = findWidgetByName(node, "tiled_decoding")?.value;
        showWidgets(node, tiledDecodingEnabled, "decoding_tile_width", "decoding_tile_height", "decoding_tile_overlap");

        // tiled diffusion
        const tiledDiffusionEnabled = findWidgetByName(node, "tiled_diffusion")?.value;
        showWidgets(
            node,
            tiledDiffusionEnabled,
            "diffusion_tile_width",
            "diffusion_tile_height",
            "diffusion_tile_overlap"
        );
    }
}

app.registerExtension({
    name: "ComfyUI-DrawThings-gRPC-Widgets",

    settings: [
        {
            id: "drawthings.node.keep_shrunk",
            type: "boolean",
            name: "Keep node shrunk",
            default: false,
            category: ["DrawThings", "Nodes", "Keep node shrunk"],
            onChange: (newVal, oldVal) => {
                if (oldVal === false && newVal === true) {
                    app.graph.nodes
                        .filter((n) => n.type === "DrawThingsSampler")
                        .forEach((n) => {
                            setTimeout(() => showWidget(n, "server", true), 10);
                        });
                }
            },
        },
    ],

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "DrawThingsSampler") {
            updateProto(nodeType, samplerWidgetsProto);
        }
        if (nodeType.comfyClass === "DrawThingsControlNet") {
            updateProto(nodeType, controlWidgetsProto);
        }
    },
});

/** @type {import("@comfyorg/litegraph").LGraphNode} */
const samplerWidgetsProto = {
    updateDynamicWidgets() {
        updateSamplerWidgets(this)
    },
    onNodeCreated() {
        this.updateDynamicWidgets();
    },
    onConfigure(data) {
        this.updateDynamicWidgets()
    },
    onInputAdded(input) {
        if (input.widget) updateInput(input, this)
    },
    onWidgetChanged(name, value, old_Value, widget) {
        this.updateDynamicWidgets();

        if (name === "res_dpt_shift") {
            const resDPTShiftAvailable = ["flux1", "sd3", "hidream_i1"].includes(this.getModelVersion());
            const resDptShiftEnabled = resDPTShiftAvailable && findWidgetByName(this, "res_dpt_shift")?.value;

            if (resDptShiftEnabled) {
                const height = findWidgetByName(this, "height").value;
                const width = findWidgetByName(this, "width").value;
                findWidgetByName(this, "shift").value = calcShift(height, width);
            }
        }
    },
};

function updateControlWidgets(node) {
    const widget = findWidgetByName(node, "control_name");
    const modifier = widget.value.value.modifier;
    const typeWidget = findWidgetByName(node, "control_input_type");
    const options = typeWidget.options.values;
    const option = options.find((option) => option.toLowerCase() == modifier);
    if (option != null) {
        typeWidget.value = option;
    }
}

/** @type {import("@comfyorg/litegraph").LGraphNode} */
const controlWidgetsProto = {
    onNodeCreated() {
        updateControlWidgets(this);
    },
    onConfigure(data) {
        updateControlWidgets(this);
    },
    onWidgetChanged(name, value, old_Value, widget) {
        updateControlWidgets(this, widget);
    },
};
