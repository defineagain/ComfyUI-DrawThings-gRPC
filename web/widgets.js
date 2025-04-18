import { app } from "../../scripts/app.js";
import { setCallback } from "./dynamicInputs.js";

const allWidgets = [
    "settings",
    "server",
    "port",
    "model",
    "strength",
    "seed",
    "seed_mode",
    "width",
    "height",
    "steps",
    // "num_frames", // For now not touching widgets that are hidden/shown by other widgets
    "cfg",
    // "speed_up",
    "sampler_name",
    // "res_dpt_shift",
    "shift",
    // zero neg
    // sep clip
    "clip_skip",
    "sharpness",
    "mask_blur",
    "mask_blur_outset",
    "preserve_original",
    // ti embed
];
const basicWidgets = [
    "server",
    "port",
    // "model",
    "strength",
    "seed",
    "width",
    "height",
    "steps",
    "cfg",
    "sampler_name",
    // "res_dpt_shift",
    "shift",
];
const advancedWidgets = [
    "seed_mode",
    // "speed_up",
    // zero neg
    // sep clip
    "clip_skip",
    "sharpness",
    "mask_blur",
    "mask_blur_outset",
    "preserve_original",
    // ti embed
];

const getSetWidgets = [
    "settings",
    "model",
    "high_res_fix",
    "tiled_decoding",
    "tiled_diffusion",
    "tea_cache",
];
const getSetTypes = ["DrawThingsSampler"];

let origProps = {};

function findWidgetByName(node, name) {
    return node.widgets.find((w) => w.name === name);
}

function doesInputWithNameExist(node, name) {
    return node.inputs ? node.inputs.some((input) => input.name === name) : false;
}

function showWidget(node, widget, show = false, suffix = "") {
    widget = findWidgetByName(node, widget);
    if (!widget || !doesInputWithNameExist(node, widget.name)) return;
    if (!origProps[widget.name]) {
        origProps[widget.name] = {
            origType: widget.type,
            origComputeSize: widget.computeSize,
            origComputedHeight: widget.computedHeight,
        };
    }
    const origSize = node.size;

    widget.type = show ? origProps[widget.name].origType : "hidden" + suffix;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];
    widget.computedHeight = show ? origProps[widget.name].origComputedHeight : 0;

    widget.linkedWidgets?.forEach((w) => showWidget(node, w, ":" + widget.name, show));

    const height = show ? Math.max(node.computeSize()[1], origSize[1]) : node.size[1];
    node.setSize([node.size[0], height]);
    app.canvas.dirty_canvas = true;
}

function widgetLogic(node, widget) {
    switch (widget.name) {
        case "settings":
            // First show all, then either hide basic or advanced
            allWidgets.forEach(listAllWidgets);
            function listAllWidgets(listedWidget) {
                console.log(listedWidget);
                showWidget(node, listedWidget, true);
            }

            switch (widget.value) {
                case "Basic":
                    // Hide advanced
                    advancedWidgets.forEach(listAdvancedWidgets);
                    function listAdvancedWidgets(listedWidget) {
                        showWidget(node, listedWidget, false);
                    }
                    break;
                case "Advanced":
                    // Hide basic
                    basicWidgets.forEach(listBasicWidgets);
                    function listBasicWidgets(listedWidget) {
                        showWidget(node, listedWidget, false);
                    }
                    break;
            }
            break;
        case "model":
            const selectedModel = widget.value;
            const version = selectedModel?.value?.version;
            if (!version) break;

            let isSD3 = version === "sd3";
            let isFlux = version === "flux1";
            let isVideo = ["svdI2v", "Video", "wan"].includes(version);

            if (isFlux === false && isVideo === false) {
                showWidget(node, "tea_cache", false);
                showWidget(node, "tea_cache_start", false);
                showWidget(node, "tea_cache_end", false);
                showWidget(node, "tea_cache_threshold", false);
            } else {
                showWidget(node, "tea_cache", true);
                if (findWidgetByName(node, "tea_cache").value === false) {
                    showWidget(node, "tea_cache_start", false);
                    showWidget(node, "tea_cache_end", false);
                    showWidget(node, "tea_cache_threshold", false);
                } else {
                    showWidget(node, "tea_cache_start", true);
                    showWidget(node, "tea_cache_end", true);
                    showWidget(node, "tea_cache_threshold", true);
                }
            }

            if (isSD3 === false && isFlux === false) {
                showWidget(node, "res_dpt_shift", false);
            } else {
                showWidget(node, "res_dpt_shift", true);
            }

            showWidget(node, "speed_up", isFlux);
            showWidget(node, "num_frames", isVideo);
            break;

        case "high_res_fix":
            showWidget(node, "high_res_fix_start_width", widget.value);
            showWidget(node, "high_res_fix_start_height", widget.value);
            showWidget(node, "high_res_fix_strength", widget.value);
            break;

        case "tiled_decoding":
            showWidget(node, "decoding_tile_width", widget.value);
            showWidget(node, "decoding_tile_height", widget.value);
            showWidget(node, "decoding_tile_overlap", widget.value);
            break;

        case "tiled_diffusion":
            showWidget(node, "diffusion_tile_width", widget.value);
            showWidget(node, "diffusion_tile_height", widget.value);
            showWidget(node, "diffusion_tile_overlap", widget.value);
            break;

        case "tea_cache":
            showWidget(node, "tea_cache_start", widget.value);
            showWidget(node, "tea_cache_end", widget.value);
            showWidget(node, "tea_cache_threshold", widget.value);
            break;
    }
}

/** @param {import("@comfyorg/litegraph").LGraphNode} node */
function getSetters(node) {
    if (node.widgets) {
        for (const w of node.widgets) {
            if (getSetWidgets.includes(w.name)) {
                widgetLogic(node, w);
                let widgetValue = w.value;

                // Define getters and setters for widget values
                Object.defineProperty(w, "value", {
                    get() {
                        return widgetValue;
                    },
                    set(newVal) {
                        if (newVal !== widgetValue) {
                            widgetValue = newVal;
                            widgetLogic(node, w);
                        }
                    },
                });

                // changing the built-in properties might have unexpected results
                // but you can use the widget callback, which fires any time the value changes

                // w.callback = function (value, canvas, node, pos, event) {
                //     widgetLogic(node, w);
                // });
            }
        }
    }
}

app.registerExtension({
    name: "ComfyUI-DrawThings-gRPC-Widgets",

    async nodeCreated(node) {
        const nodeType = node.constructor.type;
        if (getSetTypes.includes(nodeType)) {
            getSetters(node);
        }
    },
    async afterConfigureGraph() {
        app.graph._nodes.forEach(listNodes);
        function listNodes(node) {
            const nodeType = node.constructor.type;
            if (getSetTypes.includes(nodeType)) {
                getSetters(node);
            }
        }
    },
});
