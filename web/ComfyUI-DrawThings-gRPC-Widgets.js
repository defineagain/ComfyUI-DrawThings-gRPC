import { app } from "../../scripts/app.js";

let origProps = {};

function findWidgetByName(node, name) {
    return node.widgets.find((w) => w.name === name);
}

function doesInputWithNameExist(node, name) {
    return node.inputs ? node.inputs.some((input) => input.name === name) : false;
}

function showWidget(node, widget, show = false, suffix = "") {
    if (!widget || !doesInputWithNameExist(node, widget.name)) return;
    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize, origComputedHeight: widget.computedHeight };    
    }
    const origSize = node.size;

    widget.type = show ? origProps[widget.name].origType : "hidden" + suffix;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];
    widget.computedHeight = show ? origProps[widget.name].origComputedHeight : 0;

    widget.linkedWidgets?.forEach(w => showWidget(node, w, ":" + widget.name, show));    

    const height = show ? Math.max(node.computeSize()[1], origSize[1]) : node.size[1];
    node.setSize([node.size[0], height]);
    app.canvas.dirty_canvas = true
}

function widgetLogic(node, widget) {
    switch (widget.name) {
        case "model":
            const modelName = widget.value;
            if (modelName === false) { break; }

            let isSD3 = false;
            if (modelName.includes("sd3")) {
                isSD3 = true;
            }
            let isFlux = false;
            if (modelName.includes("flux")) {
                isFlux = true;
            }
            let isVideo = false;
            if (
                modelName.includes("svdI2v") ||
                modelName.includes("Video") ||
                modelName.includes("wan")
            ) {
                isVideo = true;
            }

            if (isFlux === false && isVideo === false) {
                // findWidgetByName(node, "tea_cache").value = false; // is this needed?
                showWidget(node, findWidgetByName(node, "tea_cache"), false)
                showWidget(node, findWidgetByName(node, "tea_cache_start"), false)
                showWidget(node, findWidgetByName(node, "tea_cache_end"), false)
                showWidget(node, findWidgetByName(node, "tea_cache_threshold"), false)
            } else {
                showWidget(node, findWidgetByName(node, "tea_cache"), true)
                if (findWidgetByName(node, "tea_cache").value === false) {
                    showWidget(node, findWidgetByName(node, "tea_cache_start"), false)
                    showWidget(node, findWidgetByName(node, "tea_cache_end"), false)
                    showWidget(node, findWidgetByName(node, "tea_cache_threshold"), false)
                } else {
                    showWidget(node, findWidgetByName(node, "tea_cache_start"), true)
                    showWidget(node, findWidgetByName(node, "tea_cache_end"), true)
                    showWidget(node, findWidgetByName(node, "tea_cache_threshold"), true)
                }
            }
            if (isSD3 === false && isFlux === false) {
                showWidget(node, findWidgetByName(node, "res_dpt_shift"), false)
            } else {
                showWidget(node, findWidgetByName(node, "res_dpt_shift"), true)
            }
            if (isFlux === false) {
                showWidget(node, findWidgetByName(node, "speed_up"), false)
            } else {
                showWidget(node, findWidgetByName(node, "speed_up"), true)
            }
            if (isVideo === false) {
                showWidget(node, findWidgetByName(node, "num_frames"), false)
            } else {
                showWidget(node, findWidgetByName(node, "num_frames"), true)
            }
            break;

        case "high_res_fix":
            if (widget.value === false) {
                showWidget(node, findWidgetByName(node, "high_res_fix_start_width"), false)
                showWidget(node, findWidgetByName(node, "high_res_fix_start_height"), false)
                showWidget(node, findWidgetByName(node, "high_res_fix_strength"), false)
            } else {
                showWidget(node, findWidgetByName(node, "high_res_fix_start_width"), true)
                showWidget(node, findWidgetByName(node, "high_res_fix_start_height"), true)
                showWidget(node, findWidgetByName(node, "high_res_fix_strength"), true)
            }
            break;

        case "tiled_decoding":
            if (widget.value === false) {
                showWidget(node, findWidgetByName(node, "decoding_tile_width"), false)
                showWidget(node, findWidgetByName(node, "decoding_tile_height"), false)
                showWidget(node, findWidgetByName(node, "decoding_tile_overlap"), false)
            } else {
                showWidget(node, findWidgetByName(node, "decoding_tile_width"), true)
                showWidget(node, findWidgetByName(node, "decoding_tile_height"), true)
                showWidget(node, findWidgetByName(node, "decoding_tile_overlap"), true)
            }
            break;

        case "tiled_diffusion":
            if (widget.value === false) {
                showWidget(node, findWidgetByName(node, "diffusion_tile_width"), false)
                showWidget(node, findWidgetByName(node, "diffusion_tile_height"), false)
                showWidget(node, findWidgetByName(node, "diffusion_tile_overlap"), false)
            } else {
                showWidget(node, findWidgetByName(node, "diffusion_tile_width"), true)
                showWidget(node, findWidgetByName(node, "diffusion_tile_height"), true)
                showWidget(node, findWidgetByName(node, "diffusion_tile_overlap"), true)
            }
            break;

        case "tea_cache":
            if (widget.value === false) {
                showWidget(node, findWidgetByName(node, "tea_cache_start"), false)
                showWidget(node, findWidgetByName(node, "tea_cache_end"), false)
                showWidget(node, findWidgetByName(node, "tea_cache_threshold"), false)
            } else {
                showWidget(node, findWidgetByName(node, "tea_cache_start"), true)
                showWidget(node, findWidgetByName(node, "tea_cache_end"), true)
                showWidget(node, findWidgetByName(node, "tea_cache_threshold"), true)
            }
            break;
    }
}

const getSetWidgets = [
    "server",
    "port",
    "model",
    "strength",
    "seed_mode",

    "high_res_fix",
    "tiled_decoding",
    "tiled_diffusion",
    "tea_cache"
]
const getSetTitles = [
    "DrawThingsSampler",
];

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
                    }
                });
            }
        }
    }
}

app.registerExtension({
    name: "ComfyUI-DrawThings-gRPC-Widgets",
    
    nodeCreated(node) {
        const nodeTitle = node.constructor.title;
        if (getSetTitles.includes(nodeTitle)) {
            getSetters(node);
        }
    },
    async afterConfigureGraph() {
        app.graph._nodes.forEach(listNodes);
        function listNodes(node) {
            const nodeTitle = node.constructor.type;
            if (getSetTitles.includes(nodeTitle)) {
                getSetters(node);
            }
        }
    },
});
