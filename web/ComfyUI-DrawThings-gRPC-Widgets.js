import { app } from "../../scripts/app.js";

let origProps = {};

function findWidgetByName(node, name) {
    return node.widgets.find((w) => w.name === name);
}

function doesInputWithNameExist(node, name) {
    return node.inputs ? node.inputs.some((input) => input.name === name) : false;
}

function toggleWidget(node, widget, show = false, suffix = "") {
    if (!widget || !doesInputWithNameExist(node, widget.name)) return;
    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize, origComputedHeight: widget.computedHeight };    
    }
    const origSize = node.size;

    widget.type = show ? origProps[widget.name].origType : "hidden" + suffix;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];
    widget.computedHeight = show ? origProps[widget.name].origComputedHeight : 0;

    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, ":" + widget.name, show));    

    const height = show ? Math.max(node.computeSize()[1], origSize[1]) : node.size[1];
    node.setSize([node.size[0], height]);
    app.canvas.dirty_canvas = true
}

function widgetLogic(node, widget) {
    switch (widget.name) {
        case "high_res_fix":
            if (widget.value === false) {
                toggleWidget(node, findWidgetByName(node, "high_res_fix_start_width"))
                toggleWidget(node, findWidgetByName(node, "high_res_fix_start_height"))
                toggleWidget(node, findWidgetByName(node, "high_res_fix_strength"))
            } else {
                toggleWidget(node, findWidgetByName(node, "high_res_fix_start_width"), true)
                toggleWidget(node, findWidgetByName(node, "high_res_fix_start_height"), true)
                toggleWidget(node, findWidgetByName(node, "high_res_fix_strength"), true)
            }
            break;

            case "tiled_decoding":
                if (widget.value === false) {
                    toggleWidget(node, findWidgetByName(node, "decoding_tile_width"))
                    toggleWidget(node, findWidgetByName(node, "decoding_tile_height"))
                    toggleWidget(node, findWidgetByName(node, "decoding_tile_overlap"))
                } else {
                    toggleWidget(node, findWidgetByName(node, "decoding_tile_width"), true)
                    toggleWidget(node, findWidgetByName(node, "decoding_tile_height"), true)
                    toggleWidget(node, findWidgetByName(node, "decoding_tile_overlap"), true)
                }
                break;
    
        case "tiled_diffusion":
            if (widget.value === false) {
                toggleWidget(node, findWidgetByName(node, "diffusion_tile_width"))
                toggleWidget(node, findWidgetByName(node, "diffusion_tile_height"))
                toggleWidget(node, findWidgetByName(node, "diffusion_tile_overlap"))
            } else {
                toggleWidget(node, findWidgetByName(node, "diffusion_tile_width"), true)
                toggleWidget(node, findWidgetByName(node, "diffusion_tile_height"), true)
                toggleWidget(node, findWidgetByName(node, "diffusion_tile_overlap"), true)
            }
            break;
    }
}

const getSetWidgets = [
    "high_res_fix",
    "tiled_decoding", 
    "tiled_diffusion",
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
