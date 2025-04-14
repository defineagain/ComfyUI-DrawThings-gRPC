import { app } from "../../scripts/app.js";
import {
    addServerListeners,
    DtModelTypeHandler,
    updateNodeModels,
} from "./models.js";

// Include the name of any nodes to have their DT_MODEL inputs updated
const DrawThingsNodeTypes = [
    "DrawThingsSampler",
    "DrawThingsControlNet",
    "DrawThingsLoRA",
    "DrawThingsUpscaler",
    "DrawThingsLoRANet",
];

// const widgetParentsDT = ["high_res_fix"];
function hideWidgetDT(widgetObject, hide) {
    widgetObject.disabled = hide;
    // widgetObject.hidden = hide;
}

app.registerExtension({
    name: "ComfyUI-DrawThings-gRPC",
    getCustomWidgets(app) {
        return {
            DT_MODEL: DtModelTypeHandler,
        };
    },
    async nodeCreated(node) {
        if (DrawThingsNodeTypes.includes(node?.comfyClass)) {
            updateNodeModels(node);
        }
        if (node?.comfyClass === "DrawThingsSampler") {
            addServerListeners(node);

            const original_onMouseDown = node.onMouseDown;
            node.onMouseDown = function (e, pos, canvas) {
                console.debug("Click!", node);
                return original_onMouseDown?.apply(this, arguments);
            };
        }
        if (node?.comfyClass === "DrawThingsTiled") {
            const widgetList = {};
            node.widgets.forEach(listWidgets);
            function listWidgets(widget) {
                widgetList[widget.name] = widget;
            };
            widgetList["tiled_decoding"].callback = function () {
                if (widgetList["tiled_decoding"].value == true) {
                    hideWidgetDT(widgetList["decoding_tile_width"], false);
                    hideWidgetDT(widgetList["decoding_tile_height"], false);
                    hideWidgetDT(widgetList["decoding_tile_overlap"], false);
                } else {
                    hideWidgetDT(widgetList["decoding_tile_width"], true);
                    hideWidgetDT(widgetList["decoding_tile_height"], true);
                    hideWidgetDT(widgetList["decoding_tile_overlap"], true);
                }
            };
            widgetList["tiled_diffusion"].callback = function () {
                if (widgetList["tiled_diffusion"].value == true) {
                    hideWidgetDT(widgetList["diffusion_tile_width"], false);
                    hideWidgetDT(widgetList["diffusion_tile_height"], false);
                    hideWidgetDT(widgetList["diffusion_tile_overlap"], false);
                } else {
                    hideWidgetDT(widgetList["diffusion_tile_width"], true);
                    hideWidgetDT(widgetList["diffusion_tile_height"], true);
                    hideWidgetDT(widgetList["diffusion_tile_overlap"], true);
                }
            };
        }
    },
    async loadedGraphNode(node) {
        if (
            DrawThingsNodeTypes.includes(node?.comfyClass) &&
            node?.isDtRootNode
        ) {
            updateNodeModels(node);
        }
    },
    /** @param nodeType {typeof LGraphNode} */
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (DrawThingsNodeTypes.includes(nodeType.comfyClass)) {
            // is this a 'root' node (ie - it has a server and port)
            if (
                "server" in nodeData.input.required &&
                "port" in nodeData.input.required
            ) {
                nodeType.prototype.isDtRootNode = true;
                // todo - update proto instead of adding listeners to each new instance
            }
            // support or stacker nodes
            else {
                nodeType.prototype.isDtRootNode = false;
                // check if uses the dynamic DT_MODEL type
                const allInputs = Object.values({
                    ...nodeData.input.required,
                    ...nodeData.input.optional,
                });
                if (allInputs.some(([type]) => type === "DT_MODEL")) {
                    const originalOnConnectionsChange =
                        nodeType.prototype.onConnectionsChange;

                    nodeType.prototype.onConnectionsChange = function (
                        ...args
                    ) {
                        const r = originalOnConnectionsChange?.apply(
                            this,
                            args
                        );
                        const isConnected = args[2];
                        if (isConnected) updateNodeModels(this);
                        return r;
                    };
                }
            }
        }
    },
    refreshComboInNodes(defs, app) {
        for (const type of DrawThingsNodeTypes) {
            for (const node of app.graph.findNodesByType(type)) {
                if (node.widgets.some((w) => w.options.values === "DT_MODEL"))
                    updateNodeModels(node, true);
            }
        }
    },
    async afterConfigureGraph() {
        const widgetList = [];
        app.graph._nodes.forEach(listNodes);
        function listNodes(node) {
            if (node.type === "DrawThingsTiled") {
                node.widgets.forEach(listWidgets);
                function listWidgets(widget) {
                    widgetList[widget.name] = widget;
                    // console.log(widget);
                }
                if (widgetList["tiled_decoding"].value == true) {
                    hideWidgetDT(widgetList["decoding_tile_width"], false);
                    hideWidgetDT(widgetList["decoding_tile_height"], false);
                    hideWidgetDT(widgetList["decoding_tile_overlap"], false);
                } else {
                    hideWidgetDT(widgetList["decoding_tile_width"], true);
                    hideWidgetDT(widgetList["decoding_tile_height"], true);
                    hideWidgetDT(widgetList["decoding_tile_overlap"], true);
                }
                if (widgetList["tiled_diffusion"].value == true) {
                    hideWidgetDT(widgetList["diffusion_tile_width"], false);
                    hideWidgetDT(widgetList["diffusion_tile_height"], false);
                    hideWidgetDT(widgetList["diffusion_tile_overlap"], false);
                } else {
                    hideWidgetDT(widgetList["diffusion_tile_width"], true);
                    hideWidgetDT(widgetList["diffusion_tile_height"], true);
                    hideWidgetDT(widgetList["diffusion_tile_overlap"], true);
                }
            }
        }
    },
});

/** @import { LGraphNode, WidgetCallback, IWidget } from "litegraph.js"; */
