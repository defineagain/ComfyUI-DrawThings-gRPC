import { app } from "../../scripts/app.js";
import { addServerListeners, DtModelTypeHandler, updateNodeModels } from "./models.js";

// Include the name of any nodes to have their DT_MODEL inputs updated
const DrawThingsNodeTypes = ["DrawThingsSampler", "DrawThingsControlNet", "DrawThingsLoRA", "DrawThingsUpscaler"];

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
        if (node?.comfyClass === "DrawThingsPositive") {
            node.color = LGraphCanvas.node_colors.green.color;
            node.bgcolor = LGraphCanvas.node_colors.green.bgcolor;
        }
        if (node?.comfyClass === "DrawThingsNegative") {
            node.color = LGraphCanvas.node_colors.red.color;
            node.bgcolor = LGraphCanvas.node_colors.red.bgcolor;
        }
    },
    async loadedGraphNode(node) {
        if (DrawThingsNodeTypes.includes(node?.comfyClass) && node?.isDtServerNode) {
            updateNodeModels(node);
        }
    },
    /** @param nodeType {typeof LGraphNode} */
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (DrawThingsNodeTypes.includes(nodeType.comfyClass)) {
            // is this a 'root' node (ie - it has a server and port)
            if ("server" in nodeData.input.required && "port" in nodeData.input.required) {
                nodeType.prototype.isDtServerNode = true;
                nodeType.prototype.getServer = function () {
                    const server = this.widgets.find((w) => w.name === "server")?.value;
                    const port = this.widgets.find((w) => w.name === "port")?.value;
                    return { server, port };

                    // todo - update proto instead of adding listeners to each new instance
                };

                nodeType.prototype.getModelVersion = function () {
                    return this.widgets.find((w) => w.options?.modelType === "models")?.value?.value?.version;
                };
            }
            // support or stacker nodes
            else {
                nodeType.prototype.isDtServerNode = false;
                // check if uses the dynamic DT_MODEL type
                const allInputs = Object.values({
                    ...nodeData.input.required,
                    ...nodeData.input.optional,
                });
                if (allInputs.some(([type]) => type === "DT_MODEL")) {
                    const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;

                    nodeType.prototype.onConnectionsChange = function (...args) {
                        const r = originalOnConnectionsChange?.apply(this, args);
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
                if (node.widgets.some((w) => w.options.values === "DT_MODEL")) updateNodeModels(node, true);
            }
        }
    },
});

/** @import { LGraphCanvas, LGraphNode, WidgetCallback, IWidget } from "litegraph.js"; */
