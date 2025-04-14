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

// NOTE: leaving this commented out for now in case we need it later
// const widgetParentsDT = ["high_res_fix"];
// function hideWidgetDT(widgetObject, hide) {
//     widgetObject.disabled = hide;
//     // widgetObject.hidden = hide;
// }

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

            // NOTE: leaving this commented out for now in case we need it later
            // const widgetsList = {};
            // node.widgets.forEach(listWidgets);
            // function listWidgets(widget) {
            //     widgetsList[widget.name] = widget;
            // }

            // widgetParentsDT.forEach(listParents);
            // function listParents(parent) {
            //     const widgetParent = widgetsList[parent];
            //     // console.log(widgetParent);

            //     widgetParent.callback = function () {
            //         for (let [name, child] of Object.entries(widgetsList)) {
            //             if (child.name.startsWith(parent + "_")) {
            //                 if (this.value == true) {
            //                     hideWidgetDT(child, false);
            //                 } else {
            //                     hideWidgetDT(child, true);
            //                 }
            //                 // console.log(child);
            //             }
            //         }
            //     };
            // }
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
    // NOTE: leaving this commented out for now in case we need it later
    // async afterConfigureGraph() {
    //     console.log(app);
    //     app.graph._nodes.forEach(listNodes);
    //     function listNodes(node) {
    //         if (node.type === "DrawThingsSampler") {
    //             const widgetsList = {};
    //             node.widgets.forEach(listWidgets);
    //             function listWidgets(widget) {
    //                 widgetsList[widget.name] = widget;
    //             }

    //             widgetParentsDT.forEach(listParents);
    //             function listParents(parent) {
    //                 const widgetParent = widgetsList[parent];
    //                 // console.log(widgetParent);

    //                 for (let [name, child] of Object.entries(widgetsList)) {
    //                     if (child.name.startsWith(parent + "_")) {
    //                         if (widgetParent.value == true) {
    //                             hideWidgetDT(child, false);
    //                         } else {
    //                             hideWidgetDT(child, true);
    //                         }
    //                         // console.log(child);
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // },
});

/** @import { LGraphNode, WidgetCallback, IWidget } from "litegraph.js"; */
