import { app } from "../../scripts/app.js";
import { setCallback } from "./dynamicInputs.js";
import { updateNodeModels } from "./models.js";
import { updateProto } from "./util.js";

const dtModelNodeTypes = ["DrawThingsSampler", "DrawThingsControlNet", "DrawThingsLoRA", "DrawThingsUpscaler"];
const dtServerNodeTypes = ["DrawThingsSampler"];

app.registerExtension({
    name: "ComfyUI-DrawThings-gRPC-DtModelNodes",
    beforeRegisterNodeDef: (nodeType, nodeData, app) => {
        if (dtModelNodeTypes.includes(nodeType.comfyClass)) {
            updateProto(nodeType, dtModelNodeProto);
            if (dtServerNodeTypes.includes(nodeType.comfyClass)) {
                updateProto(nodeType, dtServerNodeProto);
            } else {
                updateProto(nodeType, dtExtraNodeProto);
            }
        }
    },
});

/** @type {import("@comfyorg/litegraph").LGraphNode} */
const dtModelNodeProto = {
    saveSelectedModels() {
        const modelWidgets = this.widgets.filter((w) => w.options?.modelType);
        const selections = modelWidgets.reduce((acc, w) => {
            if (w.value?.value?.version !== "fail") acc[w.name] = w.value;
            else acc[w.name] = this._lastSelectedModel?.[w.name];
            return acc;
        }, {});

        this._lastSelectedModel = selections;
    },
    lastSelectedModel: {
        get() {
            return this._lastSelectedModel;
        },
        enumerable: true,
    },
    isDtServerNode: {
        get() {
            return dtServerNodeTypes.includes(this?.comfyClass);
        },
        enumerable: true,
    },
    onSerialize(serialised) {
        serialised._lastSelectedModel = JSON.parse(JSON.stringify(this._lastSelectedModel ?? {}))
    },
    onConfigure(serialised) {
        this._lastSelectedModel = serialised._lastSelectedModel || {};

        for (const widget of this.widgets.filter((w) => w.options?.modelType)) {
            if (widget.value?.toString() === "[object Object]") {
                const value = {
                    ...widget.value,
                    toString() {
                        return this.value.name;
                    },
                };
                widget.setValue(value);
            }
        }
    },

};

/** @type {import("@comfyorg/litegraph").LGraphNode} */
const dtServerNodeProto = {
    onNodeCreated() {
        // update when server or port changes
        const serverWidget = this.widgets.find((w) => w.name === "server");
        if (serverWidget) setCallback(serverWidget, "callback", () => updateNodeModels(this));

        const portWidget = this.widgets.find((w) => w.name === "port");
        if (portWidget) setCallback(portWidget, "callback", () => updateNodeModels(this));

        console.log("added with " + serverWidget.value);
    },

    onConfigure() {
        updateNodeModels(this);
    },

    getServer() {
        const server = this.widgets.find((w) => w.name === "server")?.value;
        const port = this.widgets.find((w) => w.name === "port")?.value;
        return { server, port };
    },

    getModelVersion() {
        return this.widgets.find((w) => w.options?.modelType === "models")?.value?.value?.version;
    },
};

/** @type {import("@comfyorg/litegraph").LGraphNode} */
const dtExtraNodeProto = {
    onConnectionsChange(...args) {
        const isConnected = args[2];
        if (isConnected) updateNodeModels(this);
    },
};
