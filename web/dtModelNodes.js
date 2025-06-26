import { app } from "../../scripts/app.js"
import { setCallback } from "./dynamicInputs.js"
import { modelService, getMenuItem } from "./models.js"
import { updateProto } from "./util.js"

export const dtModelNodeTypes = ["DrawThingsSampler", "DrawThingsControlNet", "DrawThingsLoRA", "DrawThingsUpscaler", "DrawThingsRefiner", "DrawThingsPrompt"]
export const dtServerNodeTypes = ["DrawThingsSampler"]

app.registerExtension({
    name: "ComfyUI-DrawThings-gRPC-DtModelNodes",
    beforeRegisterNodeDef: (nodeType, nodeData, app) => {
        if (dtModelNodeTypes.includes(nodeType.comfyClass)) {
            updateProto(nodeType, dtModelNodeProto)
            if (dtServerNodeTypes.includes(nodeType.comfyClass)) {
                updateProto(nodeType, dtServerNodeProto)
            } else if (nodeType.comfyClass === "DrawThingsPrompt") {
                updateProto(nodeType, dtModelPromptNodeProto)
            }
            else {
                updateProto(nodeType, dtModelStandardNodeProto)
            }
        }
    },

    afterConfigureGraph() {
        modelService.updateNodes()
    }
})

/** @type {import("@comfyorg/litegraph").LGraphNode} */
const dtModelNodeProto = {
    saveSelectedModels() {
        const modelWidgets = this.widgets.filter((w) => w.options?.modelType)
        const selections = modelWidgets.reduce((acc, w) => {
            if (typeof (w.value) === "object" || w.value === "(None selected)") acc[w.name] = w.value
            else acc[w.name] = this._lastSelectedModel?.[w.name]
            return acc
        }, {})

        this._lastSelectedModel = selections
    },
    lastSelectedModel: {
        get() {
            return this._lastSelectedModel
        },
        enumerable: true,
    },
    isDtServerNode: {
        get() {
            return dtServerNodeTypes.includes(this?.comfyClass)
        },
        enumerable: true,
    },
    onSerialize(serialised) {
        serialised._lastSelectedModel = JSON.parse(JSON.stringify(this._lastSelectedModel ?? {}))
    },
    onConfigure(serialised) {
        this._lastSelectedModel = serialised._lastSelectedModel || {}
    },
    getModelWidget() {
        return this.widgets.find((w) => w.options?.modelType)
    },
    onAdded() {
        modelService.updateNodes()
    }
}

/** @type {import("@comfyorg/litegraph").LGraphNode} */
const dtServerNodeProto = {
    onNodeCreated() {
        // update when server or port changes
        const serverWidget = this.widgets.find((w) => w.name === "server")
        if (serverWidget) setCallback(serverWidget, "callback", () => modelService.updateNodes())

        const portWidget = this.widgets.find((w) => w.name === "port")
        if (portWidget) setCallback(portWidget, "callback", () => modelService.updateNodes())

        const tlsWidget = this.widgets.find((w) => w.name === "use_tls")
        if (tlsWidget) setCallback(tlsWidget, "callback", () => modelService.updateNodes())
    },

    getServer() {
        const server = this.widgets.find((w) => w.name === "server")?.value
        const port = this.widgets.find((w) => w.name === "port")?.value
        const useTls = this.widgets.find((w) => w.name === "use_tls")?.value
        return { server, port, useTls }
    },

    getModelVersion() {
        return this.widgets.find((w) => w.options?.modelType === "models")?.value?.value?.version
    },

    /**
     * @param {import("./models.js").ModelInfo} models
     * @param {string} version
     */
    updateModels(models, version) {
        const widget = this.widgets.find(w => w.name === "model")
        if (models === null) {
            widget.options.values = ["Not connected", "Click to retry"]
            widget.value = "Not connected"
            return
        }

        widget.options.values = ["(None selected)", ...models.models.map(m => getMenuItem(m)).sort((a, b) => a.content.localeCompare(b.content))]

        if (widget.value === "Click to retry" || widget.value === "Not connected") {
            if (this._lastSelectedModel?.model) widget.value = this._lastSelectedModel.model
            else widget.value = "(None selected)"
        }

        if (widget.value?.toString() === "[object Object]") {
            const value = {
                ...widget.value,
                toString() {
                    return this.value.name
                },
            }
            widget.value = value
        }
    }
}

// for cnet, lora, upscaler, and refiner
/** @type {import("@comfyorg/litegraph").LGraphNode} */
const dtModelStandardNodeProto = {
    onConnectionsChange(type, index, isConnected, link_info, inputOrOutput) {
        if (isConnected) modelService.updateNodes()
    },

    /**
     * @param {import("./models.js").ModelInfo} models
     * @param {string} version
     */
    updateModels(models, version) {
        /** @type {import('@comfyorg/litegraph').IWidget} */
        const widget = this.getModelWidget()
        const type = widget?.options?.modelType

        if (models === null) {
            widget.options.values = ["Not connected", "Click to retry"]
            widget.value = "Not connected"
            return
        }

        if (!(type in models)) return

        widget.options.values = ["(None selected)", ...models[type]
            .map((m) => getMenuItem(m, version && m.version && m.version !== version))
            .sort((a, b) => {
                if (a.disabled && !b.disabled) return 1
                if (!a.disabled && b.disabled) return -1
                return a.content.toUpperCase().localeCompare(b.content.toUpperCase())
            })]

        if (widget.value === "Click to retry" || widget.value === "Not connected") {
            if (this._lastSelectedModel?.model) widget.value = this._lastSelectedModel.model
            else widget.value = "(None selected)"
        }

        if (widget.value?.toString() === "[object Object]") {
            const value = {
                ...widget.value,
                toString() {
                    return this.value.name
                },
            }
            widget.value = value
        }
    }
}

const dtModelPromptNodeProto = {
    onConnectionsChange(type, index, isConnected, link_info, inputOrOutput) {
        if (isConnected) modelService.updateNodes()
    },

    updateModels(models, version) {
        this._models = models?.textualInversions || null
        this._version = version
        this.updateOptions()
    },

    updateOptions() {
        /** @type {import('@comfyorg/litegraph').IWidget} */
        const widget = this.getModelWidget()

        if (this._models === null) {
            widget.options.values = ["Not connected", "Click to retry"]
            widget.value = "Not connected"
            return
        }

        /** @type {string} */
        const promptText = this.widgets.find((w) => w.name === "prompt")?.value
        const matches = [...promptText.matchAll(/<(.*?)>/gm)]
        const tags = matches.map(m => m[1])
        widget.options.values = ["...", ...this._models
            .map((m) => getMenuItem(m, this._version && m.version && m.version !== this._version && !tags.includes(m.keyword)))
            .map(m => {
                Object.defineProperty(m, 'content', {
                    get() {
                        return `${tags.includes(m.value.keyword) ? "✓ " : ""}${m.value.name} (${m.value.version})`
                    }
                })
                return m
            })
            .sort((a, b) => {
                if (a.disabled && !b.disabled) return 1
                if (!a.disabled && b.disabled) return -1
                return a.content.toUpperCase().localeCompare(b.content.toUpperCase())
            })]

        widget.value = "..."
    },
}
