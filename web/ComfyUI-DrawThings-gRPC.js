import * as App from "../../scripts/app.js"
import { DtModelTypeHandler } from "./models.js"
import { updateProto, setCallback } from "./util.js"
import { findPropertyJson, findPropertyPython } from "./configProperties.js"

/** @type {import("@comfyorg/comfyui-frontend-types").ComfyApp} */
const app = App.app

const nodePackVersion = "1.3.0"
let previewMethod = undefined

// this holds the node definition from python
let dtSamplerNodeData = null

app.registerExtension({
    name: "DrawThings-gRPC-Main",
    getCustomWidgets(app) {
        return {
            DT_MODEL: DtModelTypeHandler,
        }
    },

    beforeConfigureGraph(graph) {
        for (const node of graph.nodes) {
            if (node.type === "DrawThingsPositive" || node.type === "DrawThingsNegative") {
                node.type = "DrawThingsPrompt"
                node.properties['Node name for S&R'] = "DrawThingsPrompt"
                delete node.properties.ver
                node.widgets_values.unshift("...")
            }
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "DrawThingsSampler") {
            updateProto(nodeType, samplerProto)
            dtSamplerNodeData = nodeData
        }
        if (nodeType.comfyClass === "DrawThingsPositive" || nodeType.comfyClass === "DrawThingsNegative") {
            updateProto(nodeType, promptProto)
        }
    },

    // beforeConfigureGraph: function (graph) {
    //     const samplerNodes = graph.nodes.filter(n => n.type === "DrawThingsSampler")
    //     if (samplerNodes.some(n => n.nodePackVersion !== nodePackVersion)) {
    //         console.log("Nodes in workflow are from different version of ComfyUI-DrawThings-gRPC")
    //     }
    // },

    async setup() {
        // query the api for the preview setting
        await updatePreviewSetting()

        // listen to the manager ui to update preview setting if necessary
        const closeHandler = async () => {
            await updatePreviewSetting()
            document.getElementById('cm-close-button')?.removeEventListener("click", closeHandler)
        }
        setTimeout(() => document.querySelector('button[title="ComfyUI Manager"]').addEventListener("click", async () => {
            document.getElementById('cm-close-button')?.addEventListener("click", closeHandler)
        }), 3000)

        // if the prompt is cancelled, send a signal to the server to cancel the grpc request
        setCallback(app.api, "interrupt", async (e) => {
            if (app.graph.nodes.some(n => n.type === "DrawThingsSampler")) {
                await app.api.fetchApi(`/dt_grpc_interrupt`, {
                    method: "POST",
                })
            }
        })
    }
})

/** @type {import("@comfyorg/litegraph").LGraphNode} */
const samplerProto = {
    async onNodeCreated() {
        const inputPos = this.inputs.find((inputPos) => inputPos.name == "positive")
        const inputNeg = this.inputs.find((inputNeg) => inputNeg.name == "negative")
        inputPos.color_on =
            inputPos.color_off =
            inputNeg.color_on =
            inputNeg.color_off =
            app.canvas.default_connection_color_byType["CONDITIONING"]
        app.canvas.default_connection_color_byType["DT_LORA"] = app.canvas.default_connection_color_byType["MODEL"]
        app.canvas.default_connection_color_byType["DT_CNET"] =
            app.canvas.default_connection_color_byType["CONTROL_NET"]
    },

    onMouseDown(e, pos, canvas) {
        // this exists for easier debugging in devtools
        console.debug("Click!", this, dtSamplerNodeData)
    },

    onSerialize(serialised) {
        serialised.nodePackVersion = nodePackVersion
        serialised.widget_values_keyed = Object.fromEntries(this.widgets.map(w => ([w.name, w.value])))
    },

    onConfigure(serialised) {
        // at this point, the node should already be loaded with values from the values array

        // if there is keyed data, apply that
        if (serialised.widget_values_keyed) {
            for (const [name, value] of Object.entries(serialised.widget_values_keyed)) {
                const widget = this.widgets.find((w) => w.name === name)
                if (widget) widget.value = value
            }
        }

        // check each widget value
        const corrections = []
        for (const w of this.widgets) {
            const prop = findPropertyPython(w.name)
            if (!prop) {
                continue
            }
            const coerced = prop.coerce(w.value)
            if (coerced !== w.value) {
                corrections.push({ name: w.name, value: w.value, coerced })
                w.value = coerced
            }
        }

        if (corrections.length) {
            const message = 'The Draw Things Sampler node contained invalid values - they have been corrected:'
            const list = corrections.map((c) => `${c.name}: ${c.value} -> ${c.coerced}`)
            const detail = message + "\n\n" + list.join("\n")

            app.extensionManager.toast.add({
                severity: "info",
                summary: "Draw Things gRPC",
                detail,
                life: 8000
            })
        }

        this.updateDynamicWidgets?.()
    },

    getExtraMenuOptions(canvas, options) {
        const keepNodeShrunk = app.extensionManager.setting.get("drawthings.node.keep_shrunk")
        options.push(
            null,
            {
                content: "Paste Draw Things config",
                callback: () => {
                    navigator.clipboard.readText().then(async (text) => {
                        try {
                            const config = JSON.parse(text)
                            for (const [k, v] of Object.entries(config)) {
                                const prop = findPropertyJson(k)
                                if (!prop) {
                                    console.log('unknown prop in config:', k)
                                    continue
                                }
                                const name = prop?.python
                                const widget = this.widgets.find((w) => w.name === name)
                                await prop.import(k, v, widget, this, config)
                            }
                            this.updateDynamicWidgets?.()
                        } catch (e) {
                            alert("Failed to parse Draw Things config from clipboard\n\n" + e)
                            console.warn(e)
                        }
                    })
                },
            },
            {
                content: "Copy Draw Things config",
                callback: () => {
                    const config = {}
                    for (const w of this.widgets) {
                        const prop = findPropertyPython(w.name)
                        if (!prop)
                            continue
                        prop.export(w, this, config)
                    }
                    config.loras = []
                    config.control = []
                    navigator.clipboard.writeText(JSON.stringify(config))
                },
            },
            {
                content: (keepNodeShrunk ? "✓ " : "") + "Keep node shrunk when widgets change",
                callback: async () => {
                    try {
                        await app.extensionManager.setting.set("drawthings.node.keep_shrunk", !keepNodeShrunk)
                    } catch (error) {
                        console.error(`Error changing setting: ${error}`)
                    }
                },
            },
            null
        )
    },
}


const promptProto = {
    async onNodeCreated() {
        // Some default node colours, available are:
        // black, blue, brown, cyan, green, pale_blue, purple, red, yellow
        if (this?.comfyClass === "DrawThingsPositive") {
            this.color = LGraphCanvas.node_colors.green.color
            this.bgcolor = LGraphCanvas.node_colors.green.bgcolor
            const output = this.outputs.find((output) => output.name == "POSITIVE")
            output.color_on = output.color_off = app.canvas.default_connection_color_byType["CONDITIONING"]
        }
        if (this?.comfyClass === "DrawThingsNegative") {
            this.color = LGraphCanvas.node_colors.red.color
            this.bgcolor = LGraphCanvas.node_colors.red.bgcolor
            const output = this.outputs.find((output) => output.name == "NEGATIVE")
            output.color_on = output.color_off = app.canvas.default_connection_color_byType["CONDITIONING"]
        }
    },
}


async function updatePreviewSetting() {
    const res = await app.api.fetchApi('/manager/preview_method')
    previewMethod = await res.text()

    const body = new FormData()
    body.append("preview", previewMethod)

    await app.api.fetchApi(`/dt_grpc_preview`, {
        method: "POST",
        body,
    })
}

/** @import { LGraphCanvas, LGraphNode, WidgetCallback, IWidget } from "litegraph.js"; */
