import { app } from "../../scripts/app.js"
import { DtModelTypeHandler } from "./models.js"
import { updateProto, getWidgetName } from "./util.js"
import { findPropertyJson } from "./configProperties.js"

// Include the name of any nodes to have their DT_MODEL inputs updated
const DrawThingsNodeTypes = ["DrawThingsSampler", "DrawThingsControlNet", "DrawThingsLoRA", "DrawThingsUpscaler"]

app.registerExtension({
    name: "ComfyUI-DrawThings-gRPC",
    getCustomWidgets(app) {
        return {
            DT_MODEL: DtModelTypeHandler,
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "DrawThingsSampler") {
            updateProto(nodeType, samplerProto)
        }
        if (nodeType.comfyClass === "DrawThingsPositive" || nodeType.comfyClass === "DrawThingsNegative") {
            updateProto(nodeType, promptProto)
        }
    },
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
        console.debug("Click!", this)
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

                            this.updateDynamicWidgets?.();

                        } catch (e) {
                            alert("Failed to parse Draw Things config from clipboard\n\n" + e)
                            console.warn(e)
                        }
                    })
                },
            },
            // {
            //     content: "Copy Draw Things config",
            //     callback: () => {
            //         const config = {};
            //         for (const w of this.widgets) {
            //             if (["settings", "server", "port", "use_tls"].includes(w.name)) continue;

            //             if (w.name === "model") {
            //                 config[w.name] = w.value?.value?.file;
            //                 continue;
            //             }

            //             const propertyName = getDTPropertyName(w.name)
            //             config[propertyName ?? w.name] = w.value;
            //         }
            //         config.loras = []
            //         config.control = []
            //         navigator.clipboard.writeText(JSON.stringify(config));
            //     },
            // },
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

export const samplers = [
    "DPM++ 2M Karras",
    "Euler A",
    "DDIM",
    "PLMS",
    "DPM++ SDE Karras",
    "UniPC",
    "LCM",
    "Euler A Substep",
    "DPM++ SDE Substep",
    "TCD",
    "Euler A Trailing",
    "DPM++ SDE Trailing",
    "DPM++ 2M AYS",
    "Euler A AYS",
    "DPM++ SDE AYS",
    "DPM++ 2M Trailing",
    "DDIM Trailing",
]

export const seedModes = ["Legacy", "TorchCpuCompatible", "ScaleAlike", "NvidiaGpuCompatible"];

/** @import { LGraphCanvas, LGraphNode, WidgetCallback, IWidget } from "litegraph.js"; */
