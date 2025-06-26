import * as App from "../../scripts/app.js"
import { updateProto } from "./util.js"

/** @type {import("@comfyorg/comfyui-frontend-types").ComfyApp} */
const app = App.app

app.registerExtension({
    name: "ComfyUI-DrawThings-gRPC-DtPromptNode",

    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "DrawThingsPrompt") {
            updateProto(nodeType, promptProto)
        }
    },
})


/** @type {import("@comfyorg/litegraph").LGraphNode} */
const promptProto = {
    onWidgetChanged(name, value, old_Value, widget) {
        if (name === "insert_textual_inversion") {
            const keyword = value?.value?.keyword
            if (!keyword) return
            const tag = `<${keyword}>`

            const textWidget = this.widgets.find((w) => w.name === "prompt")
            const text = textWidget?.value ?? ""

            if (text.includes(tag)) {
                textWidget.value = text.replaceAll(tag, "")
            }
            else {
                textWidget.value = `<${keyword}> ${text}`
            }

            widget.value = "..."
        }
    },

    onConnectionsChange(type, index, isConnected, link_info, inputOrOutput) {
        // if (isConnected && inputOrOutput.name === "PROMPT") {
        //     const node = this.graph.getNodeById(link_info.target_id)
        //     if (node.comfyClass === "DrawThingsSampler") {
        //         const input = node.inputs[link_info.target_slot]
        //         if (input.name === "positive") {
        //             this.color = LGraphCanvas.node_colors.green.color
        //             this.bgcolor = LGraphCanvas.node_colors.green.bgcolor
        //         }
        //         if (input.name === "negative") {
        //             this.color = LGraphCanvas.node_colors.red.color
        //             this.bgcolor = LGraphCanvas.node_colors.red.bgcolor
        //         }
        //     }
        // }
        let isPositive = false
        let isNegative = false

        for (const linkId of this.outputs[0]?.links ?? []) {
            const link = this.graph.getLink(linkId)
            const targetId = link?.target_id
            const targetNode = this.graph.getNodeById(targetId)
            if (targetNode?.comfyClass === "DrawThingsSampler") {
                const input = targetNode.inputs[link.target_slot]
                if (input?.name === "positive") isPositive = true
                if (input?.name === "negative") isNegative = true
            }
        }

        if (isPositive && isNegative) {
            this.color = LGraphCanvas.node_colors.purple.color
            this.bgcolor = LGraphCanvas.node_colors.purple.bgcolor
        }
        else if (isPositive) {
            this.color = LGraphCanvas.node_colors.green.color
            this.bgcolor = LGraphCanvas.node_colors.green.bgcolor
        }
        else if (isNegative) {
            this.color = LGraphCanvas.node_colors.red.color
            this.bgcolor = LGraphCanvas.node_colors.red.bgcolor
        }
        else {
            this.color = undefined
            this.bgcolor = undefined
        }
    },

    onNodeCreated() {
        const output = this.outputs.find((output) => output.name == "PROMPT")
        output.color_on = output.color_off = app.canvas.default_connection_color_byType["CONDITIONING"]

        const promptWidget = this.widgets.find(w => w.name === "prompt")
        const promptNode = this
        promptWidget.element.addEventListener('change', () => {
            promptNode.updateOptions()
        })
    }
}
