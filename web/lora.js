import { DtModelTypeHandler } from './models.js'
import { updateProto } from "./util.js"
import { showWidget } from './widgets.js'

/**
 * so I don't really see an official way to remove widgets other than .splice
 * which seems weird, because then you have to call onRemove() yourself
 * So I'm going to go with "create 8 of them and hide the ones you don't need"
 * If someone needs more than 8 lora, well, they are crazy, and also they can
 * just chain another node
 *
 * we'll use model = null for loras above count
 * distinct from model = "(None selected)"
 *
 * .loraCount prop will 'add' and 'remove' (hide/show) widgets when changed
 * serialize and deserialize it with the node
 *
 * and we'll need a callback on each of the lora models so we can add control
 * image inputs if necessary
 *
 * the buttons widget is a custom widget type, so it's defined in the python
 * code, event listeners attached by the type handler
 */

/**
 * @param node {LGraphNode}
 * @param inputName {string}
 * @param inputData {["DT_MODEL", {model_type: string}]}
 */
export function DtButtonsTypeHandler(node, inputName, inputData, app) {
    const { container, buttons } = createButtons([{
        label: "Less",
        callback: () => {
            node.loraCount -= 1
        },
    }, {
        label: "More",
        callback: () => {
            node.loraCount += 1
        },
    }])
    const options = {
        hideOnZoom: false,
        getValue: () => undefined,
        setValue: (value) => { },
        getMinHeight: () => 36,
        getMaxHeight: () => 36,
        getHeight: () => 36,
        margin: 4
    }
    const widget = node.addDOMWidget("something", 'DT_BUTTONS', container, options)
    widget._buttonElements = buttons
    return { widget }
}

/** @type {import("@comfyorg/litegraph").LGraphNode} */
const loraProto = {
    onNodeCreated(graph) {
        this.loraCount = 1
    },

    onConfigure(serialised) {
        if ("loraCount" in serialised)
            this.loraCount = serialised.loraCount
        console.log(serialised)
    },

    onSerialize(serialised) {
        serialised.loraCount = this._loraCount
    },

    loraCount: {
        get() {
            return this._loraCount
        },
        set(count) {
            if (this._loraCount === count) return
            this._loraCount = Math.max(0, Math.min(count, 8))
            this.updateWidgets()

            const buttons = this.widgets[0]._buttonElements
            buttons[0].disabled = this._loraCount <= 1
            buttons[1].disabled = this._loraCount >= 8
        },
        enumerable: true
    },

    updateWidgets() {
        console.log("updating lora widgets")
        for (let i = 0; i < 8; i++) {
            const modelIndex = i * 2 + 1
            const weightIndex = i * 2 + 2
            if (i < this.loraCount) {
                console.log("show", this.widgets[modelIndex].name)
                showWidget(this, this.widgets[modelIndex].name, true)
                showWidget(this, this.widgets[weightIndex].name, true)
                if (!this.widgets[modelIndex].value) {
                    this.widgets[modelIndex].value = "(None selected)"
                }
            } else {
                console.log("hide", this.widgets[modelIndex].name)
                showWidget(this, this.widgets[modelIndex].name, false)
                showWidget(this, this.widgets[weightIndex].name, false)

                this.widgets[modelIndex].value = null
            }
        }
    }
}

/** @type {import("@comfyorg/comfyui-frontend-types").ComfyExtension}*/
export default {
    name: "loraNode",

    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "DrawThingsLoRA") {
            updateProto(nodeType, loraProto)
        }
    }
}


/**
 * @typedef ButtonDef
 * @property {string?} label
 * @property {node => void} callback
 * @property {string?} style
 * @property {string[]?} classList
 */

/** @param {ButtonDef[]} buttonsDefs */
function createButtons(buttonsDefs) {
    const container = document.createElement("div")
    container.classList.add("dt-buttons-container")
    const buttons = []

    for (const buttonDef of buttonsDefs) {
        const button = document.createElement("button")
        buttons.push(button)
        button.classList.add("dt-button")
        if (buttonDef.label)
            button.innerText = buttonDef.label
        button.addEventListener("click", () => {
            buttonDef.callback()
        })
        if (buttonDef.style) {
            button.style.cssText = buttonDef.style
        }
        if (buttonDef.classList) {
            button.classList.add(...buttonDef.classList)
        }
        container.appendChild(button)
    }

    return { container, buttons }
}
