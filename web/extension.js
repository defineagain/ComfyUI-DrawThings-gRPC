import dtPrompt from "./dtPromptNode.js"
import dtCore from "./ComfyUI-DrawThings-gRPC.js"
import dtModelNodes from "./dtModelNodes.js"
import dtDynamicInputs from "./dynamicInputs.js"
import dtWidgets from "./widgets.js"

import * as App from "../../scripts/app.js"

/** @type {import("@comfyorg/comfyui-frontend-types").ComfyApp} */
const app = App.app

const modules = [dtCore, dtPrompt, dtModelNodes, dtDynamicInputs, dtWidgets]

// different features of the nodepack extension are implemented in different modules
// here we combine them and register a single extension
app.registerExtension({
    name: "DrawThings-gRPC",

    // beforeRegisterNodeDef(nodeType, nodeData, app) {
    //     for (const module of modules) {
    //         module.beforeRegisterNodeDef?.(nodeType, nodeData, app)
    //     }
    // },

    getCustomWidgets(...args) {
        return dtCore.getCustomWidgets(...args)
    },

    beforeConfigureGraph(...args) {
        for (const module of modules) {
            try { module.beforeConfigureGraph?.(...args) }
            catch (e) {
                console.error(`Error in ${module.name} beforeConfigureGraph:`, e)
            }
        }
    },
    beforeRegisterNodeDef(...args) {
        for (const module of modules) {
            try { module.beforeRegisterNodeDef?.(...args) }
            catch (e) {
                console.error(`Error in ${module.name} beforeConfigureGraph:`, e)
            }
        }
    },
    afterConfigureGraph(...args) {
        for (const module of modules) {
            try { module.afterConfigureGraph?.(...args) }
            catch (e) {
                console.error(`Error in ${module.name} afterConfigureGraph:`, e)
            }
        }
    },
    setup(...args) {
        for (const module of modules) {
            try { module.setup?.(...args) }
            catch (e) {
                console.error(`Error in ${module.name} beforeConfigureGraph:`, e)
            }
        }
    },

    settings: modules.flatMap(m => m.settings ?? []),
})
