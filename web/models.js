import { dtModelNodeTypes } from './dtModelNodes.js'
import * as App from "../../scripts/app.js"

/** @type {import("@comfyorg/comfyui-frontend-types").ComfyApp} */
const app = App.app

class ModelService {
    constructor() {

    }

    async updateNodes() {
        const dtModelNodes = app.graph.nodes.filter(n => n.isDtServerNode !== undefined)
        const serverNodes = dtModelNodes.filter(n => n.isDtServerNode)

        if (!serverNodes.length) return

        const nodesUpdated = new Map(dtModelNodes.map(n => ([n, false])))
        const serverModels = new Map()

        for (const sn of serverNodes) {
            nodesUpdated.set(sn, true)
            const { server, port, useTls } = sn.getServer()

            if (!server || !port || useTls === undefined) continue
            const key = modelInfoStoreKey(server, port, useTls)
            if (!serverModels.has(key)) {
                serverModels.set(key, await getModels(server, port, useTls))
            }

            const models = serverModels.get(key)

            /** @param {import("@comfyorg/litegraph").LGraphNode} node */
            function updateInputs(node, version) {
                nodesUpdated.set(node, true)
                node.updateModels?.(models, version)

                const inputNodes = getInputNodes(node)
                for (const input of inputNodes) {
                    if (nodesUpdated.get(input) === false) updateInputs(input, version)
                }
            }

            updateInputs(sn, sn.getModelVersion())
        }

        // these nodes are not connected to a sampler node
        // if there's a single server, update them with those models
        const models = serverModels.size === 1 ? serverModels.values().next().value : null
        // if there's a single sampler... well let's not disable by version because it's disconnected
        for (const node of nodesUpdated.keys()) {
            if (nodesUpdated.get(node) === false) {
                nodesUpdated.set(node, true)
                node.updateModels?.(models)
            }
        }
    }
}

export const modelService = new ModelService()

/**
 * @param node {LGraphNode}
 * @param inputName {string}
 * @param inputData {["DT_MODEL", {model_type: string}]}
 */
export function DtModelTypeHandler(node, inputName, inputData, app) {
    const widget = node.addWidget(
        "combo",
        inputName,
        "(None selected)",
        /** @type WidgetCallback<IWidget<any, any>> */
        (value, graph, node) => {
            node.saveSelectedModels?.()
            modelService.updateNodes()
        },
        {
            values: ["(None selected)"],
            modelType: inputData[1].model_type,
        }
    )

    return { widget }
}

/**
 * @param {string} server
 * @param {number | string} port
 */
export async function getFiles(server, port, useTls) {
    const body = new FormData()
    body.append("server", server)
    body.append("port", port)
    body.append("use_tls", useTls)

    const api = window.comfyAPI.api.api

    const filesInfoResponse = await api.fetchApi(`/dt_grpc_files_info`, {
        method: "POST",
        body,
    })

    return filesInfoResponse
}

/* @typedef {{ models: any[], controlNets: any[], loras: any[], upscalers: any[]}} ModelInfo */

/**
 * @typedef {Object} Model
 * @property {string} name
 * @property {string} file
 * @property {string} version
 */

/**
 * @typedef {Object} ModelInfo
 * @property {Model[]} models
 */

/** @type Map<string, ModelInfo> */
const modelInfoStore = new Map()
/** @type Map<string, Promise<void>> */
const modelInfoRequests = new Map()
const modelInfoStoreKey = (server, port, useTls) => `${server}:${port}${useTls ? ":tls" : ""}`

// yes this is kind of hacky :)
const failedConnectionOptions = ["Couldn't connect to server", "Check server and click to retry"].map((c, i) => ({
    name: c,
    version: "fail",
    order: i + 1,
}))
const notConnectedOptions = ["Not connected to sampler node", "Connect to a sampler node to list available models"].map(
    (c) => ({
        name: c,
        version: "fail",
    })
)

async function getModels(server, port, useTls) {
    if (!server || !port || useTls === undefined) return
    const key = modelInfoStoreKey(server, port, useTls)

    if (modelInfoRequests.has(key)) {
        const request = modelInfoRequests.get(key)
        await request
    }
    else {
        const promise = new Promise((resolve) => {
            getFiles(server, port, useTls).then(async (response) => {
                if (!response.ok) {
                    modelInfoStore.set(key, null)
                } else {
                    const data = await response.json()
                    modelInfoStore.set(key, data)
                }
                modelInfoRequests.delete(key)
                resolve()
            })
        })
        modelInfoRequests.set(key, promise)
        await promise
    }

    return modelInfoStore.get(key)
}

const failedConnectionInfo = {
    models: failedConnectionOptions,
    controlNets: notConnectedOptions,
    loras: notConnectedOptions,
    upscalers: notConnectedOptions,
    textualInversions: notConnectedOptions,
    isNotConnected: true,
}

modelInfoStore.set(modelInfoStoreKey(), failedConnectionInfo)
let fetches = 0
let updates = 0
/** @param node {LGraphNode} */
export async function updateNodeModelsX(node, updateDisconnected = false) {
    return
    // find the sampler node
    let root = findRoot(node)
    if (!root) {
        if (!updateDisconnected) return
        root = node
    }

    // get the server and port
    const server = root.widgets.find((w) => w.name === "server")?.value
    const port = root.widgets.find((w) => w.name === "port")?.value
    const useTls = root.widgets.find((w) => w.name === "use_tls")?.value

    const key = modelInfoStoreKey(server, port)

    if (modelInfoRequests.has(key)) {
        const request = modelInfoRequests.get(key)
        await request
    }
    // fetch the models list and update store
    else if (server && port) {
        const promise = new Promise((resolve) => {
            console.debug("checking DT server", key, " (", fetches++, ")")
            getFiles(server, port, useTls).then(async (response) => {
                if (!response.ok) {
                    modelInfoStore.set(key, failedConnectionInfo)
                } else {
                    const data = await response.json()
                    modelInfoStore.set(key, data)
                }
                modelInfoRequests.delete(key)
                resolve()
            })
        })
        modelInfoRequests.set(key, promise)
        await promise
    }

    const models = modelInfoStore.get(key)
    const samplerModel = root.widgets.find((w) => w.options?.modelType === "models")?.value
    const version = samplerModel?.value?.version

    const modelInfo = getModelOptions(models, version)

    // update any connected nodes
    updateInputs(root)

    /** @param dtNode {LGraphNode} */
    function updateInputs(dtNode, version) {
        if (!dtNode) return

        updateModelWidgets(dtNode, modelInfo)

        // const inputs = dtNode.inputs.filter(isDtModelInput)
        // for (const input of inputs) {
        //     const slot = dtNode.findInputSlot(input.name)
        //     const inputNode = dtNode.getInputNode(slot)
        //     updateInputs(inputNode)
        // }

        const inputNodes = getInputNodes(dtNode)
        for (const inputNode of inputNodes) {
            if (dtModelNodeTypes.includes(inputNode.comfyClass))
                updateInputs(inputNode, version)
        }
    }
}

/** @param node {LGraphNode} */
function getInputNodes(node) {
    // return typeof input?.type === 'string' && input.type.startsWith("DT_");

    return node.inputs.map((input, i) => ([i, input]))
        .filter(([index, input]) => input.link !== null)
        .map(([index, input]) => node.getInputNode(index))
}

/** @param node {LGraphNode}; @param models {{ models: any[], controlNets: any[], loras: any[], upscalers: any[]}} */
function updateModelWidgets(node, models) {
    if (!node || !models) return
    const modelWidgets = node.widgets.filter((w) => w.options?.modelType)

    for (const widget of modelWidgets) {
        const type = widget.options.modelType

        widget.options.values = models[type]
        setValidOption(widget, node, models.isNotConnected)
    }
}

function getModelOptions(modelInfo, version) {
    function toOptions(modelGroup, disableByVersion = false) {
        return [
            {
                content: "None selected",
                value: {
                    name: "None selected",
                    file: null,
                    version: null,
                },
                toString() {
                    return "None selected"
                },
            },
            ...modelGroup
                .map((m) => getMenuItem(m, disableByVersion && version && m.version && m.version !== version))
                .sort((a, b) => {
                    if (a.value?.version === "fail" && b.value?.version === "fail") return 0
                    if (a.disabled && !b.disabled) return 1
                    if (!a.disabled && b.disabled) return -1
                    return a.content.toUpperCase().localeCompare(b.content.toUpperCase())
                }),
        ]
    }

    const models = toOptions(modelInfo.models)
    const controlNets = toOptions(modelInfo.controlNets, true)
    const loras = toOptions(modelInfo.loras, true)
    const upscalers = toOptions(modelInfo.upscalers) //  modelInfo.upscalers.map((m) => `${m.name}`).sort();
    const textualInversions = toOptions(modelInfo.textualInversions, true)
    const isNotConnected = modelInfo.isNotConnected

    return { models, controlNets, loras, upscalers, textualInversions, isNotConnected }
}

export function getMenuItem(model, disabled) {
    return {
        value: model,
        content: model.version && model.version !== "fail" ? `${model.name} (${getVersionAbbrev(model.version)})` : model.name,
        toString() {
            return model.name
        },
        // has_submenu?: boolean;
        disabled,
        // submenu?: IContextMenuSubmenu<TValue>;
        // property?: string;
        // type?: string;
        // slot?: IFoundSlot;
        // callback(this: ContextMenuDivElement<TValue>, value?: TCallbackValue, options?: unknown, event?: MouseEvent, previous_menu?: ContextMenu<TValue>, extra?: TExtra) {
        callback(...args) {
            return false
        },
    }
}

const modelComparator = (a, b) => a.version?.localeCompare(b.version) || a.name.localeCompare(b.name)

const versionNames = {
    v1: 'SD',
    v2: 'SD2',
    'kandinsky2.1': 'Kan',
    'sdxl_base_v0.9': 'SDXL',
    'sdxl_refiner_v0.9': 'SDXL',
    ssd_1b: 'SSD',
    svd_i2v: 'SVD',
    'wurstchen_v3.0_stage_c': 'Wur',
    'wurstchen_v3.0_stage_b': 'Wur',
    sd3: 'SD3',
    pixart: 'Pix',
    auraflow: 'AF',
    flux1: 'F1',
    sd3_large: 'SD3L',
    hunyuan_video: 'Hun',
    'wan_v2.1_1.3b': 'Wan',
    'wan_v2.1_14b': 'Wan',
    hidream_i1: 'HiD',
}
function getVersionAbbrev(version) {
    return versionNames[version] ?? version
}

function setValidOption(widget, node, isNotConnected) {
    if (!widget || widget.type !== "combo") return
    const values = widget.options?.values
    const selected = widget?.value

    if (selected?.value?.toString() === "[object Object]") {
        const value = {
            ...selected,
            toString() {
                return this.value.name
            },
        }
        widget.value = value
    }

    // const option = failedConnectionOptions.find((o) => o.name === selected);
    // if (option) {
    //     widget.value = values[0];
    // }

    if (!isNotConnected && selected?.value?.version === "fail") {
        // server is no connected, so switch from a 'fail' option to the last selected model, or none
        const lastSelected = node?.lastSelectedModel[widget.name]
        if (lastSelected) {
            const option = values.find((o) => o.content === lastSelected.content)
            if (option) widget.value = option
            return
        }

        widget.value = values[0]
    }
    // debugger;
    if (isNotConnected) {
        // unless "none selected", always select couldn't connect to server
        if (selected?.value?.content !== "None selected") {
            const option = values.find((o) => o.content === "Couldn't connect to server")
            if (option) widget.value = option
        }
    }
}

/** @param {LGraphNode} node */
function findRoot(node) {
    if (!node || node.id === -1) return

    if (node?.isDtServerNode === true) return node

    if (node?.isDtServerNode === false) {
        // this isn't necesarrily a safe assumption but it is for now
        for (const output of node.outputs.filter((o) => o.type.startsWith("DT_"))) {
            const outputSlot = node.findOutputSlot(output.name)
            const outputNodes = node.getOutputNodes(outputSlot) ?? []
            for (const outputNode of outputNodes) {
                const root = findRoot(outputNode)
                if (root) return root
            }
        }
    }

    return undefined
}

export function findModel(option, type) {
    const name = extractModelName(option)

    for (const info of modelInfoStore.values()) {
        const model = info[type].find((m) => m.name === name)
        if (model) return model
    }

    return undefined
}

function extractModelName(option) {
    if (typeof option === "string") {
        const matches = option.match(/^(.*) \(.+\)$/)
        return matches ? matches[1] : option
    }
    if (typeof option === "object") {
        if ("content" in option) return option.content
        if ("name" in option) return option.name
    }
    return option
}

/** @import { LGraphNode, WidgetCallback, IWidget, IComboWidget } from "@comfyorg/litegraph"; */
