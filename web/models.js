/** @type {import("@comfyorg/comfyui-frontend-types").ComfyApp} */
const app = window.comfyAPI.app.app

class ModelService {
    #updateNodesPromise = null

    constructor() {
    }

    async updateNodes() {
        // since many nodes may be configured at once, we will batch calls to updateNodes
        if (!this.#updateNodesPromise) {
            this.#updateNodesPromise = new Promise(res => {
                setTimeout(() => {
                    this.#updateNodesPromise = null
                    res(this.#updateNodes())
                }, 10)
            })
        }

        return this.#updateNodesPromise
    }

    async #updateNodes() {
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
    console.log("add dt model widget")
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

    const filesInfoResponse = await api.fetchApi(`/dt_grpc/files_info`, {
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

/** @param node {LGraphNode} */
function getInputNodes(node) {
    return node.inputs.map((input, i) => ([i, input]))
        .filter(([index, input]) => input.link !== null)
        .map(([index, input]) => node.getInputNode(index))
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


/** @import { LGraphNode, WidgetCallback, IWidget, IComboWidget } from "@comfyorg/litegraph"; */
