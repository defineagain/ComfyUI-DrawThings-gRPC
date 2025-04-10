/** @import { LGraphNode, WidgetCallback, IWidget } from "litegraph.js"; */

/** @param node {LGraphNode} */
export function addServerListeners(node) {
    const serverWidget = node.widgets.find((w) => w.name === "server");
    serverWidget.callback = function (value, graph, node) {
        updateNodeModels(node);
    };

    const portWidget = node.widgets.find((w) => w.name === "port");
    portWidget.callback = function (value, graph, node) {
        updateNodeModels(node);
    };
}

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
            const option = failedConnectionOptions.find(
                (o) => o.name === value
            );
            if (option) updateNodeModels(node);
        },
        {
            values: failedConnectionOptions.map((o) => o.name),
            modelType: inputData[1].model_type,
        }
    );
    return { widget };
}

/**
 * @param {string} server
 * @param {number | string} port
 */
export async function getFiles(server, port) {
    const body = new FormData();
    body.append("server", server);
    body.append("port", port);

    const api = window.comfyAPI.api.api;

    const filesInfoResponse = await api.fetchApi(`/dt_grpc_files_info`, {
        method: "POST",
        body,
    });

    return filesInfoResponse;
}

/** @typedef {{ models: any[], controlNets: any[], loras: any[], upscalers: any[]}} ModelInfo */
/** @type Map<string, ModelInfo> */
const modelInfoStore = new Map();
/** @type Map<string, Promise<void>> */
const modelInfoRequests = new Map();
const modelInfoStoreKey = (server, port) => `${server}:${port}`;

// yes this is kind of hacky :)
const failedConnectionOptions = [
    { name: "No connection. Check server and try again" },
    { name: "Click to retry" },
];

const failedConnectionInfo = {
    models: failedConnectionOptions,
    controlNets: failedConnectionOptions,
    loras: failedConnectionOptions,
};

modelInfoStore.set(modelInfoStoreKey(), failedConnectionInfo);
let fetches = 0;

/** @param node {LGraphNode} */
export async function updateNodeModels(node, updateDisconnected = false) {
    // find the sampler node
    let root = findRoot(node);
    if (!root) {
        if (!updateDisconnected) return;
        root = node;
    }

    // get the server and port
    const server = root.widgets.find((w) => w.name === "server")?.value;
    const port = root.widgets.find((w) => w.name === "port")?.value;

    const key = modelInfoStoreKey(server, port);

    if (modelInfoRequests.has(key)) {
        const request = modelInfoRequests.get(key);
        await request;
    }
    // fetch the models list and update store
    else if (server && port) {
        const promise = new Promise((resolve) => {
            console.debug("checking DT server", key, " (", fetches++, ")");
            getFiles(server, port).then(async (response) => {
                if (!response.ok) {
                    modelInfoStore.set(key, failedConnectionInfo);
                } else {
                    const data = await response.json();
                    modelInfoStore.set(key, data);
                }
                modelInfoRequests.delete(key);
                resolve();
            });
        });
        modelInfoRequests.set(key, promise);
        const response = await promise;
    }

    // update the node's models list
    const modelInfo = modelInfoStore.get(key);

    // update any connected nodes
    updateInputs(root);

    /** @param node {LGraphNode} */
    function updateInputs(dtNode) {
        if (!dtNode) return;

        updateModelWidgets(dtNode, modelInfo);

        const inputs = dtNode.inputs.filter((i) => i.type.startsWith("DT_"));
        for (const input of inputs) {
            const slot = dtNode.findInputSlot(input.name);
            const inputNode = dtNode.getInputNode(slot);
            updateInputs(inputNode);
        }
    }
}

/** @param node {LGraphNode}; @param models {{ models: any[], controlNets: any[], loras: any[], upscalers: any[]}} */
function updateModelWidgets(node, models) {
    if (!node || !models) return;
    const modelWidgets = node.widgets.filter((w) => w.options.modelType);

    for (const widget of modelWidgets) {
        const type = widget.options.modelType;
        widget.options.values = models[type]?.map((m) => m.name);
        setValidOption(widget);
    }
}

function setValidOption(widget) {
    if (!widget || widget.type !== "combo") return;
    const values = widget.options?.values;
    const selected = widget?.value;

    // to avoid clearing a selected model, only changing value if a 'failure' option
    // is selected
    const option = failedConnectionOptions.find((o) => o.name === selected);
    if (option) {
        widget.value = values[0];
    }
}

/** @param {LGraphNode} node */
function findRoot(node) {
    if (!node || node.id === -1) return;

    if (node?.isDtRootNode === true) return node;

    if (node?.isDtRootNode === false) {
        // this isn't necesarrily a safe assumption but it is for now
        for (const output of node.outputs.filter((o) =>
            o.type.startsWith("DT_")
        )) {
            const outputSlot = node.findOutputSlot(output.name);
            const outputNodes = node.getOutputNodes(outputSlot) ?? [];
            for (const outputNode of outputNodes) {
                const root = findRoot(outputNode);
                if (root) return root;
            }
        }
    }

    return undefined;
}
