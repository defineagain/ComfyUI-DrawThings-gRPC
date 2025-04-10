/** @import { LGraphNode, WidgetCallback, IWidget } from "litegraph.js"; */

/** @param node {LGraphNode} */
export function addServerListeners(node) {
    console.log("hook");

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

/** @type Map<string, { models: any[], controlNets: any[], loras: []} */
const modelInfoStore = new Map();
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

/** @param node {LGraphNode} */
export async function updateNodeModels(node) {
    // find the sampler node
    const root = findRoot(node);
    if (!root) return;

    // get the server and port
    const server = root.widgets.find((w) => w.name === "server").value;
    const port = root.widgets.find((w) => w.name === "port").value;

    const key = modelInfoStoreKey(server, port);
    console.log("checking", key);

    // fetch the models list and update store
    if (server && port) {
        const response = await getFiles(server, port);
        if (!response.ok) {
            modelInfoStore.set(key, failedConnectionInfo);
        } else {
            const data = await response.json();
            modelInfoStore.set(key, data);
        }
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
    if (!node) return;

    console.debug("looking for a root from", node.comfyClass, node.id);
    if (node?.isDtRootNode === true) return node;

    if (node?.isDtRootNode === false) {
        // this isn't necesarrily a safe assumption but it is for now
        for (const output of node.outputs.filter((o) =>
            o.type.startsWith("DT_")
        )) {
            const outputNodes = node.getOutputNodes(output.slot_index) ?? [];
            for (const outputNode of outputNodes) {
                const root = findRoot(outputNode);
                if (root) return root;
            }
        }
    }

    return undefined;
}
