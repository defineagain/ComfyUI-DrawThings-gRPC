import { app } from "../../../scripts/app.js";

/** @import { LGraphNode } from "litegraph.js"; */

// console.log(app);

const api = window.comfyAPI.api.api;

/**
 * @param {string} server
 * @param {number | string} port
 */
async function getFiles(server, port) {
    const body = new FormData();
    body.append("server", server);
    body.append("port", port);
    // const args = new URLSearchParams({ server, port });

    const filesInfoResponse = await api.fetchApi(`/dt_grpc_files_info`, {
        method: "POST",
        body,
    });

    return filesInfoResponse;
}

app.registerExtension({
    name: "ComfyUI-DrawThings-gRPC",
    async nodeCreated(node) {
        if (node?.comfyClass === "DrawThingsSampler") {
            console.log("created", node);
            // watching server and port for changes
            const serverWidget = node.widgets.find((w) => w.name === "server");
            serverWidget.callback = function (value, graph, node) {
                updateNodeModels(node);
            };

            const portWidget = node.widgets.find((w) => w.name === "port");
            portWidget.callback = function (value, graph, node) {
                updateNodeModels(node);
            };

            const modelWidget = node.widgets.find((w) => w.name === "model");
            modelWidget.callback = function (value, graph, node) {
                // update the models list if the selected value is one of the 'failed' options
                const option = failedConnectionOptions.find(
                    (o) => o.name === value
                );
                if (option) updateNodeModels(node);
            };

            const original_onMouseDown = node.onMouseDown;
            node.onMouseDown = function (e, pos, canvas) {
                console.log("Click!");
                console.log(node);
                getFiles("localhost", 7859);
                return original_onMouseDown?.apply(this, arguments);
            };

            // updateNodeModels(node);
        }
    },
    async loadedGraphNode(node) {
        if (node?.comfyClass === "DrawThingsSampler") {
            updateNodeModels(node);
        }
    },
    /** @param nodeType {typeof LGraphNode} */
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (
            nodeType.comfyClass == "DrawThingsLoRA" ||
            nodeType.comfyClass == "DrawThingsControlNet"
        ) {
            const currentOnConnectOutput = nodeType.prototype.onConnectOutput;

            nodeType.prototype.onConnectOutput = function (...args) {
                const r = currentOnConnectOutput?.apply(this, args);
                onConnectOutput.apply(this, args);
                return r;
            };
        }
    },
});

/** @type (outputIndex: number, inputType: INodeInputSlot["type"], inputSlot: INodeInputSlot, inputNode: LGraphNode, inputIndex: number): boolean */
function onConnectOutput(
    outputIndex,
    inputType,
    inputSlot,
    inputNode,
    inputIndex
) {
    // follow the connection all the way up to Sampler to update the model list
    /** @param node {LGraphNode} */
    function findRootSampler(node) {
        if (node?.comfyClass === "DrawThingsSampler") return node;
        if (
            node?.comfyClass === "DrawThingsControlNet" ||
            node?.comfyClass === "DrawThingsLoRA"
        ) {
            return findRootSampler(node.getInputNode(0));
        }
        return undefined;
    }
    const rootSampler = findRootSampler(inputNode);
    if (rootSampler) updateNodeModels(rootSampler);
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
async function updateNodeModels(node) {
    // get the server and port
    const server = node.widgets.find((w) => w.name === "server").value;
    const port = node.widgets.find((w) => w.name === "port").value;

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
    const modelsWidget = getModelWidget(node);
    modelsWidget.options.values = modelInfo.models.map((m) => m.name);
    setValidOption(modelsWidget);

    // debugger;
    updateCnets(
        node,
        modelInfo.controlNets.map((m) => m.name)
    );
    updateLoras(
        node,
        modelInfo.loras.map((m) => m.name)
    );
}

/**
 * @param {LGraphNode} node
 * @param {string[]} options
 */
function updateCnets(node, options) {
    const cnetSlot = node.findInputSlot("control_net");
    const cnetsNode = node.getInputNode(cnetSlot);
    if (!cnetsNode) return;

    const modelWidget = getModelWidget(cnetsNode);
    modelWidget.options.values = options;
    setValidOption(modelWidget);

    updateCnets(cnetsNode, options);
}

/**
 * @param {LGraphNode} node
 * @param {string[]} options
 */
function updateLoras(node, options) {
    const loraSlot = node.findInputSlot("lora");
    const loraNode = node.getInputNode(loraSlot);
    if (!loraNode) return;
    const modelWidget = getModelWidget(loraNode);
    modelWidget.options.values = options;
    setValidOption(modelWidget);

    updateLoras(loraNode, options);
}

function getModelWidget(node) {
    if (node?.comfyClass === "DrawThingsSampler")
        return node.widgets.find((w) => w.name === "model");
    else if (node?.comfyClass === "DrawThingsControlNet")
        return node.widgets.find((w) => w.name === "control_name");
    else if (node?.comfyClass === "DrawThingsLoRA")
        return node.widgets.find((w) => w.name === "lora_name");
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
