import { app } from "../../scripts/app.js";

// console.log(app);

app.registerExtension({
    name: "ComfyUI-DrawThings-gRPC",
    async setup() {
        console.log("Setup complete!");
    },
    async nodeCreated(node) {
        console.log("nodeCreated");
        if (node?.comfyClass === "DrawThingsSampler") {
            const original_onMouseDown = node.onMouseDown;
            node.onMouseDown = function (e, pos, canvas) {
                console.log("Click!");
                return original_onMouseDown?.apply(this, arguments);
            };
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass == "DrawThingsSampler") {
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (
                side,
                slot,
                connect,
                link_info,
                output
            ) {
                const r = onConnectionsChange?.apply(this, arguments);
                console.log("Someone changed my connection!");
                return r;
            };
        }
    },
});
