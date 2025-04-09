import { app } from "../../scripts/app.js";

// console.log(app);

app.registerExtension({
    name: "ComfyUI-DrawThings-gRPC",
    async nodeCreated(node) {
        if (node?.comfyClass === "DrawThingsSampler") {
            node.widgets.forEach(listWidgets);
            function listWidgets(widget) {
                if (widget.name == "high_res_fix") {
                    widget.callback = function () {
                        // console.log(this);
                        console.log(this.value);
                        if (this.value) {
                        }
                    };
                }
            }
        }
    },
});
