// import { app } from "../../scripts/app.js";
import { app } from "http://localhost:8188/scripts/app.js";

const widgetParentsDT = ["refiner", "high_res_fix"];

function hideWidgetDT(widgetObject, hide) {
    widgetObject.disabled = hide;
    // widgetObject.hidden = hide;
}

app.registerExtension({
    name: "ComfyUI-DrawThings-gRPC",
    async nodeCreated(node) {
        if (node?.comfyClass === "DrawThingsSampler") {
            const widgetsList = {};
            node.widgets.forEach(listWidgets);
            function listWidgets(widget) {
                widgetsList[widget.name] = widget;
            }

            widgetParentsDT.forEach(listParents);
            function listParents(parent) {
                const widgetParent = widgetsList[parent];
                // console.log(widgetParent);

                widgetParent.callback = function () {
                    for (let [name, child] of Object.entries(widgetsList)) {
                        if (child.name.startsWith(parent+"_")) {
                            if (this.value == true) {
                                hideWidgetDT(child, false);
                            } else {
                                hideWidgetDT(child, true);
                            }
                            // console.log(child);
                        }
                    }
                }
            }
        }
    },
    async afterConfigureGraph() {
        console.log(app);
        app.graph._nodes.forEach(listNodes);
        function listNodes(node) {
            if (node.type === "DrawThingsSampler") {
                const widgetsList = {};
                node.widgets.forEach(listWidgets);
                function listWidgets(widget) {
                    widgetsList[widget.name] = widget;
                }

                widgetParentsDT.forEach(listParents);
                function listParents(parent) {
                    const widgetParent = widgetsList[parent];
                    // console.log(widgetParent);

                    for (let [name, child] of Object.entries(widgetsList)) {
                        if (child.name.startsWith(parent+"_")) {
                            if (widgetParent.value == true) {
                                hideWidgetDT(child, false);
                            } else {
                                hideWidgetDT(child, true);
                            }
                            // console.log(child);
                        }
                    }
                }
            }
        }
    }
});
