// import { app } from "../../scripts/app.js";
import { app } from "http://localhost:8188/scripts/app.js";

function hideWidgetDT(widgetObjects, hide) {
    widgetObjects.forEach(listObjects);
    function listObjects(widgetObject) {
        widgetObject.disabled = hide;
    }
}

app.registerExtension({
    name: "ComfyUI-DrawThings-gRPC",
    async nodeCreated(node) {
        if (node?.comfyClass === "DrawThingsSampler") {
            const widgetsList = {};

            node.widgets.forEach(listWidgets);
            function listWidgets(widget) {
                widgetsList[widget.name] = widget;
                // console.log(widget.value);
            }

            widgetsList["refiner"].callback = function () {
                // console.log(this.value);
                if (this.value == true) {
                    hideWidgetDT([widgetsList["refiner_model"], widgetsList["refiner_start"]], false);
                } else {
                    hideWidgetDT([widgetsList["refiner_model"], widgetsList["refiner_start"]], true);
                }
            }

            widgetsList["high_res_fix"].callback = function () {
                // console.log(this.value);
                if (this.value == true) {
                    hideWidgetDT([widgetsList["high_res_fix_start_width"], widgetsList["high_res_fix_start_height"], widgetsList["high_res_fix_strength"]], false);
                } else {
                    hideWidgetDT([widgetsList["high_res_fix_start_width"], widgetsList["high_res_fix_start_height"], widgetsList["high_res_fix_strength"]], true);
                }
            }
        }
    },
    async afterConfigureGraph() {
        // console.log(app.graph._nodes);
        app.graph._nodes.forEach(listNodes);
        function listNodes(node) {
            if (node.type === "DrawThingsSampler") {
                console.log(node);
                const widgetsList = {};
    
                node.widgets.forEach(listWidgets);
                function listWidgets(widget) {
                    widgetsList[widget.name] = widget;
                    // console.log(widget.value);
                }

                if (widgetsList["refiner"].value == true) {
                    hideWidgetDT([widgetsList["refiner_model"], widgetsList["refiner_start"]], false);
                } else {
                    hideWidgetDT([widgetsList["refiner_model"], widgetsList["refiner_start"]], true);
                }

                if (widgetsList["high_res_fix"].value == true) {
                    hideWidgetDT([widgetsList["high_res_fix_start_width"], widgetsList["high_res_fix_start_height"], widgetsList["high_res_fix_strength"]], false);
                } else {
                    hideWidgetDT([widgetsList["high_res_fix_start_width"], widgetsList["high_res_fix_start_height"], widgetsList["high_res_fix_strength"]], true);
                }
            }
        }
    }
});
