import { app } from "../../scripts/app.js";
// import { app } from "http://localhost:8188/scripts/app.js";

app.registerExtension({
    name: "ComfyUI-DrawThings-gRPC",
    async nodeCreated(node) {
        if (node?.comfyClass === "DrawThingsSampler") {
            const widgetsList = {};

            node.widgets.forEach(listWidgets);
            function listWidgets(widget) {
                widgetsList[widget.name] = widget;
                console.log(widget.value);
            }

            function hideWidget(widgetNames, hide) {
                widgetNames.forEach(listNames);
                function listNames(widgetName) {
                    widgetsList[widgetName].disabled = hide;
                    console.log(widgetsList[widgetName]);
                }
            }

            if (widgetsList["refiner"].value == true) {
                hideWidget(["refiner_model", "refiner_start"], false);
            } else {
                hideWidget(["refiner_model", "refiner_start"], true);
            }

            widgetsList["refiner"].callback = function () {
                console.log(this.value);
                if (this.value == true) {
                    hideWidget(["refiner_model", "refiner_start"], false);
                } else {
                    hideWidget(["refiner_model", "refiner_start"], true);
                }
            }

            if (widgetsList["high_res_fix"].value == true) {
                hideWidget(["high_res_fix_start_width", "high_res_fix_start_height", "high_res_fix_strength"], false);
            } else {
                hideWidget(["high_res_fix_start_width", "high_res_fix_start_height", "high_res_fix_strength"], true);
            }

            widgetsList["high_res_fix"].callback = function () {
                console.log(this.value);
                if (this.value == true) {
                    hideWidget(["high_res_fix_start_width", "high_res_fix_start_height", "high_res_fix_strength"], false);
                } else {
                    hideWidget(["high_res_fix_start_width", "high_res_fix_start_height", "high_res_fix_strength"], true);
                }
            }
        }
    },
});
