/** @import { ComfyApi, ComfyApp } from "@comfyorg/comfyui-frontend-types"; */
/** @import { LGraphNode } from "@comfyorg/litegraph" */
import { app } from "../../scripts/app.js";

import { findModel } from "./models.js";

app.registerExtension({
  name: "ComfyUI-Dynamic-Inputs",
  beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeType.comfyClass === "DrawThingsLoRA") {
      console.log("a");
      // update inputs when a new node is added
      setCallback(nodeType.prototype, "onAdded", function (graph) {
        updateInputs(this);
      });

      // or when the model changes
      setCallback(nodeType.prototype, "onWidgetChanged", function (name, value, old_Value, widget) {
        updateInputs(this);
      });
    }
  },
});

/** @param {LGraphNode} node */
function updateInputs(node) {
  const s = saveSize(node);

  // the first input should always be lora
  if (node.inputs.length < 1 || node.inputs.length > 2 || node.inputs[0].type !== "DT_LORA") {
    for (let i = node.inputs.length - 1; i >= 0; i++) {
      node.removeInput(i);
    }
    node.addInput("lora", "DT_LORA");
  }

  const modelWidget = node?.widgets?.find((w) => w.options?.modelType);
  if (!modelWidget) return;
  const hintType = findModel(modelWidget.value, modelWidget.options?.modelType)?.modifier;

  if (hintType) {
    if (node.inputs.length < 2 || node.inputs[1]?.type !== "IMAGE") {
      node.removeInput(1);
      node.addInput("control_image", "IMAGE");
    }
  } else {
    while (node.inputs.length > 1) {
      node.removeInput(1);
    }
  }
  s();
}

/** @param {LGraphNode} node */
function saveSize(node) {
  const width = node.width;
  return () => node.setSize([width, node.computeSize()[1]]);
}

/**
 * @template {Object} T
 * @template {keyof T} K
 * @param {T} target
 * @param {K} callbackName
 * @param {T[K]} callback
 */
function setCallback(target, callbackName, callback) {
  const originalCallback = target[callbackName];
  target[callbackName] = function (...args) {
    const r = originalCallback?.apply(this, args);
    callback?.apply(this, args);
    return r;
  };
}
