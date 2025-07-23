import "dotenv/config";
import { expect } from "@playwright/test";
import { join } from "node:path";
import { readFile } from 'node:fs/promises';
import { test } from './fixtures';

export const workflowFolder = "./e2e/workflows";

test("load workflow from previous version", async ({ comfy }) => {
    // load workflow
    await comfy.goto()
    await comfy.openWorkflow(join(workflowFolder, "old_version.json"));

    // assert toast appears
    const toastLoc = comfy.page.locator(".p-toast-message-text")
    await expect(toastLoc).toContainText("Draw Things gRPC")
    await expect(toastLoc).toContainText("The Draw Things Sampler node contained invalid values - they have been corrected")

    // check properties
    const node = (await comfy.getNodeRef("DrawThingsSampler"))!;
    await node.centerNode();
    await node.selectWidgetOption("sampler_name", "TCD");
    await expect(await node.isWidgetVisible("stochastic_sampling_gamma")).toBeTruthy();

    // assert workflow json file used the old prompt nodes
    const workflowJson = await readFile(join(workflowFolder, "old_version.json"), "utf-8");
    expect(workflowJson).toContain("DrawThingsPositive");
    expect(workflowJson).toContain("DrawThingsNegative");
    expect(workflowJson).not.toContain("DrawThingsPrompt");

    // assert the old nodes were replaced when loaded
    const negativeNode = await comfy.getNodeRef("DrawThingsNegative", { doNotThrow: true });
    expect(negativeNode).toBeUndefined();

    const positiveNode = await comfy.getNodeRef("DrawThingsPositive", { doNotThrow: true });
    expect(positiveNode).toBeUndefined();

    const promptNode = await comfy.getNodeRef("DrawThingsPrompt");
    expect(promptNode).not.toBeUndefined();
})

test("sampler widgets serialization", async ({ comfy }) => {
    // start with empty workflow
    await comfy.goto()
    await comfy.createNewWorkflow()

    // add sampler node
    const node = await comfy.addNode(["DrawThings", "Draw Things Sampler"], 0, 0)
    await node.centerNode();

    // change various widget values

    // export workflow

    // assert workflow file has widget values by key

    // scramble the values in the array in the workflow file

    // load the workflow

    // assert values were loaded by key
})
