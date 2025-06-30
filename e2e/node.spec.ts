import "dotenv/config";
import { test } from "@playwright/test";
import { expect } from "./fixtures";
import { getNodeRef } from "./nodeRef";
import { openWorkflow } from "./util";
import { join } from "node:path";

const comfyUrl = process.env.PLAYWRIGHT_TEST_URL || "";
if (!comfyUrl) throw new Error("PLAYWRIGHT_TEST_URL is not set");

export const workflowFolder = "/Users/kcjer/Desktop/comfy_test_img/";
const outputFolder = "/Users/kcjer/sd/ComfyUI/output/";

test("widget change when settings mode changes", async ({ page }) => {
    await page.goto(comfyUrl);

    // Expect a title "to contain" a substring.
    await expect(page).toHaveTitle(/ComfyUI/);

    await openWorkflow(join(workflowFolder, "node.json"), page);

    const nodeRef = await getNodeRef(page, "DrawThingsSampler");

    // "model" on basic and advanced
    // "strength" on basic
    // "clip_skip" on advanced

    expect(await nodeRef.isWidgetVisible("settings")).toBeTruthy();

    // start on basic
    await nodeRef.clickWidget("settings");
    await page.getByRole("menuitem", { name: "Basic" }).click();

    expect(await nodeRef.isWidgetVisible("settings")).toBeTruthy();
    expect(await nodeRef.isWidgetVisible("model")).toBeTruthy();
    expect(await nodeRef.isWidgetVisible("strength")).toBeTruthy();
    expect(await nodeRef.isWidgetVisible("clip_skip")).toBeFalsy();

    // go to advanced
    await nodeRef.clickWidget("settings");
    await page.getByRole("menuitem", { name: "Advanced" }).click();

    expect(await nodeRef.isWidgetVisible("settings")).toBeTruthy();
    expect(await nodeRef.isWidgetVisible("model")).toBeTruthy();
    expect(await nodeRef.isWidgetVisible("strength")).toBeFalsy();
    expect(await nodeRef.isWidgetVisible("clip_skip")).toBeTruthy();

    // go to all
    await nodeRef.clickWidget("settings");
    await page.getByRole("menuitem", { name: "All" }).click();

    expect(await nodeRef.isWidgetVisible("settings")).toBeTruthy();
    expect(await nodeRef.isWidgetVisible("model")).toBeTruthy();
    expect(await nodeRef.isWidgetVisible("strength")).toBeTruthy();
    expect(await nodeRef.isWidgetVisible("clip_skip")).toBeTruthy();
});

test("hires, tiled diffusion, tiled decoding widgets", async ({ page }) => {
    await page.goto(comfyUrl);

    // Expect a title "to contain" a substring.
    await expect(page).toHaveTitle(/ComfyUI/);

    await openWorkflow(join(workflowFolder, "node.json"), page);

    const nodeRef = await getNodeRef(page, "DrawThingsSampler");

    // go to advanced
    await nodeRef.clickWidget("settings");
    await page.getByRole("menuitem", { name: "Advanced" }).click();

    expect(await nodeRef.isWidgetVisible("high_res_fix")).toBeTruthy();

    if (await nodeRef.getWidgetValue("high_res_fix")) {
        await nodeRef.clickWidget("high_res_fix");
    }
    await page.waitForTimeout(500);
    expect(await nodeRef.getWidgetValue("high_res_fix")).toBeFalsy();

    expect(
        await nodeRef.isWidgetVisible([
            "high_res_fix_start_width",
            "high_res_fix_start_height",
            "high_res_fix_strength",
        ])
    ).toEachBeFalsy();

    await nodeRef.clickWidget("high_res_fix");
    await page.waitForTimeout(500);

    expect(await nodeRef.getWidgetValue("high_res_fix")).toBeTruthy();

    expect(
        await nodeRef.isWidgetVisible([
            "high_res_fix_start_width",
            "high_res_fix_start_height",
            "high_res_fix_strength",
        ])
    ).toEachBeTruthy();
});
