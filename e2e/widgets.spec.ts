import "dotenv/config";
import { test, expect } from "@playwright/test";
// import { expect } from "./fixtures";
import { getNodeRef, NodeRef } from "./nodeRef";
import { openWorkflow } from "./util";
import { join } from "node:path";

const comfyUrl = process.env.PLAYWRIGHT_TEST_URL || "";
if (!comfyUrl) throw new Error("PLAYWRIGHT_TEST_URL is not set");

export const workflowFolder = "./e2e/workflows";

test("widget change when settings mode changes", async ({ page }) => {
    await openWorkflow(join(workflowFolder, "node.json"), page);

    const nodeRef = await getNodeRef(page, "DrawThingsSampler");

    // "model" on basic and advanced
    // "strength" on basic
    // "clip_skip" on advanced

    expect(await nodeRef.isWidgetVisible("settings")).toBeTruthy();

    // start on basic
    await nodeRef.clickWidget("settings");
    await page.getByRole("menuitem", { name: "Basic" }).click();
    expect(
        await nodeRef.isWidgetVisible([
            "settings",
            "model",
            "strength",
            "clip_skip",
        ])
    ).toMatchObject([true, true, true, false]);

    // go to advanced
    await nodeRef.clickWidget("settings");
    await page.getByRole("menuitem", { name: "Advanced" }).click();
    expect(
        await nodeRef.isWidgetVisible([
            "settings",
            "model",
            "strength",
            "clip_skip",
        ])
    ).toMatchObject([true, true, false, true]);

    // go to all
    await nodeRef.clickWidget("settings");
    await page.getByRole("menuitem", { name: "All" }).click();

    expect(
        await nodeRef.isWidgetVisible([
            "settings",
            "model",
            "strength",
            "clip_skip",
        ])
    ).toMatchObject([true, true, true, true]);
});

test("tcd sampler", async ({ page }) => {
    await openWorkflow(join(workflowFolder, "node.json"), page);

    const node = await getNodeRef(page, "DrawThingsSampler");
    await node.selectWidgetOption("settings", "Basic");
    await page.waitForTimeout(500);

    // select euler
    await node.selectWidgetOption("sampler_name", "Euler A");

    // assert stochastic sampling gamma is not visible
    expect(await node.isWidgetVisible("stochastic_sampling_gamma")).toBeFalsy();

    // select tcd
    await node.selectWidgetOption("sampler_name", "TCD");

    // assert stochastic sampling gamma appears
    expect(
        await node.isWidgetVisible("stochastic_sampling_gamma")
    ).toBeTruthy();

    // select euler
    await node.selectWidgetOption("sampler_name", "Euler A");

    // assert stochastic sampling gamma is not visible
    expect(await node.isWidgetVisible("stochastic_sampling_gamma")).toBeFalsy();
});

test("hires, tiled diffusion, tiled decoding widgets", async ({ page }) => {
    await openWorkflow(join(workflowFolder, "node.json"), page);

    const nodeRef = await getNodeRef(page, "DrawThingsSampler");

    // go to advanced
    await nodeRef.clickWidget("settings");
    await page.getByRole("menuitem", { name: "Advanced" }).click();

    await testDependentOptions(nodeRef, "high_res_fix", [
        "high_res_fix_start_width",
        "high_res_fix_start_height",
        "high_res_fix_strength",
    ]);

    await testDependentOptions(nodeRef, "tiled_diffusion", [
        "diffusion_tile_width",
        "diffusion_tile_height",
        "diffusion_tile_overlap",
    ]);

    await testDependentOptions(nodeRef, "tiled_decoding", [
        "decoding_tile_width",
        "decoding_tile_height",
        "decoding_tile_overlap",
    ]);
});

test("flux settings widgets", async ({ page }) => {
    await openWorkflow(join(workflowFolder, "node.json"), page);

    const nodeRef = await getNodeRef(page, "DrawThingsSampler");

    // go to basic
    await nodeRef.clickWidget("settings");
    await page.getByRole("menuitem", { name: "Basic" }).click();

    // select an sd model
    await nodeRef.clickWidget("model");
    await page
        .getByRole("menuitem", { name: /\(SD\)/ })
        .first()
        .click();

    // make sure flux widgets are not visible
    expect(await nodeRef.isWidgetVisible("res_dpt_shift")).toBeFalsy();

    await nodeRef.selectWidgetOption("settings", "Advanced");
    expect(
        await nodeRef.isWidgetVisible([
            "tea_cache",
            "speed_up",
            "separate_clip_l",
        ])
    ).toMatchObject([false, false, false]);

    // select flux model
    await nodeRef.selectWidgetOption("model", /\(F1\)/);

    await nodeRef.selectWidgetOption("settings", "Advanced");
    expect(
        await nodeRef.isWidgetVisible([
            "tea_cache",
            "speed_up",
            "separate_clip_l",
        ])
    ).toMatchObject([true, true, true]);

    // test tea_cache
    await testDependentOptions(nodeRef, "tea_cache", [
        "tea_cache_start",
        "tea_cache_end",
        "tea_cache_threshold",
        "tea_cache_max_skip_steps",
    ]);

    // test speed_up
    await testDependentOptions(
        nodeRef,
        "speed_up",
        ["guidance_embed"],
        "invert"
    );

    // test separate_clip_l
    await testDependentOptions(nodeRef, "separate_clip_l", ["clip_l_text"]);

    await nodeRef.selectWidgetOption("settings", "Basic");

    await testDependentOptions(nodeRef, "res_dpt_shift", ["shift"], "disable");
});

test("svd options", async ({ page }) => {});

test("wan options", async ({ page }) => {
    await openWorkflow(join(workflowFolder, "node.json"), page);

    const nodeRef = await getNodeRef(page, "DrawThingsSampler");

    // go to basic
    await nodeRef.clickWidget("settings");
    await page.getByRole("menuitem", { name: "Basic" }).click();

    // select an sd model
    await nodeRef.clickWidget("model");
    await page
        .getByRole("menuitem", { name: /\(SD\)/ })
        .first()
        .click();

    // make sure wan widgets are not visible
    await nodeRef.selectWidgetOption("settings", "Advanced");
    expect(
        await nodeRef.isWidgetVisible([
            "causal_inference",
            "causal_inference_pad",
            "tea_cache",
        ])
    ).toMatchObject([false, false, false]);

    // select wan model
    await nodeRef.selectWidgetOption("model", /Wan 2.1/);

    // assert widgets appear
    expect(
        await nodeRef.isWidgetVisible([
            "causal_inference",
            "causal_inference_pad",
            "tea_cache",
        ])
    ).toMatchObject([true, true, true]);

    // test tea_cache
    await testDependentOptions(nodeRef, "tea_cache", [
        "tea_cache_start",
        "tea_cache_end",
        "tea_cache_threshold",
        "tea_cache_max_skip_steps",
    ]);
});

async function testDependentOptions(
    node: NodeRef,
    primary: string,
    dependents: string[],
    mode: "normal" | "invert" | "disable" = "normal"
) {
    const allTrue = new Array(dependents.length).fill(mode !== "invert");
    const allFalse = new Array(dependents.length).fill(mode === "invert");

    const check =
        mode === "disable"
            ? (...args: Parameters<NodeRef["isWidgetDisabled"]>) =>
                  node.isWidgetDisabled(...args)
            : (...args: Parameters<NodeRef["isWidgetVisible"]>) =>
                  node.isWidgetVisible(...args);

    expect(await node.isWidgetVisible(primary)).toBeTruthy();

    // make sure option is off
    if (await node.getWidgetValue(primary)) {
        await node.clickWidget(primary);
    }
    expect(await node.getWidgetValue(primary)).toBeFalsy();

    // assert dependents are not visible
    expect(await check(dependents)).toMatchObject(allFalse);

    // turn option on
    await node.clickWidget(primary);
    expect(await node.getWidgetValue(primary)).toBeTruthy();

    // assert dependents are visible
    expect(await check(dependents)).toMatchObject(allTrue);

    // turn it back off
    await node.clickWidget(primary);
    expect(await node.getWidgetValue(primary)).toBeFalsy();

    // assert dependents are not visible
    expect(await check(dependents)).toMatchObject(allFalse);
}
