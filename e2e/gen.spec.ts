import "dotenv/config";
import { test, expect, Page } from "@playwright/test";
import sharp from "sharp";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";

const comfyUrl = process.env.PLAYWRIGHT_TEST_URL || "";
if (!comfyUrl) throw new Error("PLAYWRIGHT_TEST_URL is not set");

const workflowFolder = "./e2e/workflows";
const comfyFolder = process.env.TEST_COMFYUI_DIR || "";
if (!comfyFolder) throw new Error("TEST_COMFYUI_DIR is not set");

const outputFolder = join(comfyFolder, "output");

test.beforeEach(async () => {
    test.setTimeout(300000);
});

// note: it's entirely possibly that these tests will not work when run on a different system
// I tried to cover most features/settings with these, so if anything breaks, the generated output should be different
// (they could also break for a variety of other reasons)
// requires DT running at localhost:7859 with the following models
// Stable Diffusion 1.5
// st_monsters embedding (https://civitai.com/models/332132)
// inpainting cnet sd1.5
// Foooocus Inpaint SDXL v2.6
// FoxAI Pony Fantastic (for refiner test) (https://civitai.com/models/856827)
// Xi v2 (for refiner test) (https://civitai.com/models/259563)
// Flux dev (8bit)
// FLUX.1 Turbo Alpha

test("test output: sd1_a", async ({ page }) => {
    await compareOutput(
        page,
        "sd1_a",
        "BPifOIFwZHkANGVDADTycgCwsmmELRxvECRbbnik22D4dfFpwPt4bQD4eHsAeHh2BHg5XgT4OGsEtOBgBDyzZAQ4OXIEsjlnBYxkfwbMRn5GQsZ8gGTPYABtzxQEbLM1QCxz54hpc5KdwOMI9CQzAeHDA2wxzWAKIJnMCGJQzM0="
    );
});

test("test output: sd1_b", async ({ page }) => {
    await compareOutput(
        page,
        "sd1_b",
        "APy/EwD+/yUA/v+rAP7/XQD8/7uA8P9vQPT/zyDg/ecg4PzxQOT99EH0ef6GeVH/Bv7w/wP+9P8C/vTfAP712wDusZsE/BA8cOIixZZ4OX8ATA0eAc8d1gDYmf8A8pz/BPO8/wDt8P+A6ZH/gPmZ/wAcG/8A8gb/APCM/wB4Ov4="
    );
});

test("test output: sdxl_a", async ({ page }) => {
    await compareOutput(
        page,
        "sdxl_a",
        "AAD+9wCA/PcMwPD3AGDg/wBg5v8AcM7/ADDG/wDg9P8AYPL/AODQ/4xm+P9Mx8z/OJMM/yGTPP+as377mDM++8hjPu/EYz73pOMc9sLRFO3CkR2thOE+2QEzNtlFHx/YR0s+2ssNH9zMH4PMjEvG2ZjJxNi46yXYmIcqqZANF+c="
    );
});

test("test output: sdxl_b", async ({ page }) => {
    await compareOutput(
        page,
        "sdxl_b",
        "AAkToqBmDkQQcRiggBk4QIBceAFAXugQQD/AhEA6wJBALtCSADsJSaBJTFGQTSxRgDkySQRjGYSAZk0pAFZFogA+QZIQXCNSBxoDCRgaE+ZefGCEj9bEDqbczA4WvoCLiZ6AH+O8YCwJj4BrCY+ETwaOBE8ApwMuikEbbqagEzA="
    );
});

test("test output: flux_a", async ({ page }) => {
    test.setTimeout(300000);
    await compareOutput(
        page,
        "flux_a",
        "AAD8vwAA7P4AgO/7AED68wAw+bcAmOG3ApDh+yBwmf0g4OH9AOjp7wBgTLYA8Gy0AJw83ACybTwA2fiwAJx62QAPT9kAD0/LoJ0/cgCTd3JAkz/ywBo58oiaPfIIGD7iBFg+4GCYPeCAqDigkFh45ACYfaQQKXzkyGx85MBseKQ="
    );
});

test("test output: img2img_crop_ti", async ({ page }) => {
    await compareOutput(
        page,
        "img2img_crop_ti",
        "aIi4Oeig3zlYxi452AyOOdgmjjy4Np48q7LenJw5p5xQO5OcIhbHnA8T84yPifnMncD+zB7G+MwuzL5cZ1yzTNP42c3seJTHbtzO5x9xz+O1MefTY4P753GH/0V8jj8OfA4fH/Eujj/5w4Q/PwDgHzsC+R97ossPObHvD/FUPgc="
    );
});

test("test output: inpaint_cnet", async ({ page }) => {
    await compareOutput(
        page,
        "inpaint_cnet",
        "D5MPRT4sXEbw2dhInMGYQfjEIGXB9khhKRpIK2FfULRhbSQkv5/NJP/LkKbzsTQLudkkC57DpQ2uwbMUrgPLBo5jDovuZwyDfqfYxP5n2IT+8vwK/jHygG5DacBuZryAboa2ke7GsYje5zAK/GswDPE3MF7gH+Be4EcMlvYjx5g="
    );
});

test("test output: inpaint_sdxl", async ({ page }) => {
    await compareOutput(
        page,
        "inpaint_sdxl",
        "eMhwU2fDdyY+06gUcOmQBbQ9mU3NTTAJ2DZksZx2TLHFjxsz5+wYM9z2ZChsmIxYZ81MTGvBSWJpwUm3ZQFTO2UDhjkhAw5MAYY4TA+WuTcfmh0TP5iZE7sYXEvfGIYD3TDRA13D2BKdjmwDn5Zmx7/Gc2R74WFo33Dg1P4O4NI="
    );
});

async function compareOutput(page: Page, workflow: string, hash?: string) {
    await page.goto(comfyUrl);

    // Expect a title "to contain" a substring.
    await expect(page).toHaveTitle(/ComfyUI/);

    const fileChooserPromise = page.waitForEvent("filechooser");

    await page
        .locator("a")
        .filter({ hasText: /^Workflow$/ })
        .click();
    await page.getByText("OpenCtrl + o").click();

    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(join(workflowFolder, workflow + ".json"));

    await page.waitForTimeout(1000);

    await page.locator("#graph-canvas").click({
        position: {
            x: 137,
            y: 246,
        },
    });

    await page.locator("#graph-canvas").press(".");

    await page.waitForTimeout(1000);
    await page.getByRole("button", { name: "Queue (q)" }).click();
    await page
        .getByTestId("queue-button")
        .getByRole("button", { name: "Run" })
        .click();

    await page.waitForTimeout(1000);
    await expect(page.locator(".task-item").first()).not.toContainText(
        "Running",
        { timeout: 300000 }
    );

    await page.locator(".task-item").first().click();
    const filename = await page
        .locator(".p-galleria-item")
        .locator("img")
        .getAttribute("alt");
    const filepath = join(outputFolder, filename);

    // const fileHash = await sha256sum(filepath);
    // console.log(`hash for ${workflow}: ${fileHash}`);

    const dhash = await getDhash(filepath);
    console.log(`dhash for ${workflow}: ${dhash}`);

    expect(dhash).toBe(hash);

    // if (
    //     !(await sharp(filepath)
    //         .removeAlpha()
    //         .raw()
    //         .toBuffer({ resolveWithObject: true }))
    // ) {
    //     throw new Error("No output image");
    // }

    // await sharp(workflowFolder + workflow + ".png")
    //     .composite([{ input: filepath, blend: "difference" }])
    //     .toFile(workflowFolder + workflow + "_diff.png");

    // const outImg = await sharp(filepath)
    //     .removeAlpha()
    //     .raw()
    //     .toBuffer({ resolveWithObject: true });
    // const refImg = await sharp(join(workflowFolder, workflow + ".png"))
    //     .removeAlpha()
    //     .raw()
    //     .toBuffer({ resolveWithObject: true });

    // let totalDif = 0;
    // let maxDif = 0;

    // let pixels = 0;

    // for (let i = 0; i < outImg.data.length; i++) {
    //     pixels++;
    //     const dif = Math.abs(outImg.data[i] - refImg.data[i]);
    //     totalDif += dif;
    //     if (dif > maxDif) maxDif = dif;
    // }

    // console.log(totalDif / pixels, maxDif);

    // expect(totalDif / pixels).toBeLessThanOrEqual(6);
}

async function sha256sum(filepath: string): Promise<string> {
    const hash = createHash("sha256");
    const stream = createReadStream(filepath);
    await new Promise((resolve, reject) => {
        stream.on("error", reject);
        stream.on("end", resolve);
        stream.pipe(hash);
    });
    return hash.digest("hex");
}

async function getDhash(imagePath: string, hashSize = 32): Promise<string> {
    const image = await sharp(imagePath)
        .resize(hashSize + 1, hashSize)
        .grayscale()
        // .toFile('temp.png')
        .raw()
        .toBuffer({ resolveWithObject: true });

    // const image = await sharp(imagePath).raw().toBuffer({ resolveWithObject: true })

    let hash = "";
    const hb = new Uint8Array((hashSize * hashSize) / 8);
    let p = 0;

    for (let y = 0; y < hashSize; y++) {
        for (let x = 0; x < hashSize; x++) {
            const index = (y * (hashSize + 1) + x) * image.info.channels;
            const left = image.data[index];
            const right = image.data[index + image.info.channels];
            const bit = left > right ? 1 : 0;
            hash += bit.toString();

            hb[Math.floor(p / 8)] += bit << p % 8;
            p++;
        }
    }

    return Buffer.from(hb).toString("base64");
}
