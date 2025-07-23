import "dotenv/config";
import { Page, expect } from "@playwright/test";

const comfyUrl = process.env.PLAYWRIGHT_TEST_URL || "";
if (!comfyUrl) throw new Error("PLAYWRIGHT_TEST_URL is not set");

export async function openWorkflow(workflow: string, page: Page) {
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
    await fileChooser.setFiles(workflow);

    await page.locator("#graph-canvas").click({
        position: {
            x: 137,
            y: 246,
        },
    });

    await page.waitForTimeout(1000);
    await page.locator("#graph-canvas").press(".");
    await page.waitForTimeout(1000);
}

export async function createNewWorkflow(page: Page) {
    await page.goto(comfyUrl);

    // Expect a title "to contain" a substring.
    await expect(page).toHaveTitle(/ComfyUI/);

    await page.locator('a').filter({ hasText: /^Workflow$/ }).click();
    await page.getByRole('menuitem', { name: 'New' }).locator('a').click();

    await page.evaluate(() => {
        if (!window.app?.canvas.ds) return
        window.app.canvas.ds.offset = [50, 50]
        window.app.canvas.ds.scale = 0.8
        window.app.canvas.setDirty(true, true)
    })
}

export async function centerOnPoint(page: Page, x: number, y: number, scale = 1) {
    await page.evaluate(async ([x, y, scale]) => {
        if (!window.app?.canvas.ds) return
        window.app.canvas.ds.scale = 1
        const [_x, _y, w, h] = window.app.canvas.visible_area
        window.app.canvas.ds.offset = [x + w / 2, y + h / 2]
        window.app.canvas.setDirty(true, true)
        await new Promise((resolve) => setTimeout(resolve, 200));
    }, [x, y, scale])
}
