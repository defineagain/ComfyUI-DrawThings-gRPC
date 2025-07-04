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
