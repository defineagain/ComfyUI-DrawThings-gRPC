import { Page } from '@playwright/test';

export async function openWorkflow(workflow: string, page: Page) {
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

    await page.locator("#graph-canvas").press(".");
    await page.waitForTimeout(2000);
}
