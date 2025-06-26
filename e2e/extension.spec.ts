import 'dotenv/config'
import { test, expect, Page } from '@playwright/test';
import sharp from 'sharp';

const comfyUrl = process.env.PLAYWRIGHT_TEST_URL || ""
if (!comfyUrl) throw new Error('PLAYWRIGHT_TEST_URL is not set')

const workflowFolder = '/Users/kcjer/Desktop/comfy_test_img/'
const outputFolder = process.env.TEST_COMFYUI_DIR || ""
if (!outputFolder) throw new Error('TEST_COMFYUI_DIR is not set')

test('test output: sd1_a', async ({ page }) => {
    await compareOutput(page, 'sd1_a')
})

test('test output: sd1_b', async ({ page }) => {
    await compareOutput(page, 'sd1_b')
})

test('test output: sdxl_a', async ({ page }) => {
    await compareOutput(page, 'sdxl_a')
})

test('test output: sdxl_b', async ({ page }) => {
    await compareOutput(page, 'sdxl_b')
})

test('test output: flux_a', async ({ page }) => {
    await compareOutput(page, 'flux_a')
})

async function compareOutput(page: Page, workflow: string) {
    await page.goto(comfyUrl);

    // Expect a title "to contain" a substring.
    await expect(page).toHaveTitle(/ComfyUI/);

    const fileChooserPromise = page.waitForEvent('filechooser');

    await page.locator('a').filter({ hasText: /^Workflow$/ }).click();
    await page.getByText('OpenCtrl + o').click();

    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(workflowFolder + workflow + ".json");

    await page.locator('#graph-canvas').click({
        position: {
            x: 137,
            y: 246
        }
    });

    await page.locator('#graph-canvas').press(".")

    // await page.evaluate(() => {
    //     const node = window.app!.graph._nodes.find(n => n.type === "DrawThingsSampler")
    //     const settingsWidgets = node?.widgets?.find((w) => w.name === "settings")
    //     settingsWidgets.value = "Basic"
    //     node?.setDirtyCanvas(true, true)
    // })

    await page.waitForTimeout(1000)
    await page.getByRole('button', { name: 'Queue (q)' }).click();
    await page.getByTestId('queue-button').getByRole('button', { name: 'Run' }).click();


    await page.waitForTimeout(1000)
    await expect(page.locator('.task-item').first()).not.toContainText('Running', { timeout: 300000 });

    await page.locator('.task-item').first().click();
    const filename = await page.locator('.p-galleria-item').locator('img').getAttribute('alt')
    const filepath = outputFolder + filename

    if (!(await sharp(filepath).removeAlpha().raw().toBuffer({ resolveWithObject: true }))) {
        throw new Error('No output image')
    }

    await sharp(workflowFolder + workflow + '.png').composite([{ input: filepath, blend: 'difference' }]).toFile(workflowFolder + workflow + '_diff.png');

    const outImg = await sharp(filepath).removeAlpha().raw().toBuffer({ resolveWithObject: true })
    const refImg = await sharp(workflowFolder + workflow + '.png').removeAlpha().raw().toBuffer({ resolveWithObject: true })

    let totalDif = 0
    let maxDif = 0

    let pixels = 0

    for (let i = 0; i < outImg.data.length; i ++) {
        pixels++
        const dif = Math.abs(outImg.data[i] - refImg.data[i])
        totalDif += dif
        if (dif > maxDif) maxDif = dif


    }

    console.log(totalDif / pixels, maxDif)

    expect(totalDif / pixels).toBeLessThanOrEqual(6)
}
