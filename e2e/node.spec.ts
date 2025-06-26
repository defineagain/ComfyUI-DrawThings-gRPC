import 'dotenv/config'
import { test, expect, Page } from '@playwright/test';
import sharp from 'sharp';

const comfyUrl = process.env.PLAYWRIGHT_TEST_URL || ""
if (!comfyUrl) throw new Error('PLAYWRIGHT_TEST_URL is not set')

const workflowFolder = '/Users/kcjer/Desktop/comfy_test_img/'
const outputFolder = '/Users/kcjer/sd/ComfyUI/output/'

test('node', async ({ page }) => {
    await page.goto(comfyUrl);

    // Expect a title "to contain" a substring.
    await expect(page).toHaveTitle(/ComfyUI/);

    const fileChooserPromise = page.waitForEvent('filechooser');

    await page.locator('a').filter({ hasText: /^Workflow$/ }).click();
    await page.getByText('OpenCtrl + o').click();

    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(workflowFolder + "node.json");

    await page.locator('#graph-canvas').click({
        position: {
            x: 137,
            y: 246
        }
    });

    await page.locator('#graph-canvas').press(".")
    await page.waitForTimeout(2000)

    const { view, pos, widget, size } = await page.evaluate(() => {
        const view: [number, number, number, number] = window.app.canvas.visible_area
        const node = window.app.canvas.visible_nodes[0]
        const pos: [number, number, number, number] = node._posSize
        const widget: number = node.widgets[0].y

        const canvas = window.app.canvas.canvas.getBoundingClientRect()
        const size = {
            width: canvas.width,
            height: canvas.height,
        }

        return { view, pos, widget, size }
    })

    console.log(view, pos, widget, size)

    const yScale = (view[3] - view[1]) / size.height
    const xScale = (view[2] - view[0]) / size.width

    // okay we need to map coords in the workflow to the canvas
    const relativePos = {
        x: (pos[0] - view[0]) / (view[2] - view[0]) * size.width,
        y: (pos[1] - view[1]) / (view[3] - view[1]) * size.height,
        width: pos[2] / (view[2] - view[0]) * size.width,
        height: pos[3] / (view[3] - view[1]) * size.height
    };

    console.log('Relative Position and Size:', relativePos);

    const getPos = ([x, y]) => ([x * xScale - view[0], y * yScale - view[1]])

    /*

    0-100
    view is of 20-70
    you want x = 30
    which is 3/5 = 60
    x * 2 (width/width) - 20

    */

    // try and click the widget
    const widgetX = pos[0] + pos[3] / 2
    const widgetY = pos[1] + widget + 10
    const [clickX, clickY] = getPos([widgetX, widgetY])
    console.log(clickX, clickY)

    await page.waitForTimeout(2000)

    await page.mouse.click(clickX, clickY)
    await page.getByRole('menuitem', { name: 'Advanced' }).click();

    await page.waitForTimeout(2000)

    await page.mouse.click(clickX, clickY)
    await page.getByRole('menuitem', { name: 'All' }).click();
})
