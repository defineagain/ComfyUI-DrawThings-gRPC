import "dotenv/config"
import { Locator, Page, test as base, expect } from '@playwright/test'
import { NodeRef } from './nodeRef'


class ComfyPage {
    readonly canvas: Locator
    readonly url: string

    constructor(public readonly page: Page) {
        this.canvas = page.locator("#graph-canvas")

        if (!process.env.PLAYWRIGHT_TEST_URL) {
            throw new Error("PLAYWRIGHT_TEST_URL is not set")
        }
        this.url = process.env.PLAYWRIGHT_TEST_URL
    }

    async goto() {
        await this.page.goto(this.url)
    }

    async openWorkflow(workflowPath: string) {
        // Expect a title "to contain" a substring.
        await expect(this.page).toHaveTitle(/ComfyUI/);

        const fileChooserPromise = this.page.waitForEvent("filechooser");

        await this.page
            .locator("a")
            .filter({ hasText: /^Workflow$/ })
            .click();
        await this.page.getByText("OpenCtrl + o").click();

        const fileChooser = await fileChooserPromise;
        await fileChooser.setFiles(workflowPath);

        await this.page.locator("#graph-canvas").click({
            position: {
                x: 137,
                y: 246,
            },
        });

        await this.page.waitForTimeout(200);
        await this.page.locator("#graph-canvas").press(".");
        await this.page.waitForTimeout(200);
    }

    async getNodeRef(
        node: number | string | ((node) => boolean),
        options?: {
            doNotThrow?: boolean;
        }
    ) {
        const nodeId = await this.page.evaluate((node) => {
            if (typeof node === "number") {
                return window.app.graph._nodes[node]?.id;
            } else if (typeof node === "string") {
                return window.app.graph._nodes.find((n) => n.type === node)?.id;
            } else if (typeof node === "function") {
                return window.app.graph._nodes.find(node)?.id;
            }
            return null;
        }, node);

        if (nodeId === null || nodeId === undefined) {
            if (options?.doNotThrow) return undefined;
            throw new Error(`Node not found: ${node}`);
        }

        return new NodeRef(nodeId, this.page);
    }

    async addNode(path: string[], x: number, y: number) {
        const beforeNodes = await this.getAllNodes()
        await this.centerOnPoint(x, y);
        await this.page.waitForTimeout(200)
        const canvasSize = await this.canvas.boundingBox()
        await this.canvas.click({
            position: {
                x: canvasSize!.width / 2,
                y: canvasSize!.height / 2,
            },
            button: "right",
        })

        await this.page.getByRole("menuitem", { name: "Add node" }).first().click();

        for (const p of path) {
            const menu = await this.page.locator(".litecontextmenu").last()
            await menu.getByText(p, { exact: true }).click();
        }

        const afterNodes = await this.getAllNodes()

        const newNode = afterNodes.find((n) => !beforeNodes.map((n) => n.id).includes(n.id))
        if (!newNode) {
            throw new Error("add node failed")
        }
        return new NodeRef(newNode.id, this.page)
    }

    async centerOnPoint(x: number, y: number, scale = 1) {
        await this.page.evaluate(async ([x, y, scale]) => {
            if (!window.app?.canvas.ds) return
            window.app.canvas.ds.scale = 1
            const [_x, _y, w, h] = window.app.canvas.visible_area
            window.app.canvas.ds.offset = [x + w / 2, y + h / 2]
            window.app.canvas.setDirty(true, true)
            await new Promise((resolve) => setTimeout(resolve, 200));
        }, [x, y, scale])
    }

    async createNewWorkflow() {
        await this.page.locator('a').filter({ hasText: /^Workflow$/ }).click();
        await this.page.getByRole('menuitem', { name: 'New' }).locator('a').click();

        await this.page.evaluate(() => {
            if (!window.app?.canvas.ds) return
            window.app.canvas.ds.offset = [50, 50]
            window.app.canvas.ds.scale = 0.8
            window.app.canvas.setDirty(true, true)
        })
    }

    async getAllNodes() {
        return this.page.evaluate(() => {
            return window.app.graph._nodes.map((n) => {
                const { id, type, pos, size, title } = n
                return { id, type, pos, size, title }
            })
        })
    }
}

type ComfyFixtures = {
    comfy: ComfyPage
}

export const test = base.extend<ComfyFixtures>({
    comfy: async ({ page }, use) => {
        await use(new ComfyPage(page))
    },
})
