import { Page } from "@playwright/test";

export async function getNodeRef(
    page: Page,
    node: number | string | ((node) => boolean)
) {
    const nodeId = await page.evaluate((node) => {
        if (typeof node === "number") {
            return window.app.graph._nodes[node].id;
        } else if (typeof node === "string") {
            return window.app.graph._nodes.find((n) => n.type === node).id;
        } else if (typeof node === "function") {
            return window.app.graph._nodes.find(node).id;
        }
        return null;
    }, node);

    if (nodeId === null) {
        throw new Error(`Node not found: ${node}`);
    }

    return new NodeRef(nodeId, page);
}

export class NodeRef {
    readonly id: number;
    readonly page: Page;

    delay: number = 100;

    constructor(id: number, page: Page) {
        this.page = page;
        this.id = id;
    }

    async clickWidget(name: string) {
        const { view, pos, widgetPos, size } = await this.page.evaluate(
            ([nodeId, name]) => {
                const view: [number, number, number, number] =
                    window.app.canvas.visible_area;
                const node = app.graph.getNodeById(nodeId);
                const pos: [number, number, number, number] = node._posSize;
                const widget = node.widgets.find((w) => w.name === name);
                if (!widget) {
                    throw new Error(`Widget not found: ${name}`);
                }
                const widgetPos: number = widget.y;

                const canvas = window.app.canvas.canvas.getBoundingClientRect();
                const size = {
                    width: canvas.width,
                    height: canvas.height,
                };

                return { view, pos, widgetPos, size };
            },
            [this.id, name]
        );

        const getPos = ([x, y]: [number, number]) => {
            // view: [viewX, viewY, viewWidth, viewHeight]
            // size: { width, height }
            const [viewX, viewY, viewWidth, viewHeight] = view;
            const { width: canvasWidth, height: canvasHeight } = size;

            const canvasX = ((x - viewX) / viewWidth) * canvasWidth;
            const canvasY = ((y - viewY) / viewHeight) * canvasHeight;

            return [canvasX, canvasY];
        };

        // try and click the widget
        const widgetX = pos[0] + pos[2] / 2;
        const widgetY = pos[1] + widgetPos + 10;
        const [clickX, clickY] = getPos([widgetX, widgetY]);

        await this.page
            .locator("#graph-canvas")
            .click({ position: { x: clickX, y: clickY } });
    }

    async isWidgetVisible(name: string);
    async isWidgetVisible(names: string[]);
    async isWidgetVisible(arg: string | string[]) {
        const names = Array.isArray(arg) ? arg : [arg];

        const widgets = await this.page.evaluate(
            async ([nodeId, names, delay]) => {
                await new Promise((resolve) => setTimeout(resolve, delay));
                const node = app.graph.getNodeById(nodeId);
                const isVisible: boolean[] = [];
                for (const name of names) {
                    const w = node.widgets.find((w) => w.name === name);
                    if (!w || w.hidden) {
                        isVisible.push(false);
                        continue;
                    }
                    isVisible.push(true);
                }
                return isVisible;
            },
            [this.id, names, this.delay]
        );

        if (Array.isArray(arg)) {
            return widgets;
        } else {
            return widgets[0];
        }
        return widgets.every((v) => v === true);
    }

    async isWidgetDisabled(name: string);
    async isWidgetDisabled(names: string[]);
    async isWidgetDisabled(arg: string | string[]) {
        const names = Array.isArray(arg) ? arg : [arg];

        const widgets = await this.page.evaluate(
            async ([nodeId, names, delay]) => {
                await new Promise((resolve) => setTimeout(resolve, delay));
                const node = app.graph.getNodeById(nodeId);
                const isDisabled: boolean[] = [];
                for (const name of names) {
                    const w = node.widgets.find((w) => w.name === name);
                    if (!w || w.disabled !== true) {
                        isDisabled.push(false);
                        continue;
                    }
                    isDisabled.push(true);
                }
                return isDisabled;
            },
            [this.id, names, this.delay]
        );

        if (Array.isArray(arg)) {
            return widgets;
        } else {
            return widgets[0];
        }
    }

    async getWidgetValue(name: string) {
        return await this.page.evaluate(
            ([nodeId, name]) => {
                const node = app.graph.getNodeById(nodeId);
                const widget = node.widgets.find((w) => w.name === name);
                if (!widget) throw new Error(`Widget not found: ${name}`);
                return widget.value;
            },
            [this.id, name]
        );
    }

    async selectWidgetOption(widget: string, option: string | RegExp) {
        await this.clickWidget(widget);
        await this.page.getByRole("menuitem", { name: option }).first().click();
    }

    async getNodeColor() {
        throw new Error("not yet implemented")
    }

    async connectOutput(output: string, node: NodeRef, input: string) {
        throw new Error("not yet implemented")
    }

    async disconnectInput(input: string) {
        throw new Error("not yet implemented")
    }

    async getWidgetOptions(widget: string) {
        throw new Error("not yet implemented")
    }
}
