import { expect as baseExpect } from "@playwright/test";

export const expect = baseExpect.extend({
    toEachBeTruthy(items: any[]) {
        const assertionName = "toEachBeTruthy";

        const itemsPass = items.map((item) => {
            try {
                baseExpect(item).toBeTruthy();
                return true;
            } catch (e) {
                return false;
            }
        });

        const pass = itemsPass.every((item) => item === true);

        const message = pass
            ? () =>
                  this.utils.matcherHint(assertionName, undefined, undefined, {
                      isNot: this.isNot,
                  }) +
                  "\n\n" +
                  `Expected all items to be truthy, and they were.\n` +
                  `Expected: ${new Array(items.length).fill(true)}\n` +
                  `Received: ${itemsPass}`
            : () =>
                  this.utils.matcherHint(assertionName, undefined, undefined, {
                      isNot: this.isNot,
                  }) +
                  "\n\n" +
                  `Expected all items to be truthy, but at least one was not.\n` +
                  `Expected: ${new Array(items.length).fill(true)}\n` +
                `Received: ${itemsPass}`;

        return {
            message,
            pass,
            expected: new Array(items.length).fill(true),
            received: itemsPass,
            actual: itemsPass,
            name: assertionName,
        }
    },

    toEachBeFalsy(items: any[]) {
        const assertionName = "toEachBeFalsy";

        const itemsPass = items.map((item) => {
            try {
                baseExpect(item).toBeFalsy();
                return false;
            } catch (e) {
                return true;
            }
        });

        const pass = itemsPass.every((item) => item === false);

        const message = pass
            ? () =>
                  this.utils.matcherHint(assertionName, undefined, undefined, {
                      isNot: this.isNot,
                  }) +
                  "\n\n" +
                  `Expected all items to be falsy, and they were.\n` +
                  `Expected: ${new Array(items.length).fill(false)}\n` +
                  `Received: ${itemsPass}`
            : () =>
                  this.utils.matcherHint(assertionName, undefined, undefined, {
                      isNot: this.isNot,
                  }) +
                  "\n\n" +
                  `Expected all items to be falsy, but at least one was not.\n` +
                  `Expected: ${new Array(items.length).fill(false)}\n` +
                  `Received: ${itemsPass}`;

        return {
            message,
            pass,
            expected: new Array(items.length).fill(false),
            received: itemsPass,
            actual: itemsPass,
            name: assertionName,
        }
    },
});
