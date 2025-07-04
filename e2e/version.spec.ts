import test from '@playwright/test';

test("load workflow from previous version", async ({ page }) => {
    // load workflow

    // assert toast appears

    // check properties

    // assert prompt nodes were replaced
})

test("sampler widgets serialization", async ({ page }) => {
    // start with empty workflow

    // add sampler node

    // change various widget values

    // export workflow

    // assert workflow file has widget values by key

    // scramble the values in the array in the workflow file

    // load the workflow

    // assert values were loaded by key
})
