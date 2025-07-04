import test from '@playwright/test';

test("models update with server", async ({ page }) => {
    // load workflow
    // set server to invalid address
    // assert error model options
    // sampler, lora, cnet, prompt, upscaler, refiner
    // set server to valid address
    // assert model options are updated
    // sampler, lora, cnet, prompt, upscaler, refiner
})

test("sampler model version", async ({ page }) => {
    // load workflow
    // select (none selected)
    // assert lora, cnet, prompt node models are all enabled
    // select an sd1 model
    // assert non-sd1 models are disabled
    // select an sdxl model
    // assert sd1 models are disabled, and sdxl models are enabled
})

test("single server workflow", async ({ page }) => { })

test("multiple server workflow", async ({ page }) => { })

test("save/restore previous model", async ({ page }) => { })
