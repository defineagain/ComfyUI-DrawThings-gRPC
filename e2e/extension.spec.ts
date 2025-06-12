import 'dotenv/config'
import { test, expect } from '@playwright/test';

const comfyUrl = process.env.PLAYWRIGHT_TEST_URL
if (!comfyUrl) throw new Error('PLAYWRIGHT_TEST_URL is not set')

test('has title', async ({ page }) => {
  await page.goto(comfyUrl);

  // Expect a title "to contain" a substring.
  await expect(page).toHaveTitle(/Playwright/);
});
