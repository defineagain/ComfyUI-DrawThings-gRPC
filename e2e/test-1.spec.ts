import { test, expect } from '@playwright/test';

test('test', async ({ page }) => {
  await page.goto('http://localhost:8188/');
  await page.locator('a').filter({ hasText: /^Workflow$/ }).click();
  await page.getByText('OpenCtrl + o').click();
  await page.locator('#pv_id_10_list').setInputFiles('dt-grpc-text2image-test.json');
  await page.getByRole('button', { name: 'Close' }).click();
  await page.locator('#graph-canvas').click({
    position: {
      x: 667,
      y: 206
    }
  });
});
