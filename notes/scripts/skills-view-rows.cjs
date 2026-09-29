const { chromium } = require('C:/Users/johnn/OneDrive/Documents/GitHub/_bellafoo/goose/ui/node_modules/@playwright/test');
const out = process.argv[2];
(async () => {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9333');
  const page = browser.contexts().flatMap(c => c.pages())[0];
  const cards = page.locator('div:has(> div > div > div > span:text-matches("^MCP · "))').filter({ has: page.locator('h3') });
  const rows = page.locator('h3').locator('xpath=ancestor::div[contains(@class,"mb-2")][1]').filter({ hasText: /MCP · / });
  const n = await rows.count();
  for (let i = 0; i < n; i++) {
    const row = rows.nth(i);
    const name = (await row.locator('h3').innerText()).trim();
    const lines = (await row.innerText()).split('\n').map(s => s.trim()).filter(Boolean);
    console.log(JSON.stringify({ name, tag: lines[1], badge: lines[2], credit: lines[lines.length - 1] === lines[3] ? '' : lines[lines.length - 1] }));
    await row.scrollIntoViewIfNeeded();
    await row.screenshot({ path: `${out}/row-${String(i + 1).padStart(2, '0')}-${name}.png` });
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
