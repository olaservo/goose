const { chromium } = require('C:/Users/johnn/OneDrive/Documents/GitHub/_bellafoo/goose/ui/node_modules/@playwright/test');
const out = process.argv[2];
(async () => {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9333');
  const page = browser.contexts().flatMap(c => c.pages())[0];
  await page.evaluate(() => { window.location.hash = '#/'; });
  await page.waitForTimeout(2000);
  const input = page.locator('[data-testid="chat-input"], textarea').first();
  await input.click();
  await input.fill('Reply with the single word: ready');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(25000);
  await page.screenshot({ path: `${out}/03-chat.png` });
  await page.evaluate(() => { window.location.hash = '#/skills'; });
  await page.waitForTimeout(6000);
  const r = await page.evaluate(() => {
    const t = document.body.innerText;
    return { mcp: (t.match(/MCP · [^\n]+/g) || []), badges: (t.match(/\n(attributed|partial|uncredited)\n/g) || []).map(s => s.trim()) };
  });
  console.log(JSON.stringify(r));
  await page.screenshot({ path: `${out}/04-skills-top.png` });
  const tag = page.locator('text=/MCP · /').first();
  if (await tag.count()) {
    await tag.scrollIntoViewIfNeeded();
    await page.waitForTimeout(800);
    await page.screenshot({ path: `${out}/05-skills-mcp.png` });
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
