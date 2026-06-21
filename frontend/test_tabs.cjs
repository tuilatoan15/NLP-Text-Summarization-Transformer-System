const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const ARTIFACT_DIR = 'C:\\Users\\ASUS\\.gemini\\antigravity\\brain\\8d82049c-9aba-4bd8-abf5-3c62d3989bb7';
const SCREENSHOT_DIR = path.join(ARTIFACT_DIR, 'screenshots');

if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}

const TABS = [
  { name: 'Dashboard', url: 'http://localhost:5173/' },
  { name: 'Summarization', url: 'http://localhost:5173/summarize' },
  { name: 'AI Chat', url: 'http://localhost:5173/chat' },
  { name: 'Semantic Search', url: 'http://localhost:5173/search' },
  { name: 'Model Comparison', url: 'http://localhost:5173/compare' },
  { name: 'Analytics', url: 'http://localhost:5173/analytics' },
  { name: 'Datasets', url: 'http://localhost:5173/documents' },
  { name: 'Models', url: 'http://localhost:5173/benchmark' },
  { name: 'Settings', url: 'http://localhost:5173/settings' }
];

async function run() {
  console.log('Connecting to Chrome on port 9555...');
  let browser;
  try {
    browser = await puppeteer.connect({
      browserURL: 'http://localhost:9555',
      defaultViewport: { width: 1280, height: 800 }
    });
    console.log('Connected successfully!');
  } catch (err) {
    console.error('Failed to connect to Chrome:', err);
    process.exit(1);
  }

  const results = [];

  for (const tab of TABS) {
    console.log(`\nNavigating to ${tab.name} (${tab.url})...`);
    const page = await browser.newPage();
    const consoleLogs = [];
    const consoleErrors = [];

    page.on('console', msg => {
      const type = msg.type();
      const text = msg.text();
      if (type === 'error' || type === 'warning' || type === 'warn') {
        consoleErrors.push(`[${type}] ${text}`);
      } else {
        consoleLogs.push(`[${type}] ${text}`);
      }
    });

    page.on('pageerror', err => {
      consoleErrors.push(`[Uncaught Error] ${err.toString()}`);
    });

    try {
      await page.goto(tab.url, { waitUntil: 'networkidle0', timeout: 15000 });
      console.log(`Page loaded: ${tab.name}`);
    } catch (err) {
      console.log(`Navigation timeout or error on ${tab.name}, waiting 3 more seconds...`);
      try {
        await new Promise(resolve => setTimeout(resolve, 3000));
      } catch(e) {}
    }

    // Check if the page is blank by checking body content
    const bodyContent = await page.evaluate(() => document.body.innerText.trim());
    const isBlank = bodyContent.length === 0;

    // Take screenshot
    const screenshotFilename = `${tab.name.toLowerCase().replace(/ /g, '_')}.png`;
    const screenshotPath = path.join(SCREENSHOT_DIR, screenshotFilename);
    await page.screenshot({ path: screenshotPath });
    console.log(`Screenshot saved to ${screenshotPath}`);

    results.push({
      tabName: tab.name,
      url: tab.url,
      isBlank,
      consoleLogs,
      consoleErrors,
      screenshotFile: screenshotPath,
      screenshotRelative: `screenshots/${screenshotFilename}`
    });

    await page.close();
  }

  await browser.disconnect();

  // Write results to JSON
  const resultsPath = path.join(ARTIFACT_DIR, 'test_results.json');
  fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
  console.log(`\nAll tests completed! Results written to ${resultsPath}`);
}

run();
