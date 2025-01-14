const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({
    headless: false,
  });

  const page = await browser.newPage();

  // Listen for console messages
  page.on('console', msg => {
    if (msg.text().startsWith('<puppeteer>')) {
      // Handle puppeteer command
      const command = msg.text().substring('<puppeteer>'.length);
      if (command === 'shutdown') {
        browser.close();
        process.exit(0);
      }
    } else {
      console.log(msg.text());
    }
});

  // Listen for page.close() requests
  page.on('close', () => {
    console.log('## Page requested close');
    browser.close();
    process.exit(0);
  });

  // Listen for page errors
  page.on('pageerror', error => {
    console.error('## Page error:', error);
    browser.close();
    process.exit(1);
  });

  // Listen for worker errors
  page.on('workercreated', worker => {
    worker.on('error', error => {
      console.error('## Worker error:', error);
      browser.close();
      process.exit(1);
    });
  });

  await page.goto('http://localhost:3000');
})();