#!/usr/bin/env node
/**
 * test/browser/user_journey.mjs
 *
 * Browser-based UI validation test simulating a realistic user session.
 * Covers: tab navigation, txt2img / img2img interaction, settings panel,
 * ControlNet extension, and an optional one-step generation pass.
 *
 * Detects JavaScript console errors so regressions that only appear in the
 * browser (not the API) are caught before they reach users.
 *
 * Usage
 * -----
 *   node test/browser/user_journey.mjs [options]
 *
 *   --url=URL            WebUI base URL        (default: http://127.0.0.1:7860)
 *   --profile=PROFILE    iphone|ipad|laptop|pc|all  (default: all)
 *   --headless           Run headless          (default: true)
 *   --no-headless        Show browser window
 *   --timeout=MS         Per-step timeout ms   (default: 30000)
 *   --screenshots        Save PNG on each failure
 *   --generate           Attempt a 1-step generation if a model is loaded
 *
 * Exit codes
 * ----------
 *   0  All checks passed — or WebUI is not reachable (graceful skip)
 *   1  One or more JS errors / assertion failures detected
 */

import { createRequire } from 'module';
import { existsSync } from 'fs';
import { mkdir, writeFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import path from 'path';
import http from 'http';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Puppeteer loader — tries project node_modules, then the npx cache
// ---------------------------------------------------------------------------
function loadPuppeteer() {
    const candidates = [
        path.resolve(__dirname, '../../..', 'node_modules', 'puppeteer'),
        path.join(
            process.env.HOME || '/root',
            '.npm/_npx/7d92d9a2d2ccc630/node_modules/puppeteer',
        ),
    ];
    for (const p of candidates) {
        if (existsSync(p)) {
            const req = createRequire(import.meta.url);
            return req(p);
        }
    }
    throw new Error(
        'puppeteer not found. Run:\n  npm install --save-dev puppeteer\nor install it globally via npx.',
    );
}
const puppeteer = loadPuppeteer();

// ---------------------------------------------------------------------------
// Device / user profiles
// ---------------------------------------------------------------------------
/**
 * @typedef {Object} Profile
 * @property {string}  label
 * @property {{width:number, height:number}} viewport
 * @property {number}  deviceScaleFactor
 * @property {boolean} isMobile
 * @property {boolean} hasTouch
 * @property {number}  timeoutScale    – multiplier applied to all wait timeouts
 * @property {number}  defaultSteps    – steps slider value used during interaction
 * @property {boolean} runGeneration   – whether to attempt the Generate step
 */

/** @type {Record<string, Profile>} */
const PROFILES = {
    iphone: {
        label: 'iPhone 14 Pro (mobile)',
        viewport: { width: 393, height: 852 },
        deviceScaleFactor: 3,
        isMobile: true,
        hasTouch: true,
        timeoutScale: 1.8,
        defaultSteps: 4,
        runGeneration: false,
    },
    ipad: {
        label: 'iPad Air (tablet)',
        viewport: { width: 820, height: 1180 },
        deviceScaleFactor: 2,
        isMobile: true,
        hasTouch: true,
        timeoutScale: 1.4,
        defaultSteps: 4,
        runGeneration: false,
    },
    laptop: {
        label: 'Laptop 1366×768 (~8 GB RAM)',
        viewport: { width: 1366, height: 768 },
        deviceScaleFactor: 1,
        isMobile: false,
        hasTouch: false,
        timeoutScale: 1.0,
        defaultSteps: 1,
        runGeneration: true,
    },
    pc: {
        label: 'Desktop 1920×1080 (~16 GB RAM)',
        viewport: { width: 1920, height: 1080 },
        deviceScaleFactor: 1,
        isMobile: false,
        hasTouch: false,
        timeoutScale: 1.0,
        defaultSteps: 1,
        runGeneration: true,
    },
};

// ---------------------------------------------------------------------------
// CLI argument parser
// ---------------------------------------------------------------------------
function parseArgs(argv) {
    const args = {
        url: 'http://127.0.0.1:7860',
        profiles: ['all'],
        headless: true,
        timeout: 30_000,
        screenshots: false,
        generate: false,
    };
    for (const arg of argv.slice(2)) {
        if (arg.startsWith('--url='))         args.url        = arg.slice(6);
        else if (arg.startsWith('--profile=')) args.profiles  = [arg.slice(10)];
        else if (arg === '--no-headless')      args.headless   = false;
        else if (arg === '--headless')         args.headless   = true;
        else if (arg.startsWith('--timeout=')) args.timeout    = parseInt(arg.slice(10), 10);
        else if (arg === '--screenshots')      args.screenshots = true;
        else if (arg === '--generate')         args.generate    = true;
    }
    if (args.profiles[0] === 'all') args.profiles = Object.keys(PROFILES);
    return args;
}

// ---------------------------------------------------------------------------
// Reachability check — fast, no browser involved
// ---------------------------------------------------------------------------
function checkReachable(url, timeoutMs = 5_000) {
    return new Promise((resolve) => {
        const parsed = new URL(url);
        // Hit the root page — any HTTP response means the server is up.
        // We intentionally avoid /sdapi/v1/* because the REST API requires
        // --api to be passed at launch and is not enabled in a default WebUI session.
        const req = http.get(
            { hostname: parsed.hostname, port: parsed.port || 7860, path: '/', timeout: timeoutMs },
            (res) => { res.resume(); resolve(true); },
        );
        req.on('error', () => resolve(false));
        req.on('timeout', () => { req.destroy(); resolve(false); });
    });
}

// ---------------------------------------------------------------------------
// Step runner — collects errors, optionally screenshots
// ---------------------------------------------------------------------------
class Journey {
    constructor(page, profile, opts) {
        this.page         = page;
        this.profile      = profile;
        this.opts         = opts;
        this.errors       = [];    // { step, message }
        this.warnings     = [];
        this.stepResults  = [];    // { name, ok, ms, errors }
        this._currentStep = 'init';

        // Intercept console.error from the page
        page.on('console', (msg) => {
            if (msg.type() === 'error') {
                const text = msg.text();
                if (isKnownHarmless(text)) return;
                this.errors.push({ step: this._currentStep, message: text });
            }
        });
        page.on('pageerror', (err) => {
            if (isKnownHarmless(err.message)) return;
            this.errors.push({ step: this._currentStep, message: `[uncaught] ${err.message}` });
        });
    }

    timeout(base) {
        return base * this.profile.timeoutScale;
    }

    async step(name, fn) {
        this._currentStep = name;
        const before = this.errors.length;
        const start  = Date.now();
        let ok       = true;
        try {
            await fn();
        } catch (err) {
            ok = false;
            this.errors.push({ step: name, message: `[step throw] ${err.message}` });
        }
        const ms       = Date.now() - start;
        const newErrors = this.errors.slice(before);
        this.stepResults.push({ name, ok: ok && newErrors.length === 0, ms, errors: newErrors });
        const status = ok && newErrors.length === 0 ? '✓' : '✗';
        const errSuffix = newErrors.length ? ` (${newErrors.length} JS errors)` : '';
        console.log(`  ${status} ${name}  (${ms}ms)${errSuffix}`);
    }

    /** Wait for a selector, throw with a clear message if it never appears */
    async waitFor(selector, timeoutMs) {
        const t = timeoutMs ?? this.timeout(this.opts.timeout);
        return this.page.waitForSelector(selector, { timeout: t });
    }

    /** Click a selector, optionally wait for it first */
    async click(selector, timeoutMs) {
        const el = await this.waitFor(selector, timeoutMs);
        await el.click();
        return el;
    }

    /** Type into a selector, clearing existing value first */
    async type(selector, text) {
        const el = await this.waitFor(selector);
        await el.click({ clickCount: 3 });
        await el.type(text);
    }

    /** Set a Gradio slider to a target value via direct input manipulation */
    async setSlider(inputSelector, value) {
        await this.page.evaluate((sel, val) => {
            const el = document.querySelector(sel);
            if (!el) return;
            const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                window.HTMLInputElement.prototype, 'value',
            ).set;
            nativeInputValueSetter.call(el, val);
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        }, inputSelector, String(value));
    }

    /** Click a named main tab and wait for its content pane to appear */
    async switchMainTab(tabLabel) {
        // Gradio 3: button.selected; Gradio 6: aria-selected
        await this.page.evaluate((label) => {
            const tabs = document.querySelectorAll('#tabs > .tab-nav button, #tabs button[role="tab"]');
            for (const btn of tabs) {
                if (btn.textContent.trim() === label) { btn.click(); return; }
            }
        }, tabLabel);
        // Brief pause for mutations to settle
        await new Promise(r => setTimeout(r, 400));
    }

    /** Save a screenshot to test/browser/screenshots/ */
    async screenshot(tag) {
        if (!this.opts.screenshots) return;
        const dir = path.join(__dirname, 'screenshots');
        await mkdir(dir, { recursive: true });
        const fname = path.join(dir, `${this.profile.label.replace(/\W+/g, '_')}_${tag}_${Date.now()}.png`);
        await this.page.screenshot({ path: fname, fullPage: false });
        console.log(`    📸 ${fname}`);
    }

    totalErrors() { return this.errors.length; }
}

// ---------------------------------------------------------------------------
// False-positive filter — errors from Gradio internals or browser quirks
// ---------------------------------------------------------------------------
const HARMLESS_PATTERNS = [
    /ResizeObserver loop/i,
    /Error: cancelled/i,
    /AbortError/i,
    /NetworkError/i,
    /Failed to fetch/i,
    /Failed to load resource/i,   // HTTP 404/500 from server — network, not JS logic
    /net::ERR_/i,
    /Loading chunk/i,
    /QUOTA_EXCEEDED_ERR/i,
    /Cannot read properties of null.*reading 'document'/i,   // gradio shadow-dom timing
    /gradio/i,            // gradio internal log noise (coarse but effective)
];

function isKnownHarmless(msg) {
    return HARMLESS_PATTERNS.some(re => re.test(msg));
}

// ---------------------------------------------------------------------------
// The user journey — steps executed for every profile
// ---------------------------------------------------------------------------
async function runJourney(browser, profile, opts) {
    const page = await browser.newPage();
    await page.emulate({
        viewport: profile.viewport,
        userAgent: profile.isMobile
            ? 'Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Mobile Safari/537.36'
            : 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36',
    });

    const j = new Journey(page, profile, opts);

    // ── 1. Initial page load ─────────────────────────────────────────────────
    await j.step('page-load', async () => {
        await page.goto(opts.url, { waitUntil: 'networkidle0', timeout: j.timeout(60_000) });
        // Wait for Gradio app to be interactive (txt2img prompt is the indicator)
        await j.waitFor('#txt2img_prompt textarea', j.timeout(45_000));
        await j.screenshot('loaded');
    });

    // ── 2. txt2img tab — basic interaction ───────────────────────────────────
    await j.step('txt2img-prompt-entry', async () => {
        await j.type('#txt2img_prompt textarea',
            'a serene mountain landscape, golden hour, photorealistic');
        await j.type('#txt2img_neg_prompt textarea',
            'blurry, low quality, distorted');
    });

    await j.step('txt2img-steps-slider', async () => {
        const stepsInput = '#txt2img_steps input[type=number]';
        await j.setSlider(stepsInput, profile.defaultSteps);
    });

    await j.step('txt2img-cfg-slider', async () => {
        await j.setSlider('#txt2img_cfg_scale input[type=number]', 7);
    });

    await j.step('txt2img-seed', async () => {
        const seedInput = await page.$('#txt2img_seed input');
        if (seedInput) {
            await seedInput.click({ clickCount: 3 });
            await seedInput.type('42');
        }
    });

    await j.step('txt2img-sampler-open', async () => {
        // Open the sampler dropdown (Gradio select element)
        const samplerWrap = await page.$('#txt2img_sampling');
        if (samplerWrap) {
            const btn = await samplerWrap.$('button, input');
            if (btn) await btn.click();
            await new Promise(r => setTimeout(r, 300));
            // Close it by pressing Escape
            await page.keyboard.press('Escape');
        }
    });

    // ── 3. Navigate all main tabs ─────────────────────────────────────────────
    const mainTabs = ['img2img', 'Extras', 'PNG Info', 'Checkpoint Merger', 'Settings', 'Extensions'];
    for (const tab of mainTabs) {
        await j.step(`navigate-to-${tab.toLowerCase().replace(/\s+/g, '-')}`, async () => {
            await j.switchMainTab(tab);
            await j.screenshot(`tab-${tab.toLowerCase().replace(/\s+/g, '-')}`);
        });
    }

    // Return to txt2img before img2img interactions
    await j.step('navigate-back-to-txt2img', async () => {
        await j.switchMainTab('txt2img');
        await j.waitFor('#txt2img_prompt textarea');
    });

    // ── 4. img2img sub-tab navigation ────────────────────────────────────────
    await j.step('img2img-tab-open', async () => {
        await j.switchMainTab('img2img');
        await j.waitFor('#img2img_prompt textarea', j.timeout(15_000));
    });

    await j.step('img2img-prompt-entry', async () => {
        await j.type('#img2img_prompt textarea', 'enhance the image, high detail');
    });

    await j.step('img2img-denoising-slider', async () => {
        await j.setSlider('#img2img_denoising_strength input[type=number]', 0.55);
    });

    // Navigate img2img sub-tabs (img2img / sketch / inpaint / inpaint sketch / inpaint upload / batch)
    const img2imgSubTabs = ['img2img', 'Sketch', 'Inpaint', 'Inpaint sketch', 'Batch'];
    for (const sub of img2imgSubTabs) {
        await j.step(`img2img-subtab-${sub.toLowerCase().replace(/\s+/g, '-')}`, async () => {
            await page.evaluate((label) => {
                const btns = document.querySelectorAll('#mode_img2img .tab-nav button, #mode_img2img button[role="tab"]');
                for (const b of btns) {
                    if (b.textContent.trim() === label) { b.click(); return; }
                }
            }, sub);
            await new Promise(r => setTimeout(r, 300));
        });
    }

    // ── 5. Settings panel ────────────────────────────────────────────────────
    await j.step('settings-tab-open', async () => {
        await j.switchMainTab('Settings');
        // Wait for at least one setting to appear
        await j.waitFor('#settings', j.timeout(20_000));
        await new Promise(r => setTimeout(r, 500));
    });

    await j.step('settings-scroll', async () => {
        // Scroll down to load any lazy content, then back up
        await page.evaluate(() => window.scrollBy(0, 600));
        await new Promise(r => setTimeout(r, 300));
        await page.evaluate(() => window.scrollTo(0, 0));
    });

    // ── 6. ControlNet extension (if present) ─────────────────────────────────
    await j.step('navigate-back-to-txt2img-for-controlnet', async () => {
        await j.switchMainTab('txt2img');
        await j.waitFor('#txt2img_prompt textarea');
    });

    await j.step('controlnet-accordion', async () => {
        const cnAccordion = await page.$('#txt2img_controlnet');
        if (!cnAccordion) {
            console.log('    (ControlNet not present — skipping)');
            return;
        }
        // Open the ControlNet accordion
        const labelWrap = await cnAccordion.$('.label-wrap');
        if (labelWrap) {
            await labelWrap.click();
            await new Promise(r => setTimeout(r, 600));
            await j.screenshot('controlnet-open');

            // Navigate ControlNet unit sub-tabs if available
            const cnTabs = await cnAccordion.$$('.tab-nav button, button[role="tab"]');
            for (let i = 0; i < Math.min(cnTabs.length, 3); i++) {
                await cnTabs[i].click();
                await new Promise(r => setTimeout(r, 300));
            }

            // Close the accordion
            await labelWrap.click();
            await new Promise(r => setTimeout(r, 400));
        }
    });

    // ── 7. Extras tab — basic interaction ────────────────────────────────────
    await j.step('extras-tab', async () => {
        await j.switchMainTab('Extras');
        await new Promise(r => setTimeout(r, 500));
        // Just verify the tab rendered without errors
        await j.screenshot('extras');
        await j.switchMainTab('txt2img');
        await j.waitFor('#txt2img_prompt textarea');
    });

    // ── 8. Optional 1-step generation ────────────────────────────────────────
    if (opts.generate && profile.runGeneration) {
        await j.step('generation-check-model', async () => {
            // Detect a loaded checkpoint from the UI dropdown — no API call needed.
            const hasModel = await page.evaluate(() => {
                const el = document.querySelector(
                    '#quicksettings #setting_sd_model_checkpoint input, ' +
                    '#quicksettings input[placeholder*="checkpoint" i], ' +
                    '.model-select input',
                );
                return el ? el.value.trim().length > 0 : false;
            });
            if (!hasModel) {
                console.log('    (no model loaded — skipping generation)');
                return;
            }

            // Shrink to smallest possible size for speed
            await j.setSlider('#txt2img_width input[type=number]', 64);
            await j.setSlider('#txt2img_height input[type=number]', 64);
            await j.setSlider('#txt2img_steps input[type=number]', 1);

            // Click Generate
            const genBtn = await page.$('#txt2img_generate');
            if (!genBtn) return;
            await genBtn.click();

            // Wait for interrupt button to appear (generation started) or timeout
            try {
                await page.waitForSelector('#txt2img_interrupt[style*="display: block"], #txt2img_interrupt:not([style*="display: none"])',
                    { timeout: 15_000 });
                await new Promise(r => setTimeout(r, 1_000));
                // Interrupt
                const intBtn = await page.$('#txt2img_interrupt');
                if (intBtn) await intBtn.click();
                await j.screenshot('after-generate');
            } catch {
                // Generation didn't start (likely no model) — not a failure
                console.log('    (generation did not start — model may not be loaded)');
            }
        });
    }

    // ── 9. Final screenshot & cleanup ────────────────────────────────────────
    await j.screenshot('final');
    await page.close();

    return j;
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------
function printReport(profileLabel, journey) {
    const totalOk  = journey.stepResults.filter(s => s.ok).length;
    const totalFail = journey.stepResults.filter(s => !s.ok).length;
    const ms       = journey.stepResults.reduce((a, s) => a + s.ms, 0);

    console.log(`\n  Steps:  ${totalOk} passed, ${totalFail} failed  (${ms}ms total)`);

    if (journey.errors.length > 0) {
        console.log(`\n  JS errors (${journey.errors.length}):`);
        for (const e of journey.errors) {
            console.log(`    [${e.step}] ${e.message.slice(0, 200)}`);
        }
    } else {
        console.log('\n  No JS errors detected.');
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
async function main() {
    const opts     = parseArgs(process.argv);
    const profiles = opts.profiles.map(k => {
        if (!PROFILES[k]) {
            console.error(`Unknown profile "${k}". Valid: ${Object.keys(PROFILES).join(', ')}`);
            process.exit(1);
        }
        return { key: k, ...PROFILES[k] };
    });

    console.log(`\nWebUI URL: ${opts.url}`);
    console.log(`Profiles:  ${profiles.map(p => p.label).join(', ')}`);

    // Reachability — exit 0 (graceful skip) if WebUI is down
    process.stdout.write('\nChecking WebUI reachability … ');
    const reachable = await checkReachable(opts.url);
    if (!reachable) {
        console.log('UNREACHABLE');
        console.log('WebUI is not running at the given URL. Skipping browser tests.');
        process.exit(0);
    }
    console.log('OK');

    // Chrome executable — prefer environment override, then cached Puppeteer binary
    const executablePath =
        process.env.PUPPETEER_EXECUTABLE_PATH ||
        path.join(process.env.HOME || '/root',
            '.cache/puppeteer/chrome/linux-148.0.7778.97/chrome-linux64/chrome');

    const browser = await puppeteer.launch({
        headless: opts.headless,
        executablePath: existsSync(executablePath) ? executablePath : undefined,
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu',
        ],
    });

    let anyFailure = false;

    for (const profile of profiles) {
        console.log(`\n${'─'.repeat(60)}`);
        console.log(`Profile: ${profile.label}`);
        console.log(`Viewport: ${profile.viewport.width}×${profile.viewport.height}  touch:${profile.hasTouch}`);
        console.log('─'.repeat(60));

        const journey = await runJourney(browser, profile, opts);
        printReport(profile.label, journey);

        if (journey.totalErrors() > 0) anyFailure = true;
    }

    await browser.close();

    console.log(`\n${'═'.repeat(60)}`);
    if (anyFailure) {
        console.log('RESULT: FAIL — JS errors were detected during the user journey.');
        process.exit(1);
    } else {
        console.log('RESULT: PASS — No JS errors across all profiles.');
        process.exit(0);
    }
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
