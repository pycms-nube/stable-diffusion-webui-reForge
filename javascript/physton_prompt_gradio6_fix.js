// Fixes sd-webui-prompt-all-in-one positioning under Gradio 6.x.
//
// In Gradio 3.x the extension did:  $prompt.parentElement.parentElement.after($physton)
// which landed right after the prompt Row.  Gradio 6.x adds extra wrapper divs so the
// same traversal overshoots to the Column level, dumping physton at the bottom.
//
// This shim watches only for physton container nodes being *added*, relocates them once,
// then disconnects to avoid fighting Gradio's Svelte reconciler.

(function () {
    "use strict";

    const TARGETS = [
        { phystonId: "phystonPrompt_txt2img_prompt",    rowId: "txt2img_prompt_row" },
        { phystonId: "phystonPrompt_txt2img_neg_prompt", rowId: "txt2img_neg_prompt_row" },
        { phystonId: "phystonPrompt_img2img_prompt",    rowId: "img2img_prompt_row" },
        { phystonId: "phystonPrompt_img2img_neg_prompt", rowId: "img2img_neg_prompt_row" },
    ];

    // Track which IDs have already been relocated so we never move them twice.
    const relocated = new Set();

    function getRoot() {
        const apps = document.getElementsByTagName("gradio-app");
        if (apps.length > 0 && apps[0].shadowRoot) return apps[0].shadowRoot;
        return document;
    }

    function relocateOne(phystonId, rowId, root) {
        if (relocated.has(phystonId)) return false;

        const physton = root.querySelector("#" + phystonId);
        const row     = root.querySelector("#" + rowId);
        if (!physton || !row) return false;

        // Already in the right spot — nothing to do.
        if (row.nextElementSibling === physton) {
            relocated.add(phystonId);
            return false;
        }

        // Disconnect before touching the DOM so our own mutation doesn't re-trigger.
        stopObserving();
        row.after(physton);
        relocated.add(phystonId);

        // Re-arm only if there are still physton elements left to place.
        if (relocated.size < TARGETS.length) startObserving();
        return true;
    }

    function tryAll() {
        const root = getRoot();
        let moved = 0;
        for (const { phystonId, rowId } of TARGETS) {
            if (relocateOne(phystonId, rowId, root)) moved++;
        }
        return moved;
    }

    // ── Observer lifecycle ────────────────────────────────────────────────────

    let observer = null;

    function onMutation(mutations) {
        // Only act when a physton container node was actually added to the DOM.
        let relevantAdd = false;
        for (const rec of mutations) {
            for (const node of rec.addedNodes) {
                if (node.nodeType !== 1) continue;
                const id = node.id || "";
                if (id.startsWith("phystonPrompt_")) { relevantAdd = true; break; }
                // physton might be inserted inside a larger subtree — check descendants.
                if (node.querySelector && node.querySelector("[id^='phystonPrompt_']")) {
                    relevantAdd = true; break;
                }
            }
            if (relevantAdd) break;
        }
        if (relevantAdd) tryAll();
    }

    function startObserving() {
        if (observer) return;
        const root   = getRoot();
        const target = root === document ? document.body : root;
        observer = new MutationObserver(onMutation);
        observer.observe(target, { childList: true, subtree: true });
    }

    function stopObserving() {
        if (!observer) return;
        observer.disconnect();
        observer = null;
    }

    function init() {
        // Already placed (e.g. hot-reload)?
        tryAll();
        // Only arm the observer if there's still work to do.
        if (relocated.size < TARGETS.length) startObserving();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
