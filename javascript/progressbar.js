// code related to showing and updating progressbar shown as the image is being made

function rememberGallerySelection() {

}

function getGallerySelectedIndex() {

}

function pad2(x) {
    return x < 10 ? '0' + x : x;
}

function formatTime(secs) {
    if (secs > 3600) {
        return pad2(Math.floor(secs / 60 / 60)) + ":" + pad2(Math.floor(secs / 60) % 60) + ":" + pad2(Math.floor(secs) % 60);
    } else if (secs > 60) {
        return pad2(Math.floor(secs / 60)) + ":" + pad2(Math.floor(secs) % 60);
    } else {
        return Math.floor(secs) + "s";
    }
}


let originalAppTitle = undefined;

onUiLoaded(function() {
    originalAppTitle = document.title;
});

function setTitle(progress) {
    let title = originalAppTitle;

    if (opts.show_progress_in_title && progress) {
        title = '[' + progress.trim() + '] ' + title;
    }

    if (document.title != title) {
        document.title = title;
    }
}


function randomId() {
    return "task(" + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + ")";
}

// Starts an EventSource connection to /internal/progress-stream, creating a progressbar
// above progressbarContainer and a live preview inside gallery. One persistent stream
// carries both progress info and live preview frames (the server used to require two
// separate polling loops for this -- see PHASE5.md/PHASE6.md), cutting request volume
// roughly in half and pushing updates instead of waiting for the next poll tick.
// Cleans up everything when the task is over and calls atEnd. Calls onProgress on every
// update, same contract as before this rewrite -- ui.js/extensions.js/textualInversion.js
// call this and don't need to change.
function requestProgress(id_task, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout = 40) {
    const dateStart = new Date();
    let wasEverActive = false;
    const parentProgressbar = progressbarContainer.parentNode;
    let wakeLock = null;
    let eventSource = null;
    let livePreview = null;
    let removed = false;
    let consecutiveErrors = 0;
    const MAX_CONSECUTIVE_ERRORS = 3;

    const requestWakeLock = async function() {
        if (!opts.prevent_screen_sleep_during_generation || wakeLock !== null) return;
        try {
            wakeLock = await navigator.wakeLock.request('screen');
        } catch (err) {
            console.error('Wake Lock is not supported.');
            wakeLock = false;
        }
    };

    const releaseWakeLock = async function() {
        if (!opts.prevent_screen_sleep_during_generation || !wakeLock) return;
        try {
            await wakeLock.release();
            wakeLock = null;
        } catch (err) {
            console.error('Wake Lock release failed', err);
        }
    };

    const divProgress = document.createElement('div');
    divProgress.className = 'progressDiv';
    divProgress.style.display = opts.show_progressbar ? "block" : "none";
    const divInner = document.createElement('div');
    divInner.className = 'progress';

    divProgress.appendChild(divInner);
    parentProgressbar.insertBefore(divProgress, progressbarContainer);

    const removeProgressBar = function() {
        if (removed) return;
        removed = true;

        releaseWakeLock();
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }

        setTitle("");
        parentProgressbar.removeChild(divProgress);
        if (gallery && livePreview) gallery.removeChild(livePreview);
        atEnd();
    };

    const renderLivePreview = function(res) {
        if (!res.live_preview || !gallery) return;

        const img = new Image();
        img.onload = function() {
            if (!livePreview) {
                livePreview = document.createElement('div');
                livePreview.className = 'livePreview';
                gallery.insertBefore(livePreview, gallery.firstElementChild);
            }

            livePreview.appendChild(img);
            if (livePreview.childElementCount > 2) {
                livePreview.removeChild(livePreview.firstElementChild);
            }
        };
        img.src = res.live_preview;
    };

    const handleUpdate = function(res) {
        requestWakeLock();

        if (res.completed) {
            removeProgressBar();
            return;
        }

        let progressText = "";

        divInner.style.width = ((res.progress || 0) * 100.0) + '%';
        divInner.style.background = res.progress ? "" : "transparent";

        if (res.progress > 0) {
            progressText = ((res.progress || 0) * 100.0).toFixed(0) + '%';
        }

        if (res.eta) {
            progressText += " ETA: " + formatTime(res.eta);
        }

        setTitle(progressText);

        if (res.textinfo && res.textinfo.indexOf("\n") == -1) {
            progressText = res.textinfo + " " + progressText;
        }

        divInner.textContent = progressText;

        const elapsedFromStart = (new Date() - dateStart) / 1000;

        if (res.active) wasEverActive = true;

        if (!res.active && wasEverActive) {
            removeProgressBar();
            return;
        }

        if (elapsedFromStart > inactivityTimeout && !res.queued && !res.active) {
            removeProgressBar();
            return;
        }

        renderLivePreview(res);

        if (onProgress) {
            onProgress(res);
        }
    };

    const params = new URLSearchParams({id_task: id_task, live_preview: gallery ? "true" : "false"});
    eventSource = new EventSource("./internal/progress-stream?" + params.toString());

    eventSource.onmessage = function(event) {
        consecutiveErrors = 0;

        let res;
        try {
            res = JSON.parse(event.data);
        } catch (error) {
            console.error(error);
            return;
        }

        handleUpdate(res);
    };

    eventSource.onerror = function() {
        consecutiveErrors++;
        // the browser auto-reconnects on transient drops; only give up (and tear down
        // the progressbar) after a few in a row, so a single hiccup isn't fatal like it
        // was with the old fail-fast XHR polling
        if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
            removeProgressBar();
        }
    };
}
