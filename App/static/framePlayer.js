/*
Main writer: Claire Bams
Reviewer: Noah Nuelandt
Contributors: Keez Cuijpers
*/

import { frameIndicator, framePlayToggle, videoErrorBanner, videoFeed } from "./dom.js";
import { setAllTracks, setPredictionTracks, currentSelM, updateGroundTruth } from "./state.js";

let currentVideo = "MOT20-01";
let frameIds = [];
let fps = 25;
let width = 1000;
let height = 562;
let currentIndex = 0;
let timer = null;

async function loadTracksAndPredictions() {
    try {
        const gtRes = await fetch(`/api/sequences/${currentVideo}/channels/gt/tracks`);
        if (gtRes.ok) {
            const data = await gtRes.json();
            setAllTracks(data.tracks || []);
        } else {
            console.error("Failed to load ground truth tracks:", gtRes.status);
            setAllTracks([]);
        }
    } catch (err) {
        console.error("Error loading ground truth tracks:", err);
        setAllTracks([]);
    }

    try {
        const model = currentSelM || "kalman";
        const predRes = await fetch(`/api/sequences/${currentVideo}/channels/${model}/tracks`);
        if (predRes.ok) {
            const data = await predRes.json();
            setPredictionTracks(data.tracks || []);
        } else {
            setPredictionTracks([]);
        }
    } catch (err) {
        console.error("Error loading prediction tracks:", err);
        setPredictionTracks([]);
    }
}

function updatePlayButtonLabel() {
    if (!framePlayToggle) return;
    if (timer) {
        framePlayToggle.textContent = "Pause";
        return;
    }
    if (frameIds.length > 0 && currentIndex >= frameIds.length - 1) {
        framePlayToggle.textContent = "Reset";
        return;
    }
    framePlayToggle.textContent = "Play";
}

function showError(message) {
    if (!videoErrorBanner) return;
    videoErrorBanner.textContent = message;
    videoErrorBanner.classList.remove("hidden");
}

function hideError() {
    if (!videoErrorBanner) return;
    videoErrorBanner.textContent = "";
    videoErrorBanner.classList.add("hidden");
}

function updateIndicator() {
    if (!frameIndicator || frameIds.length === 0) return;
    frameIndicator.textContent = `Frame: ${frameIds[currentIndex]} (${currentIndex + 1}/${frameIds.length})`;
}

function emitFrameChanged(frameId) {
    window.dispatchEvent(new CustomEvent("frameChanged", {
        detail: {
            frameId,
            sourceWidth: width,
            sourceHeight: height,
        },
    }));
}

function renderFrame(index) {
    if (!videoFeed || frameIds.length === 0) return;

    currentIndex = Math.max(0, Math.min(index, frameIds.length - 1));
    const frameId = frameIds[currentIndex];
    const frameUrl = `/api/sequences/${currentVideo}/frames/${frameId}/img?_=${Date.now()}`;

    videoFeed.src = frameUrl;
    updateIndicator();
    
    emitFrameChanged(frameId);
    
    // Update ground truth tracks for the current frame
    updateGroundTruth(frameId);
    
    // groundTruthUpdated event must be fired so visualization draws
    window.dispatchEvent(new CustomEvent("groundTruthUpdated"));
    
    updatePlayButtonLabel();
}

function stepForward() {
    if (frameIds.length === 0) return;
    const nextIndex = currentIndex + 1;
    if (nextIndex >= frameIds.length) {
        stopPlayback();
        return;
    }
    renderFrame(nextIndex);
}

function startPlayback() {
    if (timer || frameIds.length === 0) return;
    const intervalMs = Math.max(20, Math.floor(1000 / fps));
    timer = setInterval(stepForward, intervalMs);
    updatePlayButtonLabel();
}

function stopPlayback() {
    if (timer) {
        clearInterval(timer);
        timer = null;
    }
    updatePlayButtonLabel();
}

async function loadManifest() {
    const response = await fetch(`/api/sequences/${currentVideo}/manifest`);
    if (!response.ok) {
        throw new Error(`Failed to load frame manifest (${response.status})`);
    }

    const manifest = await response.json();
    
    // Instead of manifest.frame_ids which no longer exists, we generate 1 to frame_count
    const count = Number(manifest.frame_count || 0);
    frameIds = Array.from({ length: count }, (_, i) => i + 1);
    
    fps = Number(manifest.fps || 25);
    width = Number(manifest.width || 1000);
    height = Number(manifest.height || 562);

    if (frameIds.length === 0) {
        throw new Error("Frame manifest is empty for the active video.");
    }
    
    await loadTracksAndPredictions();

    currentIndex = 0;
}

export async function resetFramePlayer() {
    stopPlayback();
    try {
        await loadManifest();
        hideError();
        renderFrame(0);
    } catch (error) {
        showError(error.message);
        if (frameIndicator) frameIndicator.textContent = "Frame: unavailable";
    }
}

export function setupFramePlayer() {
    if (!videoFeed) return;

    videoFeed.addEventListener("error", () => {
        showError("Failed to load frame image from dataset path.");
    });

    videoFeed.addEventListener("load", () => {
        hideError();
    });

    if (framePlayToggle) {
        framePlayToggle.addEventListener("click", () => {
            if (timer) {
                stopPlayback();
            } else if (frameIds.length > 0 && currentIndex >= frameIds.length - 1) {
                renderFrame(0);
            } else {
                startPlayback();
            }
        });
    }

    window.addEventListener("activeVideoChanged", (event) => {
        if (event.detail && event.detail.video) {
            currentVideo = event.detail.video;
        }
        resetFramePlayer();
    });

    window.addEventListener("modelChanged", async () => {
        await loadTracksAndPredictions();
    });

    resetFramePlayer();
}
