/*
Main writer: Claire Bams
Reviewer: Noah Nuelandt
Contributors: Keez Cuijpers
*/

import { prsAde, prsFde } from "./dom.js";
import { currentSelectedTrackId, updateGenValue, currentSelM } from "./state.js";
import { updGenMetrics } from "./ui.js";

const UNAVAILABLE = "Unavailable";
const PLACEHOLDER = "—";

let inFlightVideo = null;
let queuedVideo = null;
let lastLoadedVideo = null;
let lastActiveVideo = null;
let targetRequestToken = 0;

function formatCount(value) {
    const normalized = Number(value);
    return Number.isFinite(normalized) ? String(Math.round(normalized)) : UNAVAILABLE;
}

function formatMetric(value) {
    const normalized = Number(value);
    return Number.isFinite(normalized) ? normalized.toFixed(3) : UNAVAILABLE;
}

function applyStatistics(data) {
    updateGenValue({
        fps: formatCount(data?.person_count),
        ade: formatMetric(data?.ade),
        fde: formatMetric(data?.fde),
    });
    updGenMetrics();
}

function applyUnavailableStatistics() {
    updateGenValue({
        fps: UNAVAILABLE,
        ade: UNAVAILABLE,
        fde: UNAVAILABLE,
    });
    updGenMetrics();
}

async function fetchStatistics(video) {
    const model = currentSelM || "kalman";
    const response = await fetch(`/api/sequences/${video}/channels/${model}/statistics`);
    if (!response.ok) {
        throw new Error(`Failed to load statistics (${response.status})`);
    }
    return response.json();
}

function applySelectedTargetStatistics(data) {
    if (prsAde) prsAde.textContent = formatMetric(data?.ade);
    if (prsFde) prsFde.textContent = formatMetric(data?.fde);
}

function setSelectedTargetStatistics(value) {
    if (prsAde) prsAde.textContent = value;
    if (prsFde) prsFde.textContent = value;
}

async function refreshStatistics(video) {
    if (!video) {
        return;
    }

    if (video === inFlightVideo || video === lastLoadedVideo) {
        return;
    }

    if (inFlightVideo) {
        queuedVideo = video;
        return;
    }

    inFlightVideo = video;

    try {
        const payload = await fetchStatistics(video);
        applyStatistics(payload);
        lastLoadedVideo = video;
    } catch (error) {
        console.error("Failed to refresh statistics:", error);
        applyUnavailableStatistics();
    } finally {
        inFlightVideo = null;

        if (queuedVideo && queuedVideo !== lastLoadedVideo) {
            const nextVideo = queuedVideo;
            queuedVideo = null;
            void refreshStatistics(nextVideo);
            return;
        }

        queuedVideo = null;
    }
}

async function refreshSelectedTargetStatistics(trackId) {
    const normalizedTrackId = Number(trackId);
    const requestToken = ++targetRequestToken;

    if (!Number.isFinite(normalizedTrackId)) {
        setSelectedTargetStatistics(PLACEHOLDER);
        return;
    }

    // Backend track-level stats endpoint was removed.
    setSelectedTargetStatistics(UNAVAILABLE);
}

export function setupStatisticsPanel() {
    window.addEventListener("activeVideoChanged", (event) => {
        const video = event?.detail?.video ?? null;
        lastActiveVideo = video;
        void refreshStatistics(video);
        void refreshSelectedTargetStatistics(currentSelectedTrackId);
    });

    window.addEventListener("selectedTrackChanged", (event) => {
        const trackId = event?.detail?.trackId ?? null;
        if (!lastActiveVideo) {
            setSelectedTargetStatistics(Number.isFinite(Number(trackId)) ? UNAVAILABLE : PLACEHOLDER);
            return;
        }
        void refreshSelectedTargetStatistics(trackId);
    });

    window.addEventListener("modelChanged", () => {
        if (lastActiveVideo) {
            lastLoadedVideo = null; // force reload
            void refreshStatistics(lastActiveVideo);
        }
    });
}
