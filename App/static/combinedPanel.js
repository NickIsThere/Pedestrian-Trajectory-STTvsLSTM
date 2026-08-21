/*
Main writer: Claire Bams
Reviewer: Noah Nuelandt
Contributors: Keez Cuijpers,Néo Deward
*/

import { combinedPanel, combinedPanelTitle } from "./dom.js";
import { currentSelectedPerson, currentSelectedTrackId, currentSelM, getCurrentSelectedModelLabel, groundTruthState, predictionState } from "./state.js";

const PERSON_ORDER = ["A", "B", "C", "D"];

let sourceWidth = 1000;
let sourceHeight = 562;
let currentFrameId = null;
let currentHorizon = 30;
let compareMode = false;
const predictionHistoryByTrack = new Map();

let svgEl = null;
let messageEl = null;
let controlsEl = null;
let modeToggleEl = null;
let clearHistoryEl = null;

function ensurePanelMarkup() {
    if (!combinedPanel) return;
    if (svgEl && messageEl && controlsEl) return;

    combinedPanel.innerHTML = "";

    controlsEl = document.createElement("div");
    controlsEl.id = "combined-controls";

    modeToggleEl = document.createElement("button");
    modeToggleEl.id = "combined-mode-toggle";
    modeToggleEl.type = "button";
    modeToggleEl.textContent = "Mode: Live";

    clearHistoryEl = document.createElement("button");
    clearHistoryEl.id = "combined-clear-history";
    clearHistoryEl.type = "button";
    clearHistoryEl.textContent = "Clear history";

    controlsEl.appendChild(modeToggleEl);
    controlsEl.appendChild(clearHistoryEl);

    svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svgEl.setAttribute("id", "combined-svg");
    svgEl.setAttribute("viewBox", `0 0 ${sourceWidth} ${sourceHeight}`);
    svgEl.setAttribute("preserveAspectRatio", "xMidYMid meet");

    messageEl = document.createElement("div");
    messageEl.id = "combined-message";

    combinedPanel.appendChild(controlsEl);
    combinedPanel.appendChild(svgEl);
    combinedPanel.appendChild(messageEl);

    modeToggleEl.addEventListener("click", () => {
        compareMode = !compareMode;
        modeToggleEl.textContent = compareMode ? "Mode: Compare" : "Mode: Live";
        renderCombinedTrack();
    });

    clearHistoryEl.addEventListener("click", () => {
        predictionHistoryByTrack.clear();
        renderCombinedTrack();
    });
}


function getModelName() {
    return getCurrentSelectedModelLabel();
}

function setMessage(text) {
    if (messageEl) messageEl.textContent = text;
}

function clearSvg() {
    if (svgEl) svgEl.innerHTML = "";
}

function mapSelectedPersonToTrackId() {
    if (!currentSelectedPerson || !PERSON_ORDER.includes(currentSelectedPerson)) {
        return null;
    }

    const trackIds = groundTruthState.allTracks.map((t) => t.id).sort((a, b) => a - b);
    const index = PERSON_ORDER.indexOf(currentSelectedPerson);
    return trackIds[index] ?? null;
}

function getActiveTrackId() {
    if (Number.isFinite(currentSelectedTrackId)) {
        return currentSelectedTrackId;
    }
    return mapSelectedPersonToTrackId();
}

function getTrackHistory(trackId) {
    if (!Number.isFinite(Number(trackId))) {
        return [];
    }
    return predictionHistoryByTrack.get(Number(trackId)) || [];
}

function addPredictionSnapshot(trackId, frameId, anchor, forecast) {
    const normalizedTrackId = Number(trackId);
    if (!Number.isFinite(normalizedTrackId) || !Array.isArray(forecast) || !forecast.length) {
        return;
    }

    const existing = predictionHistoryByTrack.get(normalizedTrackId) || [];
    const withoutSameFrame = existing.filter((item) => Number(item.frameId) !== Number(frameId));
    withoutSameFrame.push({
        frameId: Number(frameId),
        anchor,
        forecast,
    });
    withoutSameFrame.sort((a, b) => a.frameId - b.frameId);
    predictionHistoryByTrack.set(normalizedTrackId, withoutSameFrame);
}

function drawPolyline(points, { stroke, width, dash = null, opacity = 1 }) {
    if (!svgEl || points.length < 2) return;

    const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    polyline.setAttribute("points", points.map((p) => `${p.x},${p.y}`).join(" "));
    polyline.setAttribute("fill", "none");
    polyline.setAttribute("stroke", stroke);
    polyline.setAttribute("stroke-width", String(width));
    polyline.setAttribute("stroke-linecap", "round");
    polyline.setAttribute("stroke-linejoin", "round");
    polyline.setAttribute("opacity", String(opacity));
    if (dash) {
        polyline.setAttribute("stroke-dasharray", dash);
    }

    svgEl.appendChild(polyline);
}

function drawCurrentPoint(point, stroke = "#ff7a7a") {
    if (!svgEl || !point) return;

    const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    marker.setAttribute("cx", String(point.x));
    marker.setAttribute("cy", String(point.y));
    marker.setAttribute("r", "7");
    marker.setAttribute("fill", "#ffffff");
    marker.setAttribute("stroke", stroke);
    marker.setAttribute("stroke-width", "3");
    svgEl.appendChild(marker);
}

function drawPredictedPoints(points) {
    if (!svgEl || !points.length) return;

    points.forEach((point, index) => {
        const t = index / Math.max(1, points.length - 1);
        const radius = 5 - 2 * t;
        const alpha = 0.95 - 0.55 * t;

        const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        dot.setAttribute("cx", String(point.x));
        dot.setAttribute("cy", String(point.y));
        dot.setAttribute("r", String(Math.max(2.2, radius)));
        dot.setAttribute("fill", "#45b7d1");
        dot.setAttribute("opacity", String(Math.max(0.25, alpha)));
        svgEl.appendChild(dot);
    });
}

function drawAnchorPoint(point) {
    if (!svgEl || !point) return;

    const anchor = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    anchor.setAttribute("cx", String(point.x));
    anchor.setAttribute("cy", String(point.y));
    anchor.setAttribute("r", "6");
    anchor.setAttribute("fill", "#3ddc97");
    anchor.setAttribute("stroke", "#ffffff");
    anchor.setAttribute("stroke-width", "2");
    anchor.setAttribute("opacity", "0.95");
    svgEl.appendChild(anchor);
}

function renderCombinedTrack() {
    ensurePanelMarkup();
    clearSvg();

    if (!groundTruthState.allTracks.length) {
        setMessage(`${getModelName()} forecast context unavailable for this video.`);
        return;
    }

    const selectedTrackId = getActiveTrackId();
    if (selectedTrackId === null) {
        setMessage("Select a person circle in video (or A/B/C/D) to view forecast context.");
        return;
    }

    const selectedTrack = groundTruthState.allTracks.find((t) => Number(t.id) === Number(selectedTrackId));
    if (!selectedTrack || !selectedTrack.points?.length) {
        setMessage(`No trajectory data for track ${selectedTrackId}.`);
        return;
    }

    const tracePoints = compareMode
        ? selectedTrack.points
        : (currentFrameId == null
            ? selectedTrack.points
            : selectedTrack.points.filter((p) => p.frame_id <= currentFrameId));

    if (!tracePoints.length) {
        setMessage(`No points yet for track ${selectedTrackId} at current frame.`);
        return;
    }

    const currentTracePoint = tracePoints[tracePoints.length - 1];

    drawPolyline(tracePoints, {
        stroke: "#ff7a7a",
        width: 4,
        opacity: 0.9,
    });
    drawCurrentPoint(currentTracePoint, "#ff7a7a");

    const predictionTrack = predictionState.tracks.find((t) => Number(t.id) === Number(selectedTrackId));
    if (predictionTrack) {
        const predPoint = predictionTrack.points.find(p => p.frame_id === currentFrameId);
        if (predPoint) {
            const forecast = predictionTrack.points
                .filter(p => p.frame_id > currentFrameId && p.frame_id <= currentFrameId + currentHorizon)
                .map(p => ({ x: Number(p.x), y: Number(p.y) }));
            if (forecast.length) {
                addPredictionSnapshot(
                    selectedTrackId,
                    currentFrameId,
                    { x: Number(predPoint.x), y: Number(predPoint.y) },
                    forecast
                );
            }
        }
    }

    const history = getTrackHistory(selectedTrackId);

    if (compareMode) {
        if (!history.length) {
            setMessage(`Track ${selectedTrackId}: no stored ${getModelName()} snapshots yet. Play the video to build forecast history.`);
            return;
        }

        history.forEach((snapshot, index) => {
            const startPoint = snapshot.anchor;
            if (!startPoint || !snapshot.forecast?.length) {
                return;
            }

            const path = [
                { x: Number(startPoint.x), y: Number(startPoint.y) },
                ...snapshot.forecast,
            ];

            const opacity = 0.1 + 0.35 * ((index + 1) / history.length);
            drawPolyline(path, {
                stroke: "#45b7d1",
                width: 1.6,
                dash: "8 6",
                opacity,
            });
        });

        const highlighted = currentFrameId == null
            ? history[history.length - 1]
            : (history.find((h) => Number(h.frameId) === Number(currentFrameId)) || history[history.length - 1]);

        const highlightedStart = highlighted.anchor;

        if (highlightedStart && highlighted.forecast?.length) {
            const highlightedPath = [
                { x: Number(highlightedStart.x), y: Number(highlightedStart.y) },
                ...highlighted.forecast,
            ];
            drawAnchorPoint({ x: Number(highlightedStart.x), y: Number(highlightedStart.y) });
            drawPolyline(highlightedPath, {
                stroke: "#7be7ff",
                width: 3.4,
                dash: "10 7",
                opacity: 0.98,
            });
            drawPredictedPoints(highlighted.forecast);
        }

        setMessage(`Track ${selectedTrackId}: forecast context mode with ${history.length} stored ${getModelName()} snapshots.`);
        return;
    }

    const currentSnapshot = history.find(h => Number(h.frameId) === Number(currentFrameId));
    if (currentSnapshot && currentSnapshot.forecast?.length) {
        const forecastAnchor = currentSnapshot.anchor;
        drawAnchorPoint(forecastAnchor);

        const forecastPath = [
            forecastAnchor,
            ...currentSnapshot.forecast,
        ];

        drawPolyline(forecastPath, {
            stroke: "#45b7d1",
            width: 3.4,
            dash: "10 7",
            opacity: 0.9,
        });
        drawPredictedPoints(currentSnapshot.forecast);
        setMessage(`Track ${selectedTrackId}: trace context (${tracePoints.length}) + ${getModelName()} forecast (${currentSnapshot.forecast.length}).`);
        return;
    }

    setMessage(`Track ${selectedTrackId}: trace shown. Prediction not available for this frame.`);
}

function updateCombinedPanelTitle() {
    if (combinedPanelTitle) {
        combinedPanelTitle.textContent = `${getModelName()} Forecast Context`;
    }
}

export function setupCombinedPanel() {
    ensurePanelMarkup();
    setMessage(`Loading ${getModelName()} forecast context...`);

    window.addEventListener("tracksLoaded", () => {
        renderCombinedTrack();
    });

    window.addEventListener("predictionTracksLoaded", () => {
        renderCombinedTrack();
    });

    window.addEventListener("modelChanged", () => {
        predictionHistoryByTrack.clear();
        clearSvg();
        setMessage(`Loading ${getModelName()} forecast context...`);
        renderCombinedTrack();
    });

    window.addEventListener("activeVideoChanged", () => {
        predictionHistoryByTrack.clear();
        renderCombinedTrack();
    });

    window.addEventListener("selectedPersonChanged", () => {
        renderCombinedTrack();
    });

    window.addEventListener("selectedTrackChanged", () => {
        renderCombinedTrack();
    });

    window.addEventListener("frameChanged", (event) => {
        const detail = event?.detail || {};
        if (Number.isFinite(detail.frameId)) {
            currentFrameId = detail.frameId;
        }
        if (Number.isFinite(detail.sourceWidth) && detail.sourceWidth > 0) {
            sourceWidth = detail.sourceWidth;
        }
        if (Number.isFinite(detail.sourceHeight) && detail.sourceHeight > 0) {
            sourceHeight = detail.sourceHeight;
        }
        if (svgEl) {
            svgEl.setAttribute("viewBox", `0 0 ${sourceWidth} ${sourceHeight}`);
        }
        renderCombinedTrack();
    });

    renderCombinedTrack();
}
