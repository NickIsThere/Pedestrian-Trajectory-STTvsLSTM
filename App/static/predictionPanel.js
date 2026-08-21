/*
Main writer: Keez Cuijpers
Reviewer: Noah Nuelandt
Contributors: Claire Bams,Néo Deward
*/

import { futurePredictionPanel } from "./dom.js";
import { currentSelectedPerson, currentSelectedTrackId, getCurrentSelectedModelLabel , currentSelM, groundTruthState, predictionState } from "./state.js";


const PERSON_ORDER = ["A", "B", "C", "D"];
const MAX_DRAW_POINTS = 24;
const CENTER_POINT_RADIUS = 12;
const MIN_DRAW_PADDING_X = 110;
const MIN_DRAW_PADDING_Y = 88;
const MAX_DIRECTION_SCALE = 6;

let sourceWidth = 1000;
let sourceHeight = 562;
let currentFrameId = null;
let currentHorizon = 24;

let svgEl = null;
let messageEl = null;

function getModelName() {
    return getCurrentSelectedModelLabel();
}

function ensurePanelMarkup() {
    if (!futurePredictionPanel) return;
    if (svgEl && messageEl) return;

    futurePredictionPanel.innerHTML = "";

    svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svgEl.setAttribute("id", "prediction-svg");
    svgEl.setAttribute("viewBox", `0 0 ${sourceWidth} ${sourceHeight}`);
    svgEl.setAttribute("preserveAspectRatio", "xMidYMid meet");

    messageEl = document.createElement("div");
    messageEl.id = "prediction-message";

    futurePredictionPanel.appendChild(svgEl);
    futurePredictionPanel.appendChild(messageEl);
}

function setMessage(text) {
    if (messageEl) {
        messageEl.textContent = text || "";
    }
}

function clearSvg() {
    if (svgEl) {
        svgEl.innerHTML = "";
    }
}

function setFullViewBox() {
    if (!svgEl) return;
    svgEl.setAttribute("viewBox", `0 0 ${sourceWidth} ${sourceHeight}`);
}

function normalizePoint(point) {
    const x = Number(point?.x);
    const y = Number(point?.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
        return null;
    }
    return { x, y };
}

function mapSelectedPersonToTrackId() {
    if (!currentSelectedPerson || !PERSON_ORDER.includes(currentSelectedPerson)) {
        return null;
    }

    const trackIds = groundTruthState.allTracks.map((track) => track.id).sort((a, b) => a - b);
    const index = PERSON_ORDER.indexOf(currentSelectedPerson);
    return trackIds[index] ?? null;
}

function getActiveTrackId() {
    if (Number.isFinite(currentSelectedTrackId)) {
        return currentSelectedTrackId;
    }
    return mapSelectedPersonToTrackId();
}

function getTrackById(trackId, tracksArray) {
    return tracksArray.find((track) => Number(track.id) === Number(trackId)) || null;
}

function getTrackPointAtOrBeforeFrame(track, frameId) {
    if (!track?.points?.length || !Number.isFinite(Number(frameId))) {
        return null;
    }

    const eligible = track.points.filter((point) => Number(point.frame_id) <= Number(frameId));
    if (!eligible.length) {
        return null;
    }

    return normalizePoint(eligible[eligible.length - 1]);
}

function getTrackFuturePoints(track, frameId, horizon) {
    if (!track?.points?.length || !Number.isFinite(Number(frameId)) || !Number.isFinite(Number(horizon))) {
        return [];
    }

    const lastFrame = Number(frameId) + Number(horizon);
    return track.points
        .filter((point) => Number(point.frame_id) > Number(frameId) && Number(point.frame_id) <= lastFrame)
        .slice(0, MAX_DRAW_POINTS)
        .map((point) => normalizePoint(point))
        .filter(Boolean);
}

function appendArrowMarkers() {
    if (!svgEl) return;

    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");

    const actualMarker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
    actualMarker.setAttribute("id", "prediction-arrow-actual");
    actualMarker.setAttribute("viewBox", "0 0 10 10");
    actualMarker.setAttribute("refX", "9");
    actualMarker.setAttribute("refY", "5");
    actualMarker.setAttribute("markerWidth", "8");
    actualMarker.setAttribute("markerHeight", "8");
    actualMarker.setAttribute("orient", "auto-start-reverse");
    const actualPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
    actualPath.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    actualPath.setAttribute("fill", "#ff8b7c");
    actualMarker.appendChild(actualPath);

    const modelMarker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
    modelMarker.setAttribute("id", "prediction-arrow-model");
    modelMarker.setAttribute("viewBox", "0 0 10 10");
    modelMarker.setAttribute("refX", "9");
    modelMarker.setAttribute("refY", "5");
    modelMarker.setAttribute("markerWidth", "8");
    modelMarker.setAttribute("markerHeight", "8");
    modelMarker.setAttribute("orient", "auto-start-reverse");
    const modelPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
    modelPath.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    modelPath.setAttribute("fill", "#63c7eb");
    modelMarker.appendChild(modelPath);

    defs.appendChild(actualMarker);
    defs.appendChild(modelMarker);
    svgEl.appendChild(defs);
}

function drawPolyline(points, { stroke, width, dash = null, opacity = 1, markerEnd = null }) {
    if (!svgEl || points.length < 2) return;

    const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    polyline.setAttribute("points", points.map((point) => `${point.x},${point.y}`).join(" "));
    polyline.setAttribute("fill", "none");
    polyline.setAttribute("stroke", stroke);
    polyline.setAttribute("stroke-width", String(width));
    polyline.setAttribute("stroke-linecap", "round");
    polyline.setAttribute("stroke-linejoin", "round");
    polyline.setAttribute("opacity", String(opacity));
    if (dash) {
        polyline.setAttribute("stroke-dasharray", dash);
    }
    if (markerEnd) {
        polyline.setAttribute("marker-end", markerEnd);
    }

    svgEl.appendChild(polyline);
}

function drawCircle(point, { fill, stroke, strokeWidth, radius }) {
    if (!svgEl || !point) return;

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", String(point.x));
    circle.setAttribute("cy", String(point.y));
    circle.setAttribute("r", String(radius));
    circle.setAttribute("fill", fill);
    circle.setAttribute("stroke", stroke);
    circle.setAttribute("stroke-width", String(strokeWidth));
    svgEl.appendChild(circle);
}

function drawDirectionLabel(point, center, label, color, siblingPoint = null, variant = "primary") {
    if (!svgEl || !point) return;

    const dx = point.x - center.x;
    const dy = point.y - center.y;
    let offsetX = Math.abs(dx) < 14 ? 0 : (dx >= 0 ? 18 : -18);
    let offsetY = dy <= 0 ? -16 : 22;
    let textAnchor = Math.abs(dx) < 14 ? "middle" : (dx >= 0 ? "start" : "end");

    if (siblingPoint) {
        const distance = Math.hypot(point.x - siblingPoint.x, point.y - siblingPoint.y);
        if (distance < 56) {
            offsetY += variant === "actual" ? -18 : 18;
            if (Math.abs(dx) < 20) {
                offsetX += variant === "actual" ? -10 : 10;
                textAnchor = variant === "actual" ? "end" : "start";
            }
        }
    }

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", String(point.x + offsetX));
    text.setAttribute("y", String(point.y + offsetY));
    text.setAttribute("fill", color);
    text.setAttribute("font-size", "18");
    text.setAttribute("font-weight", "700");
    text.setAttribute("text-anchor", textAnchor);
    text.setAttribute("stroke", "#09111a");
    text.setAttribute("stroke-width", "5");
    text.setAttribute("paint-order", "stroke");
    text.textContent = label;
    svgEl.appendChild(text);
}

function buildRelativePath(points, origin) {
    if (!origin || !points.length) {
        return [];
    }

    return points.map((point) => ({
        dx: point.x - origin.x,
        dy: point.y - origin.y,
    }));
}

function computeDirectionScale(relativePaths) {
    const vectors = relativePaths.flat();
    if (!vectors.length) {
        return 1;
    }

    const maxAbsX = Math.max(...vectors.map((vector) => Math.abs(vector.dx)), 1);
    const maxAbsY = Math.max(...vectors.map((vector) => Math.abs(vector.dy)), 1);
    const availableHalfWidth = Math.max(60, sourceWidth / 2 - MIN_DRAW_PADDING_X);
    const availableHalfHeight = Math.max(60, sourceHeight / 2 - MIN_DRAW_PADDING_Y);
    const scaleX = availableHalfWidth / maxAbsX;
    const scaleY = availableHalfHeight / maxAbsY;
    const scale = Math.min(scaleX, scaleY);

    if (!Number.isFinite(scale) || scale <= 0) {
        return 1;
    }

    return Math.min(scale, MAX_DIRECTION_SCALE);
}

function projectRelativePath(relativePath, center, scale) {
    return relativePath.map((vector) => ({
        x: center.x + vector.dx * scale,
        y: center.y + vector.dy * scale,
    }));
}

function renderPredictionTrack() {
    ensurePanelMarkup();
    clearSvg();
    setFullViewBox();
    if (!groundTruthState.allTracks.length) {
        setMessage("Direction view unavailable for this video.");
        return;
    }

    const selectedTrackId = getActiveTrackId();
    if (selectedTrackId === null) {
        setMessage("Select a track to compare actual vs predicted direction.");
        return;
    }

    const selectedTrack = getTrackById(selectedTrackId, groundTruthState.allTracks);
    if (!selectedTrack) {
        setMessage(`No trajectory data for track ${selectedTrackId}.`);
        return;
    }

    const predictionTrack = getTrackById(selectedTrackId, predictionState.tracks);
    if (!predictionTrack) {
        setMessage(`No direction data for track ${selectedTrackId} at this frame.`);
        return;
    }

    const predPoint = predictionTrack.points.find(p => p.frame_id === currentFrameId);
    if (!predPoint) {
        setMessage(`No direction data for track ${selectedTrackId} at this frame.`);
        return;
    }

    const anchorPoint = normalizePoint({ x: predPoint.x, y: predPoint.y });
    const actualCurrent = getTrackPointAtOrBeforeFrame(selectedTrack, currentFrameId);
    const referenceCurrent = actualCurrent || anchorPoint;

    if (!referenceCurrent) {
        setMessage(`Track ${selectedTrackId}: current position unavailable.`);
        return;
    }

    const forecastAbsolute = getTrackFuturePoints(predictionTrack, currentFrameId, currentHorizon);
    const actualFutureAbsolute = getTrackFuturePoints(
        selectedTrack,
        currentFrameId,
        Math.max(currentHorizon, forecastAbsolute.length),
    );

    const actualRelativePath = buildRelativePath(actualFutureAbsolute, referenceCurrent);
    const modelRelativePath = buildRelativePath(forecastAbsolute, anchorPoint || referenceCurrent);
    const centerPoint = {
        x: sourceWidth / 2,
        y: sourceHeight / 2,
    };
    const scale = computeDirectionScale([actualRelativePath, modelRelativePath]);
    const actualLocalPath = projectRelativePath(actualRelativePath, centerPoint, scale);
    const modelLocalPath = projectRelativePath(modelRelativePath, centerPoint, scale);

    appendArrowMarkers();

    if (actualLocalPath.length) {
        drawPolyline([centerPoint, ...actualLocalPath], {
            stroke: "#ff8b7c",
            width: 4.2,
            opacity: 0.96,
            markerEnd: "url(#prediction-arrow-actual)",
        });
    }

    if (modelLocalPath.length) {
        drawPolyline([centerPoint, ...modelLocalPath], {
            stroke: "#63c7eb",
            width: 4,
            dash: "12 8",
            opacity: 0.98,
            markerEnd: "url(#prediction-arrow-model)",
        });
    }

    drawCircle(centerPoint, {
        fill: "#f8fbff",
        stroke: "#ff8b7c",
        strokeWidth: 4,
        radius: CENTER_POINT_RADIUS,
    });

    if (modelLocalPath.length) {
        drawDirectionLabel(
            modelLocalPath[modelLocalPath.length - 1],
            centerPoint,
            getModelName(),
            "#9ae7ff",
            actualLocalPath[actualLocalPath.length - 1] || null,
            "model",
        );
    }

    if (actualLocalPath.length || modelLocalPath.length) {
        if (!actualLocalPath.length) {
            setMessage("Actual future path unavailable at this frame.");
            return;
        }
        if (!modelLocalPath.length) {
            setMessage("Model forecast unavailable at this frame.");
            return;
        }
        setMessage("");
        return;
    }

    setMessage("No future direction available for this track at this frame.");
}

export function setupPredictionPanel() {
    ensurePanelMarkup();
    setMessage("Loading direction view...");

    window.addEventListener("tracksLoaded", () => {
        renderPredictionTrack();
    });

    window.addEventListener("predictionTracksLoaded", () => {
        renderPredictionTrack();
    });

    window.addEventListener("selectedPersonChanged", () => {
        renderPredictionTrack();
    });

    window.addEventListener("selectedTrackChanged", () => {
        renderPredictionTrack();
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
        setFullViewBox();
        renderPredictionTrack();
    });

    window.addEventListener("modelChanged", () => {
        clearSvg();
        setMessage(`Loading ${getModelName()} direction view...`);
        renderPredictionTrack();
    });

    renderPredictionTrack();
}
