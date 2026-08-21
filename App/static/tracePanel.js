import { tracePanel } from "./dom.js";
import { currentSelectedPerson, currentSelectedTrackId, groundTruthState } from "./state.js";

const PERSON_ORDER = ["A", "B", "C", "D"];

let sourceWidth = 1000;
let sourceHeight = 562;
let currentFrameId = null;

let svgEl = null;
let messageEl = null;

function ensurePanelMarkup() {
    if (!tracePanel) return;
    if (svgEl && messageEl) return;

    tracePanel.innerHTML = "";

    svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svgEl.setAttribute("id", "trace-svg");
    svgEl.setAttribute("viewBox", `0 0 ${sourceWidth} ${sourceHeight}`);
    svgEl.setAttribute("preserveAspectRatio", "xMidYMid meet");

    messageEl = document.createElement("div");
    messageEl.id = "trace-message";

    tracePanel.appendChild(svgEl);
    tracePanel.appendChild(messageEl);
}

function setMessage(message) {
    if (!messageEl) return;
    messageEl.textContent = message;
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

function drawPath(points) {
    if (!svgEl || points.length < 2) return;

    const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    polyline.setAttribute("points", points.map((p) => `${p.x},${p.y}`).join(" "));
    polyline.setAttribute("fill", "none");
    polyline.setAttribute("stroke", "#ff7a7a");
    polyline.setAttribute("stroke-width", "4");
    polyline.setAttribute("stroke-linecap", "round");
    polyline.setAttribute("stroke-linejoin", "round");
    polyline.setAttribute("opacity", "0.9");

    svgEl.appendChild(polyline);
}

function drawCurrentPoint(point) {
    if (!svgEl || !point) return;

    const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    marker.setAttribute("cx", String(point.x));
    marker.setAttribute("cy", String(point.y));
    marker.setAttribute("r", "8");
    marker.setAttribute("fill", "#ffffff");
    marker.setAttribute("stroke", "#ff7a7a");
    marker.setAttribute("stroke-width", "3");
    svgEl.appendChild(marker);
}

function renderTrace() {
    ensurePanelMarkup();
    clearSvg();

    if (!groundTruthState.allTracks.length) {
        setMessage("Trace unavailable for this video.");
        return;
    }

    const selectedTrackId = getActiveTrackId();
    if (selectedTrackId === null) {
        setMessage("Select a person circle in video (or A/B/C/D demo buttons) to view trace.");
        return;
    }

    const selectedTrack = groundTruthState.allTracks.find((t) => t.id === selectedTrackId);
    if (!selectedTrack || !selectedTrack.points?.length) {
        setMessage(`No trajectory data for selected track ${selectedTrackId}.`);
        return;
    }

    const pointsUntilFrame = currentFrameId == null
        ? selectedTrack.points
        : selectedTrack.points.filter((p) => p.frame_id <= currentFrameId);

    if (!pointsUntilFrame.length) {
        setMessage(`No points yet for track ${selectedTrackId} at current frame.`);
        return;
    }

    drawPath(pointsUntilFrame);
    drawCurrentPoint(pointsUntilFrame[pointsUntilFrame.length - 1]);
    if (Number.isFinite(currentSelectedTrackId)) {
        setMessage(`Tracking selected track ${selectedTrackId}.`);
    } else {
        setMessage(`Person ${currentSelectedPerson} mapped to track ${selectedTrackId}.`);
    }
}

export function setupTracePanel() {
    ensurePanelMarkup();
    setMessage("Loading traces...");

    window.addEventListener("tracksLoaded", () => {
        renderTrace();
    });

    window.addEventListener("selectedPersonChanged", () => {
        renderTrace();
    });

    window.addEventListener("selectedTrackChanged", () => {
        renderTrace();
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
        renderTrace();
    });

    renderTrace();
}
