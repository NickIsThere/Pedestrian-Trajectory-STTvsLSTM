/*
Main writer: Claire Bams
Reviewer: Noah Nuelandt
Contributors: Keez Cuijpers
*/

import { groundTruthState } from "./state.js";
import { setTrackMode } from "./ui.js";

// Configuration
const SVG_VIEWBOX_WIDTH = 1000;
const SVG_VIEWBOX_HEIGHT = 562;

// DOM references
let videoElement = null;
let svgElement = null;
let groundTruthGroup = null;

let sourceWidth = 1000;
let sourceHeight = 562;


/**
 * Initialize visualization system
 */
export function initVisualization() {
    videoElement = document.getElementById("video-feed");
    svgElement = document.querySelector("svg.paths");
    
    if (!videoElement || !svgElement) {
        console.warn("Video or SVG element not found");
        return;
    }
    
    // Create a group for ground truth visualization
    groundTruthGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    groundTruthGroup.setAttribute("id", "ground-truth-layer");
    groundTruthGroup.setAttribute("class", "ground-truth");
    svgElement.appendChild(groundTruthGroup);
    
    window.addEventListener("frameChanged", handleFrameChanged);
    
    // Listen for ground truth updates
    window.addEventListener("groundTruthUpdated", handleGroundTruthUpdate);
    
    console.log("Visualization system initialized");
}


/**
 * Handle frame updates from the image-sequence player.
 */
function handleFrameChanged(event) {
    const detail = event?.detail || {};
    if (Number.isFinite(detail.sourceWidth) && detail.sourceWidth > 0) {
        sourceWidth = detail.sourceWidth;
    }
    if (Number.isFinite(detail.sourceHeight) && detail.sourceHeight > 0) {
        sourceHeight = detail.sourceHeight;
    }
}


/**
 * Handle ground truth data received from backend
 */
function handleGroundTruthUpdate() {
    drawGroundTruth();
}


/**
 * Draw ground truth points on SVG
 */
export function drawGroundTruth() {
    if (!groundTruthGroup) return;
    
    // Clear previous frame's drawing
    groundTruthGroup.innerHTML = "";
    
    const tracks = groundTruthState.tracks || [];
    
    if (tracks.length === 0) {
        return;
    }
    
    // Split tracks into different visual groups
    const tracksById = {};
    tracks.forEach(track => {
        if (!tracksById[track.track_id]) {
            tracksById[track.track_id] = [];
        }
        tracksById[track.track_id].push(track);
    });
    
    // Draw each track with a unique color
    Object.entries(tracksById).forEach(([trackId, points]) => {
        const color = getColorForTrack(parseInt(trackId));
        
        // Draw all points for this track in current frame
        points.forEach(point => {
            drawPoint(point.x, point.y, color, trackId);
        });
    });
}


/**
 * Draw a single point on the SVG
 */
function drawPoint(x, y, color, trackId) {
    // Scale coordinates from video space to viewBox space
    const scaledX = (x / sourceWidth) * SVG_VIEWBOX_WIDTH;
    const scaledY = (y / sourceHeight) * SVG_VIEWBOX_HEIGHT;
    
    // Create circle element
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", scaledX);
    circle.setAttribute("cy", scaledY);
    circle.setAttribute("r", "12");
    circle.setAttribute("fill", color);
    circle.setAttribute("opacity", "0.7");
    circle.setAttribute("class", `ground-truth-point track-${trackId}`);
    circle.style.cursor = "pointer";
    circle.setAttribute("pointer-events", "auto");
    circle.setAttribute("data-track-id", trackId);
    circle.setAttribute("data-x", x.toFixed(1));
    circle.setAttribute("data-y", y.toFixed(1));
    circle.addEventListener("click", () => {
        setTrackMode(Number(trackId));
    });
    
    // Create text label with track ID
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", scaledX);
    text.setAttribute("y", scaledY + 4);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("font-size", "10");
    text.setAttribute("fill", "white");
    text.setAttribute("pointer-events", "none");
    text.textContent = trackId;
    
    groundTruthGroup.appendChild(circle);
    groundTruthGroup.appendChild(text);
}


/**
 * Get consistent color for each track ID
 */
function getColorForTrack(trackId) {
    const colors = [
        "#FF6B6B", // Red
        "#4ECDC4", // Teal
        "#45B7D1", // Blue
        "#FFA07A", // Light salmon
        "#98D8C8", // Mint
        "#F7DC6F", // Yellow
        "#BB8FCE", // Purple
        "#85C1E2", // Light blue
        "#F8B88B", // Peach
        "#81C995", // Green
    ];
    return colors[trackId % colors.length];
}


/**
 * Clear all ground truth visualization
 */
export function clearVisualization() {
    if (groundTruthGroup) {
        groundTruthGroup.innerHTML = "";
    }
}


/**
 * Show/hide ground truth layer
 */
export function toggleGroundTruthVisibility(visible) {
    if (groundTruthGroup) {
        groundTruthGroup.style.display = visible ? "block" : "none";
    }
}
