/*
Main writer: Claire Bams
Reviewer: Noah Nuelandt
Contributors: Keez Cuijpers
*/

import { genButton, personButtons, videoSelect, modelSelect } from "./dom.js";
import { setGeneralMode, setPersonMode, setVideo } from "./ui.js";
import { setupFramePlayer } from "./framePlayer.js";
import { initVisualization } from "./visualization.js";
import { setupCombinedPanel } from "./combinedPanel.js";
import { setupPredictionPanel } from "./predictionPanel.js";
import { setupStatisticsPanel } from "./statisticsPanel.js";
import { setCurrentSelM } from "./state.js";

/**
 * Fetches available models from the backend and populates the dropdown.
 */
async function loadDynamicModels() {
    if (!modelSelect) return;

    try {
        const res = await fetch("/api/models");
        const data = await res.json();

        // Clear existing static options
        modelSelect.innerHTML = "";

        // Fill with discovered models from the /checkpoints folder
        data.models.forEach(m => {
            const opt = document.createElement("option");
            opt.value = m.id;
            opt.textContent = m.name;
            modelSelect.appendChild(opt);
        });

        // Set the initial model state to the first item in the list
        setCurrentSelM(modelSelect.value);
    } catch (e) {
        console.error("Failed to load models from backend:", e);
    }
}

/**
 * Main App Initialization
 */
loadDynamicModels().then(() => {

    // Listen for model selection changes
    if (modelSelect) {
        modelSelect.addEventListener("change", () => {
            const model = modelSelect.value;
            const modelLabel = modelSelect.options[modelSelect.selectedIndex].text;

            setCurrentSelM(model);

            // Update UI elements with the new model name
            const combinedTitle = document.getElementById("combined-panel-title");
            if (combinedTitle) combinedTitle.textContent = `${modelLabel} Forecast Context`;

            const modelSetupEl = document.getElementById("model-setup");
            if (modelSetupEl) modelSetupEl.textContent = modelLabel;

            // Trigger global refresh for prediction layers
            window.dispatchEvent(new CustomEvent("modelChanged", { detail: { model } }));
        });
    }

    // Top Bar Control Setup
    if (genButton) {
        genButton.addEventListener("click", () => setGeneralMode());
    }

    personButtons.forEach((button) => {
        button.addEventListener("click", () => setPersonMode(button.dataset.person));
    });

    // Video Selection Setup
    if (videoSelect) {
        videoSelect.addEventListener("change", () => {
            setGeneralMode();
            setVideo(videoSelect.value, { emitEvent: true });
        });
    }

    initVisualization();
    setupFramePlayer();
    setupCombinedPanel();
    setupPredictionPanel();
    setupStatisticsPanel();

    setGeneralMode();
    if (videoSelect) {
        setVideo(videoSelect.value, { emitEvent: true });
    }
});