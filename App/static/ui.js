/*
Main writer: Keez Cuijpers
Reviewer: Noah Nuelandt
Contributors: Claire Bams
*/

import {genButton, personButtons, generalPanel, personPanel, videoSelectedTarget, videoSubtitle, selectedTarget, prsId, prsStatus,
    prsHistory, prsHorizon, modelSetup, prsAde, prsFde, generalFps, generalAde, generalFde, personFps, videoSelect, videoLabel} from "./dom.js";

import {generalMetrics, getPersonData, setCurrentSelPerson, setCurrentSelectedTrackId, getCurrentSelectedModelLabel} from "./state.js";


// updates the dashboard to display the current sequence-level statistics and accuracy metrics
export function updGenMetrics() {

    if (generalFps) generalFps.textContent = generalMetrics.fps;
    if (generalAde) generalAde.textContent = generalMetrics.ade;
    if (generalFde) generalFde.textContent = generalMetrics.fde;
    if (personFps) personFps.textContent = generalMetrics.fps;
}


// switches the interface to General mode by showing the general panel, hiding person details, clearing selections, and displaying overall metrics
export function setGeneralMode() {
    setCurrentSelPerson(null);
    setCurrentSelectedTrackId(null);
    window.dispatchEvent(new CustomEvent("selectedPersonChanged", { detail: { person: null } }));
    window.dispatchEvent(new CustomEvent("selectedTrackChanged", { detail: { trackId: null } }));

    if (generalPanel) generalPanel.classList.remove("hidden");
    if (personPanel) personPanel.classList.add("hidden");

    if (genButton) genButton.classList.add("active");
    personButtons.forEach((btn) => btn.classList.remove("active"));

    if (videoSubtitle) videoSubtitle.style.display = "none";
    if (videoSelectedTarget) videoSelectedTarget.textContent = "No selection";
    if (selectedTarget) selectedTarget.textContent = "No selection";


    // these are still frontend resets, but the real values in person mode will come from backend data.
    if (prsId) prsId.textContent = "—";
    if (prsStatus) prsStatus.textContent = "—";
    if (prsHistory) prsHistory.textContent = "—";
    if (prsHorizon) prsHorizon.textContent = "—";
    if (prsAde) prsAde.textContent = "—";
    if (prsFde) prsFde.textContent = "—";

    // LATER: generalMetrics will be filled by backend updates
    updGenMetrics();

    // LATER: if backend sends model/config info, this can be filled there too
    if (modelSetup) {
        modelSetup.textContent = getCurrentSelectedModelLabel();
    }
}



export function setPersonMode(person) {
    setCurrentSelPerson(person);
    setCurrentSelectedTrackId(null);
    window.dispatchEvent(new CustomEvent("selectedPersonChanged", { detail: { person } }));
    window.dispatchEvent(new CustomEvent("selectedTrackChanged", { detail: { trackId: null } }));
    const data = getPersonData(person);

    if (generalPanel) generalPanel.classList.add("hidden");
    if (personPanel) personPanel.classList.remove("hidden");

    if (genButton) genButton.classList.remove("active");
    personButtons.forEach((btn) => btn.classList.remove("active"));

    const clickedButton = document.querySelector(`.chip[data-person="${person}"]`);
    if (clickedButton) {
        clickedButton.classList.add("active");
    }

    if (videoSubtitle) videoSubtitle.style.display = "block";
    if (videoSelectedTarget) videoSelectedTarget.textContent = data.id;
    if (selectedTarget) selectedTarget.textContent = data.id;

    // LATER: these should come from backend, via state.js
    if (prsId) prsId.textContent = data.id;
    if (prsStatus) prsStatus.textContent = data.status;
    if (prsHistory) prsHistory.textContent = data.history;
    if (prsHorizon) prsHorizon.textContent = data.horizon;
    if (prsAde) prsAde.textContent = data.ade;
    if (prsFde) prsFde.textContent = data.fde;

    if (personFps) personFps.textContent = generalMetrics.fps;

    if (modelSetup) {
        modelSetup.textContent = getCurrentSelectedModelLabel();
    }
}


export function setTrackMode(trackId) {
    const normalizedTrackId = Number(trackId);
    if (!Number.isFinite(normalizedTrackId)) {
        return;
    }

    setCurrentSelPerson(null);
    setCurrentSelectedTrackId(normalizedTrackId);
    window.dispatchEvent(new CustomEvent("selectedPersonChanged", { detail: { person: null } }));
    window.dispatchEvent(new CustomEvent("selectedTrackChanged", { detail: { trackId: normalizedTrackId } }));

    if (generalPanel) generalPanel.classList.add("hidden");
    if (personPanel) personPanel.classList.remove("hidden");

    if (genButton) genButton.classList.remove("active");
    personButtons.forEach((btn) => btn.classList.remove("active"));

    const trackLabel = `Track ${normalizedTrackId}`;
    if (videoSubtitle) videoSubtitle.style.display = "block";
    if (videoSelectedTarget) videoSelectedTarget.textContent = trackLabel;
    if (selectedTarget) selectedTarget.textContent = trackLabel;

    if (prsId) prsId.textContent = trackLabel;
    if (prsStatus) prsStatus.textContent = "Tracked";
    if (prsHistory) prsHistory.textContent = "live";
    if (prsHorizon) prsHorizon.textContent = "live";
    if (prsAde) prsAde.textContent = "—";
    if (prsFde) prsFde.textContent = "—";

    if (personFps) personFps.textContent = generalMetrics.fps;
    if (modelSetup) modelSetup.textContent = getCurrentSelectedModelLabel();
}


export function setVideo(video, { emitEvent = true } = {}) {
    if (videoSelect) {
        videoSelect.value = video;
    }

    if (videoLabel) videoLabel.textContent = `Fixed camera feed • Video ${video}`;
    if (emitEvent) {
        window.dispatchEvent(new CustomEvent("activeVideoChanged", { detail: { video } }));
    }
}
