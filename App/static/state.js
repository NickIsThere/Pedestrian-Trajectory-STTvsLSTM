/*
Main writer: Keez Cuijpers
Reviewer: Noah Nuelandt
Contributors: Claire Bams.Néo Deward
*/

export let currentSelectedPerson = null;
export let currentSelectedTrackId = null;

export const generalMetrics = {fps: "—", ade: "—", fde: "—"};

export const groundTruthState = {
    frameId: 0,
    tracks: [],
    allTracks: [],
    isLoading: false
};

export const predictionState = {
    tracks: [],
};

export function setPredictionTracks(tracks) {
    predictionState.tracks = tracks || [];
    window.dispatchEvent(new CustomEvent("predictionTracksLoaded"));
}

export const peopleData = {
    A: { id: "Person A", status: "—", history: "—", horizon: "—", ade: "—", fde: "—", model: "—" },
    B: { id: "Person B", status: "—", history: "—", horizon: "—", ade: "—", fde: "—", model: "—" },
    C: { id: "Person C", status: "—", history: "—", horizon: "—", ade: "—", fde: "—", model: "—" },
    D: { id: "Person D", status: "—", history: "—", horizon: "—", ade: "—", fde: "—", model: "—" }
};

export function setCurrentSelPerson(person) {
    currentSelectedPerson = person;
}

export function setCurrentSelectedTrackId(trackId) {
    if (trackId === null || trackId === undefined) {
        currentSelectedTrackId = null;
        return;
    }
    const normalized = Number(trackId);
    currentSelectedTrackId = Number.isFinite(normalized) ? normalized : null;
}

export function getPersonData(person) {
    return peopleData[person];
}

export function updateGenValue(data){
    if (data.fps !== undefined) generalMetrics.fps = data.fps;
    if (data.ade !== undefined) generalMetrics.ade = data.ade;
    if (data.fde !== undefined) generalMetrics.fde = data.fde;
}

export function updatePersonData(person, data){
    if (!peopleData[person]) return;
    for (const key in data) {
        peopleData[person][key] = data[key];
    }
}

export function setAllTracks(tracks) {
    groundTruthState.allTracks = tracks || [];
    updateGroundTruth(groundTruthState.frameId);
    window.dispatchEvent(new CustomEvent("tracksLoaded"));
}

export function updateGroundTruth(frameId, tracks) {
    if (tracks !== undefined) {
        groundTruthState.frameId = frameId;
        groundTruthState.tracks = tracks || [];
        return;
    }
    groundTruthState.frameId = frameId;
    const currentTracks = [];
    for (const track of groundTruthState.allTracks) {
        const point = track.points.find(p => Number(p.frame_id) === Number(frameId));
        if (point) {
            currentTracks.push({
                track_id: track.id,
                x: point.x,
                y: point.y
            });
        }
    }
    groundTruthState.tracks = currentTracks;
}


export let currentSelM = "kalman";

export function setCurrentSelM(model) {
    currentSelM = model || "kalman";
}

export function getCurrentSelectedModelLabel() {
    const select = document.getElementById("model-select");
    if (select && select.options.length > 0 && select.selectedIndex >= 0) {
        return select.options[select.selectedIndex].text;
    }
    return currentSelM === "kalman" ? "Kalman Filter (Baseline)" : currentSelM;
}