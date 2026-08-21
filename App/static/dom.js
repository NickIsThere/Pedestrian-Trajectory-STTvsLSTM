/*
Main writer: Claire Bams
Reviewer: Noah Nuelandt
Contributors: Keez Cuijpers
*/

// buttons (top bar of GUI)
export const genButton = document.getElementById("general-button");
export const personButtons = document.querySelectorAll(".chip[data-person]");

// panels (right side dashboard)
export const generalPanel = document.getElementById("general-panel");
export const personPanel = document.getElementById("person-panel");

// selected target fields
export const videoSelectedTarget = document.getElementById("video-selected-target");
export const videoSubtitle = document.getElementById("video-subtitle");
export const selectedTarget = document.getElementById("selected-target");


// person details
export const prsId = document.getElementById("person-id");
export const prsStatus = document.getElementById("person-status");//status of the selected person (Tracked, lost, occluded, standing, walking) not really neces ( could eb remo
export const prsHistory = document.getElementById("person-history"); //how much past trajectory data is used (8 frames, 2 sec, 12 obs)
export const prsHorizon = document.getElementById("person-horizon");// how far into the future the model predicts
export const modelSetup = document.getElementById("model-setup");


// person mode metrics --> then it is the values for a certain person
export const prsAde = document.getElementById("person-ade");
export const prsFde = document.getElementById("person-fde");

// General mode metrics
export const generalFps = document.getElementById("general-fps");
export const generalAde = document.getElementById("general-ade");
export const generalFde = document.getElementById("general-fde");

export const personFps = document.getElementById("person-fps");

//video related
export const videoSelect = document.getElementById("video-select");
export const videoFeed = document.getElementById("video-feed");
export const videoLabel = document.getElementById("video-label");
export const videoErrorBanner = document.getElementById("video-error-banner");
export const framePlayToggle = document.getElementById("frame-play-toggle");
export const frameIndicator = document.getElementById("frame-indicator");
export const combinedPanel = document.getElementById("combined-panel");
export const futurePredictionPanel = document.getElementById("future-prediction-panel");

export const modelSelect = document.getElementById("model-select");
export const combinedPanelTitle = document.getElementById("combined-panel-title");