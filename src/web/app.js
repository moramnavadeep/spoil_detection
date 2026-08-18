/**
 * NutriFresh AI - Frontend Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");
  const dropzonePrompt = document.getElementById("dropzonePrompt");
  const previewContainer = document.getElementById("previewContainer");
  const previewImage = document.getElementById("previewImage");
  const removeBtn = document.getElementById("removeBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const analyzeSpinner = document.getElementById("analyzeSpinner");
  const idleState = document.getElementById("idleState");
  const resultsContent = document.getElementById("resultsContent");
  const latencyBadge = document.getElementById("latencyBadge");

  // Results fields
  const statusBanner = document.getElementById("statusBanner");
  const resultStatusIcon = document.getElementById("resultStatusIcon");
  const resultStatusLabel = document.getElementById("resultStatusLabel");
  const resultConfidenceText = document.getElementById("resultConfidenceText");
  const probFreshVal = document.getElementById("probFreshVal");
  const probFreshBar = document.getElementById("probFreshBar");
  const probRottenVal = document.getElementById("probRottenVal");
  const probRottenBar = document.getElementById("probRottenBar");
  const metaLatency = document.getElementById("metaLatency");
  const metaVersion = document.getElementById("metaVersion");

  let currentFile = null;

  // File browser trigger
  browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener("click", () => {
    if (!currentFile) fileInput.click();
  });

  // Drag & drop handlers
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleSelectedFile(files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  removeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUpload();
  });

  function handleSelectedFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file (JPG, PNG, WEBP).");
      return;
    }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (event) => {
      previewImage.src = event.target.result;
      dropzonePrompt.style.display = "none";
      previewContainer.style.display = "flex";
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function resetUpload() {
    currentFile = null;
    fileInput.value = "";
    previewImage.src = "";
    dropzonePrompt.style.display = "flex";
    previewContainer.style.display = "none";
    analyzeBtn.disabled = true;
    idleState.style.display = "flex";
    resultsContent.style.display = "none";
    latencyBadge.style.display = "none";
  }

  // Sample Chips
  document.querySelectorAll(".sample-chip").forEach((chip) => {
    chip.addEventListener("click", async () => {
      const src = chip.getAttribute("data-src");
      try {
        const response = await fetch(src);
        const blob = await response.blob();
        const filename = src.split("/").pop();
        const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
        handleSelectedFile(file);
        // Automatically trigger prediction for seamless demo
        setTimeout(runInference, 200);
      } catch (err) {
        console.error("Failed to load sample image:", err);
      }
    });
  });

  // Run Inference API
  analyzeBtn.addEventListener("click", runInference);

  async function runInference() {
    if (!currentFile) return;

    analyzeBtn.disabled = true;
    analyzeSpinner.style.display = "inline-block";

    const formData = new FormData();
    formData.append("file", currentFile);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Inference failed.");
      }

      const result = await response.json();
      displayResults(result);
    } catch (err) {
      alert(`Error analyzing image: ${err.message}`);
    } finally {
      analyzeBtn.disabled = false;
      analyzeSpinner.style.display = "none";
    }
  }

  function displayResults(data) {
    idleState.style.display = "none";
    resultsContent.style.display = "flex";
    latencyBadge.style.display = "inline-block";
    latencyBadge.textContent = `${data.latency_ms} ms`;

    const isFresh = data.is_fresh;

    // Update status banner
    statusBanner.className = `status-banner ${isFresh ? "is-fresh" : "is-rotten"}`;
    resultStatusIcon.textContent = isFresh ? "🍏" : "🥀";
    resultStatusLabel.textContent = isFresh ? "FRESH & CONSUMABLE" : "SPOILED / ROTTEN";
    resultConfidenceText.textContent = `Confidence: ${data.confidence_percentage}%`;

    // Probability breakdown
    const freshPct = ((data.probabilities.Fresh || 0) * 100).toFixed(1);
    const rottenPct = ((data.probabilities.Rotten || 0) * 100).toFixed(1);

    probFreshVal.textContent = `${freshPct}%`;
    probFreshBar.style.width = `${freshPct}%`;

    probRottenVal.textContent = `${rottenPct}%`;
    probRottenBar.style.width = `${rottenPct}%`;

    // Technical metadata
    metaLatency.textContent = `${data.latency_ms} ms`;
    metaVersion.textContent = `v${data.model_version}`;
  }

  // Health check on boot
  fetch("/health")
    .then((r) => r.json())
    .then((data) => {
      const statusEl = document.getElementById("systemStatus");
      if (data.status === "healthy") {
        statusEl.innerHTML = `<span class="status-indicator"></span> Model Ready (v${data.version})`;
      }
    })
    .catch(() => {
      const statusEl = document.getElementById("systemStatus");
      statusEl.style.color = "#dc2626";
      statusEl.innerHTML = `<span class="status-indicator" style="background:#dc2626"></span> Server Offline`;
    });
});
