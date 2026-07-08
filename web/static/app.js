const inputFile = document.getElementById("inputFile");
const outputFile = document.getElementById("outputFile");
const inputVideo = document.getElementById("inputVideo");
const outputVideo = document.getElementById("outputVideo");
const inputName = document.getElementById("inputName");
const outputName = document.getElementById("outputName");
const inputPlaceholder = document.getElementById("inputPlaceholder");
const outputPlaceholder = document.getElementById("outputPlaceholder");
const swapBtn = document.getElementById("swapBtn");
const processBtn = document.getElementById("processBtn");
const statusEl = document.getElementById("status");
const statusDot = document.getElementById("statusDot");
const progressBar = document.getElementById("progressBar");
const progressText = document.getElementById("progressText");
const metaStatus = document.getElementById("metaStatus");

let selectedInputFile = null;
let pollTimer = null;
let inputObjectUrl = null;
let outputObjectUrl = null;

const clearDotState = () => {
  statusDot.classList.remove("working", "success", "error");
};

const setStatus = (message, state = "idle") => {
  statusEl.textContent = message;
  statusEl.classList.remove(
    "state-idle",
    "state-working",
    "state-success",
    "state-error"
  );
  clearDotState();

  // Update meta status indicator
  if (metaStatus) {
    if (state === "working") {
      metaStatus.textContent = "Processing...";
    } else if (state === "success") {
      metaStatus.textContent = "Complete";
    } else if (state === "error") {
      metaStatus.textContent = "Error";
    } else {
      metaStatus.textContent = "Ready";
    }
  }

  if (state === "working") {
    statusEl.classList.add("state-working");
    statusDot.classList.add("working");
    return;
  }

  if (state === "success") {
    statusEl.classList.add("state-success");
    statusDot.classList.add("success");
    return;
  }

  if (state === "error") {
    statusEl.classList.add("state-error");
    statusDot.classList.add("error");
    return;
  }

  statusEl.classList.add("state-idle");
};

const setProgress = (value) => {
  const pct = Math.max(0, Math.min(100, Math.floor(value || 0)));
  progressBar.value = pct;
  progressText.textContent = `${pct}%`;
};

const setVideo = (file, videoEl, labelEl, placeholderEl) => {
  if (!file) {
    labelEl.textContent = "none";
    return;
  }

  const url = URL.createObjectURL(file);
  if (videoEl === inputVideo) {
    if (inputObjectUrl) {
      URL.revokeObjectURL(inputObjectUrl);
    }
    inputObjectUrl = url;
  } else if (videoEl === outputVideo) {
    if (outputObjectUrl) {
      URL.revokeObjectURL(outputObjectUrl);
    }
    outputObjectUrl = url;
  }

  videoEl.src = url;
  videoEl.load();
  placeholderEl.style.display = "none";
  labelEl.textContent = file.name;
};

const stopPolling = () => {
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
};

const setBusy = (isBusy) => {
  processBtn.disabled = isBusy;
  inputFile.disabled = isBusy;
};

const pollJobProgress = async (jobId) => {
  try {
    const progressResp = await fetch(`/api/progress/${jobId}`);
    if (!progressResp.ok) {
      throw new Error("Failed to get progress");
    }

    const progress = await progressResp.json();
    const pct = progress.percent || 0;
    setProgress(pct);
    setStatus(`${progress.message || "Processing"} (${pct}%)`, "working");

    if (progress.status === "completed") {
      stopPolling();
      if (!progress.output_url) {
        setStatus("Error: Completed state has no output URL.", "error");
        setBusy(false);
        return;
      }

      outputVideo.src = `${progress.output_url}?t=${Date.now()}`;
      outputVideo.load();
      outputVideo.play().catch(() => {});
      outputPlaceholder.style.display = "none";
      outputName.textContent = progress.output_name || "processed.mp4";
      setProgress(100);
      setStatus("Processing complete. Output loaded.", "success");
      setBusy(false);
      return;
    }

    if (progress.status === "error") {
      stopPolling();
      setStatus(`Error: ${progress.message || "Processing failed"}`, "error");
      setBusy(false);
      return;
    }

    pollTimer = setTimeout(() => pollJobProgress(jobId), 1000);
  } catch (err) {
    stopPolling();
    setStatus(`Error: ${err.message}`, "error");
    setBusy(false);
  }
};

inputFile.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) {
    selectedInputFile = file;
  }
  setVideo(file, inputVideo, inputName, inputPlaceholder);
  setStatus("Ready to process.", "idle");
});

outputFile.addEventListener("change", (event) => {
  const [file] = event.target.files;
  setVideo(file, outputVideo, outputName, outputPlaceholder);
  if (file) {
    setStatus("Output loaded for review.", "success");
  }
});

swapBtn?.addEventListener("click", () => {
  const inputParent = document.querySelector(".screen[data-role='input']");
  const outputParent = document.querySelector(".screen[data-role='output']");
  const stage = document.querySelector(".stage");

  stage.style.opacity = "0.6";
  stage.insertBefore(outputParent, inputParent);
  window.setTimeout(() => {
    stage.style.opacity = "1";
  }, 160);
});

processBtn.addEventListener("click", async () => {
  if (!selectedInputFile) {
    setStatus("Select an input video first.", "error");
    return;
  }

  stopPolling();
  setStatus("Processing... this can take a few minutes.", "working");
  setBusy(true);
  setProgress(0);

  const formData = new FormData();
  formData.append("file", selectedInputFile);

  try {
    const response = await fetch("/api/process", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || "Failed to process video");
    }

    const { job_id: jobId } = await response.json();
    setStatus("Processing started...", "working");
    pollJobProgress(jobId);
  } catch (error) {
    setStatus(`Error: ${error.message}`, "error");
    setBusy(false);
  }
});
