/* ══════════════════════════════════════════════════════════════════════
   Sentry Console — front end
   Tabs: Upload -> People (registration) -> Processing -> Results
   ══════════════════════════════════════════════════════════════════════ */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// ── toast ────────────────────────────────────────────────────────────────
let toastTimer = null;
function toast(msg, isError = false) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.toggle("error", isError);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), 3200);
}

// ── theme (dark / light) ─────────────────────────────────────────────────
const THEME_KEY = "sentry-console-theme";
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  $("#themeToggle")?.setAttribute("aria-pressed", String(theme === "light"));
}
function initTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  const prefersLight = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
  applyTheme(saved || (prefersLight ? "light" : "dark"));
}
$("#themeToggle").addEventListener("click", () => {
  const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
  applyTheme(next);
  localStorage.setItem(THEME_KEY, next);
  logEvent("info", `Theme switched to ${next} mode`);
});
initTheme();

// ── tabs ─────────────────────────────────────────────────────────────────
function goToTab(name) {
  $$(".tab-btn").forEach((b) => {
    const active = b.dataset.tab === name;
    b.classList.toggle("active", active);
    b.setAttribute("aria-selected", String(active));
  });
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  if (name === "people") loadPersons();
  if (name === "results") {
    loadStorage();
    if (currentSessionId) loadResults();
  }
  if (name === "telemetry") startTelemetry();
  else stopTelemetry();
  if (name === "process" && activePreviewCamId) {
    // panel just became visible — canvas had 0 size while hidden, so resize before resuming
    requestAnimationFrame(resumePreviewPlayback);
  } else {
    pausePreviewPlayback();
  }
}
$$(".tab-btn").forEach((b) => b.addEventListener("click", () => goToTab(b.dataset.tab)));

/* ══════════════════════════════════════════════════════════════════════
   01 — UPLOAD
   ══════════════════════════════════════════════════════════════════════ */

let cameraFiles = {}; // slotIndex -> File
let currentSessionId = null;
let sessionCameraLabels = {};

const cameraCountInput = $("#cameraCount");
const cameraSlots = $("#cameraSlots");

function renderCameraSlots() {
  const n = Math.max(1, Math.min(12, parseInt(cameraCountInput.value || "1", 10)));
  cameraCountInput.value = n;
  cameraSlots.innerHTML = "";

  const keepFiles = {};
  for (let i = 0; i < n; i++) {
    const existing = cameraFiles[i];
    if (existing) keepFiles[i] = existing;

    const slot = document.createElement("div");
    slot.className = "camera-slot" + (existing ? " filled" : "");
    slot.innerHTML = `
      <div class="camera-slot-head">
        <input class="camera-label" type="text" value="Camera ${i + 1}" data-idx="${i}" />
        <span class="badge ${existing ? "ok" : ""}">${existing ? "READY" : "EMPTY"}</span>
      </div>
      <label class="camera-slot-drop" data-idx="${i}">
        ${existing ? "Click to replace clip" : "Click to choose a video clip"}
        <input type="file" accept="video/*" hidden data-idx="${i}" />
      </label>
      <div class="camera-slot-file" data-idx="${i}">${existing ? existing.name : ""}</div>
    `;
    cameraSlots.appendChild(slot);
  }
  cameraFiles = keepFiles;

  $$(".camera-slot-drop input[type=file]").forEach((input) => {
    input.addEventListener("change", (e) => {
      const idx = e.target.dataset.idx;
      const file = e.target.files[0];
      if (file) {
        cameraFiles[idx] = file;
        renderCameraSlots();
      }
    });
  });
  updateUploadReadiness();
}

function updateUploadReadiness() {
  const n = parseInt(cameraCountInput.value || "1", 10);
  const filled = Object.keys(cameraFiles).length;
  const btn = $("#startProcessingBtn");
  const hint = $("#uploadHint");
  const ready = filled === n && n > 0;
  btn.disabled = !ready;
  hint.textContent = ready
    ? `${filled} of ${n} camera clips ready.`
    : `${filled} of ${n} camera clips attached — attach a video to every slot to continue.`;
}

$("#camMinus").addEventListener("click", () => {
  cameraCountInput.value = Math.max(1, parseInt(cameraCountInput.value || "1", 10) - 1);
  renderCameraSlots();
});
$("#camPlus").addEventListener("click", () => {
  cameraCountInput.value = Math.min(12, parseInt(cameraCountInput.value || "1", 10) + 1);
  renderCameraSlots();
});
cameraCountInput.addEventListener("change", renderCameraSlots);

$("#startProcessingBtn").addEventListener("click", async () => {
  const labels = $$(".camera-label").map((i) => i.value.trim() || i.dataset.idx);
  const files = Object.keys(cameraFiles)
    .sort((a, b) => a - b)
    .map((k) => cameraFiles[k]);

  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  labels.forEach((l) => form.append("labels", l));

  $("#startProcessingBtn").disabled = true;
  $("#uploadHint").textContent = "Uploading and starting the pipeline…";

  try {
    const resp = await fetch("/api/session/start", { method: "POST", body: form });
    if (!resp.ok) throw new Error((await resp.text()) || "Failed to start processing");
    const data = await resp.json();
    currentSessionId = data.session_id;
    sessionCameraLabels = {};
    data.cameras.forEach((camId, i) => (sessionCameraLabels[camId] = labels[i]));

    // release any object URLs from a previous run, then create fresh ones
    // so the Processing tab can play the actual uploaded clips underneath
    // the bounding-box overlay canvas.
    Object.values(sessionVideoUrls).forEach((url) => URL.revokeObjectURL(url));
    Object.keys(sessionVideoUrls).forEach((k) => delete sessionVideoUrls[k]);
    data.cameras.forEach((camId, i) => {
      sessionVideoUrls[camId] = URL.createObjectURL(files[i]);
    });
    populatePreviewCameras(data.cameras);

    toast("Processing started");
    logEvent("info", `Session ${currentSessionId} started — ${files.length} camera(s)`);
    goToTab("process");
    startProgressPolling();
  } catch (err) {
    toast(`Error: ${err.message}`, true);
  } finally {
    $("#startProcessingBtn").disabled = false;
    updateUploadReadiness();
  }
});

renderCameraSlots();

/* ══════════════════════════════════════════════════════════════════════
   02 — PEOPLE / IDENTITY DATABASE
   ══════════════════════════════════════════════════════════════════════ */

let pendingPhotos = [];

function renderPhotoPreview(container, files, onRemove) {
  container.innerHTML = "";
  files.forEach((file, idx) => {
    const chip = document.createElement("div");
    chip.className = "photo-chip";
    const url = URL.createObjectURL(file);
    chip.innerHTML = `<img src="${url}" alt="${file.name}" /><button type="button" aria-label="Remove photo">×</button>`;
    chip.querySelector("button").addEventListener("click", () => onRemove(idx));
    container.appendChild(chip);
  });
}

function refreshRegisterPreview() {
  renderPhotoPreview($("#photoPreview"), pendingPhotos, (idx) => {
    pendingPhotos.splice(idx, 1);
    refreshRegisterPreview();
  });
}
$("#personPhotos").addEventListener("change", (e) => {
  pendingPhotos = pendingPhotos.concat(Array.from(e.target.files));
  refreshRegisterPreview();
  e.target.value = "";
});

$("#registerForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = $("#personName").value.trim();
  const hint = $("#registerHint");

  if (!name) return toast("Enter a name first", true);
  if (pendingPhotos.length === 0) return toast("Add at least one photo", true);

  const form = new FormData();
  form.append("name", name);
  pendingPhotos.forEach((f) => form.append("files", f));

  const btn = $("#registerBtn");
  btn.disabled = true;
  hint.textContent = "Generating embeddings…";
  hint.className = "hint";

  try {
    const resp = await fetch("/api/persons", { method: "POST", body: form });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || "Registration failed");
    }
    toast(`${name} registered`);
    logEvent("info", `Registered "${name}" with ${pendingPhotos.length} photo(s)`);
    btn.classList.add("success-pulse");
    setTimeout(() => btn.classList.remove("success-pulse"), 700);
    $("#personName").value = "";
    pendingPhotos = [];
    refreshRegisterPreview();
    loadPersons();
  } catch (err) {
    hint.textContent = err.message;
    hint.className = "hint error";
  } finally {
    btn.disabled = false;
  }
});

function initials(name) {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0].toUpperCase())
    .join("");
}

function personCard(p) {
  const el = document.createElement("div");
  el.className = "person-card";
  const when = p.registered_at ? new Date(p.registered_at).toLocaleDateString() : "—";
  const photos = p.photos || [];
  const thumbsHtml = photos
    .slice(0, 5)
    .map(
      (url, i) =>
        `<button type="button" class="gallery-thumb" data-photo-idx="${i}" aria-label="View photo ${i + 1} of ${escapeHtml(p.name)}">
           <img src="${url}" alt="" loading="lazy" />
         </button>`
    )
    .join("");
  const moreCount = photos.length - 5;

  el.innerHTML = `
    <div class="person-card-head">
      <div class="person-avatar">${initials(p.name)}</div>
      <div>
        <div class="person-name">${escapeHtml(p.name)}</div>
        <div class="person-meta">${p.num_images} photo${p.num_images === 1 ? "" : "s"} · added ${when}</div>
      </div>
    </div>
    ${
      photos.length
        ? `<div class="gallery-strip">${thumbsHtml}${moreCount > 0 ? `<span class="gallery-more">+${moreCount}</span>` : ""}</div>`
        : `<div class="gallery-strip empty">No photos on file</div>`
    }
    <div class="person-card-actions">
      <button class="btn ghost small" data-edit="${escapeHtml(p.name)}">Edit</button>
      <button class="btn danger small" data-delete="${escapeHtml(p.name)}">Delete</button>
    </div>
  `;

  el.querySelectorAll(".gallery-thumb").forEach((btn) => {
    btn.addEventListener("click", () => openLightbox(p.name, photos, Number(btn.dataset.photoIdx)));
  });

  return el;
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

async function loadPersons(query = "") {
  const list = $("#personList");
  const empty = $("#personEmpty");
  const count = $("#personCount");
  try {
    const url = query ? `/api/persons/search?q=${encodeURIComponent(query)}` : "/api/persons";
    const resp = await fetch(url);
    if (!resp.ok) throw new Error("Failed to load identity database");
    const persons = await resp.json();

    list.innerHTML = "";
    empty.hidden = persons.length > 0;
    count.textContent = query
      ? `${persons.length} match${persons.length === 1 ? "" : "es"} for "${query}"`
      : `${persons.length} registered`;

    persons.forEach((p) => list.appendChild(personCard(p)));

    list.querySelectorAll("[data-edit]").forEach((btn) =>
      btn.addEventListener("click", () => openEditModal(btn.dataset.edit))
    );
    list.querySelectorAll("[data-delete]").forEach((btn) =>
      btn.addEventListener("click", () => openDeleteModal(btn.dataset.delete))
    );
  } catch (err) {
    toast(err.message, true);
  }
}

let searchDebounce = null;
$("#personSearch").addEventListener("input", (e) => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(() => loadPersons(e.target.value.trim()), 250);
});

// ── delete modal ─────────────────────────────────────────────────────────
let deleteTarget = null; // { kind: "person"|"run"|"all", id: string|null }
function openDeleteModal(name) {
  deleteTarget = { kind: "person", id: name };
  $("#confirmTitle").textContent = "Delete this person?";
  $("#confirmBody").textContent = `This removes "${name}"'s photos and embeddings from the identity database. This can't be undone.`;
  $("#confirmModal").hidden = false;
}
$("#confirmCancel").addEventListener("click", () => ($("#confirmModal").hidden = true));
$("#confirmDelete").addEventListener("click", async () => {
  if (!deleteTarget) return;
  const { kind, id } = deleteTarget;
  $("#confirmDelete").disabled = true;
  try {
    if (kind === "person") {
      const resp = await fetch(`/api/persons/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (!resp.ok) throw new Error("Delete failed");
      toast(`${id} removed`);
      logEvent("warn", `Removed "${id}" from the identity database`);
      loadPersons($("#personSearch").value.trim());
    } else if (kind === "run") {
      const resp = await fetch(`/api/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (!resp.ok) throw new Error((await resp.text()) || "Delete failed");
      toast("Run deleted");
      afterSessionDelete();
    } else if (kind === "all") {
      const resp = await fetch(`/api/sessions`, { method: "DELETE" });
      if (!resp.ok) throw new Error((await resp.text()) || "Delete failed");
      toast("Completed runs cleared");
      afterSessionDelete();
    }
  } catch (err) {
    toast(err.message, true);
  } finally {
    $("#confirmDelete").disabled = false;
    $("#confirmModal").hidden = true;
    deleteTarget = null;
  }
});

// ── run storage (Results tab) ────────────────────────────────────────────
function fmtBytes(n) {
  if (!Number.isFinite(n) || n <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

async function loadStorage() {
  try {
    const resp = await fetch(`/api/storage`);
    if (!resp.ok) throw new Error("Storage unavailable");
    const data = await resp.json();
    const thisRun = currentSessionId
      ? data.sessions.find((s) => s.session_id === currentSessionId)
      : null;
    $("#storageRunCount").textContent = data.session_count;
    $("#storageAllRuns").textContent = fmtBytes(data.total_size);
    $("#storageThisRun").textContent = thisRun ? fmtBytes(thisRun.size) : "—";
  } catch (err) {
    toast(err.message, true);
  }
}

function resetResultsView() {
  $("#statDetected").textContent = 0;
  $("#statMatched").textContent = 0;
  $("#statUnknown").textContent = 0;
  $("#statCameras").textContent = 0;
  $("#matchedList").innerHTML = "";
  $("#unmatchedList").innerHTML = "";
  lastMatched = [];
}

async function afterSessionDelete() {
  if (currentSessionId) {
    try {
      const resp = await fetch(`/api/session/${currentSessionId}/progress`);
      if (resp.ok) {
        // session survived (still processing) — keep the UI attached to it
        loadStorage();
        return;
      }
    } catch (_) {}
    currentSessionId = null;
  }
  resetResultsView();
  Object.values(sessionVideoUrls).forEach((url) => URL.revokeObjectURL(url));
  Object.keys(sessionVideoUrls).forEach((k) => delete sessionVideoUrls[k]);
  Object.keys(realTrackCache).forEach((k) => delete realTrackCache[k]);
  pausePreviewPlayback();
  $("#previewPanel").hidden = true;
  activePreviewCamId = null;
  loadStorage();
}

$("#deleteRunBtn").addEventListener("click", () => {
  if (!currentSessionId) {
    toast("No active run to delete", true);
    return;
  }
  deleteTarget = { kind: "run", id: currentSessionId };
  $("#confirmTitle").textContent = "Delete this run?";
  $("#confirmBody").textContent =
    "This removes this run's output video, evidence frames, and uploaded source clip. Registered people are not affected.";
  $("#confirmModal").hidden = false;
});

$("#clearRunsBtn").addEventListener("click", () => {
  deleteTarget = { kind: "all", id: null };
  $("#confirmTitle").textContent = "Clear all completed runs?";
  $("#confirmBody").textContent =
    "This removes output videos, evidence frames, and uploaded source clips for every completed run. Runs still processing and registered people are not affected.";
  $("#confirmModal").hidden = false;
});

// ── edit modal ───────────────────────────────────────────────────────────
let editTarget = null;
let editPendingPhotos = [];

function openEditModal(name) {
  editTarget = name;
  editPendingPhotos = [];
  $("#editName").value = name;
  $("#editPhotoPreview").innerHTML = "";
  $("#editHint").textContent = "";
  $("#editModal").hidden = false;
}
$("#editCancel").addEventListener("click", () => ($("#editModal").hidden = true));
function refreshEditPreview() {
  renderPhotoPreview($("#editPhotoPreview"), editPendingPhotos, (idx) => {
    editPendingPhotos.splice(idx, 1);
    refreshEditPreview();
  });
}
$("#editPhotos").addEventListener("change", (e) => {
  editPendingPhotos = editPendingPhotos.concat(Array.from(e.target.files));
  refreshEditPreview();
  e.target.value = "";
});

$("#editSave").addEventListener("click", async () => {
  const newName = $("#editName").value.trim();
  const hint = $("#editHint");
  if (!newName) return (hint.textContent = "Name can't be empty");

  try {
    if (editPendingPhotos.length > 0) {
      const form = new FormData();
      editPendingPhotos.forEach((f) => form.append("files", f));
      const resp = await fetch(`/api/persons/${encodeURIComponent(editTarget)}/images`, {
        method: "PUT",
        body: form,
      });
      if (!resp.ok) throw new Error("Could not add photos");
    }

    if (newName !== editTarget) {
      const form = new FormData();
      form.append("new_name", newName);
      const resp = await fetch(`/api/persons/${encodeURIComponent(editTarget)}/rename`, {
        method: "PUT",
        body: form,
      });
      if (!resp.ok) throw new Error("Could not rename — name may already exist");
    }

    toast("Changes saved");
    $("#editModal").hidden = true;
    loadPersons($("#personSearch").value.trim());
  } catch (err) {
    hint.textContent = err.message;
    hint.className = "hint error";
  }
});

/* ══════════════════════════════════════════════════════════════════════
   03 — PROCESSING STATUS
   ══════════════════════════════════════════════════════════════════════ */

let progressPollTimer = null;

function statusBadge(status) {
  if (status === "completed") return `<span class="badge ok">DONE</span>`;
  if (status === "error") return `<span class="badge err">ERROR</span>`;
  if (status === "running") return `<span class="badge warn">PROCESSING</span>`;
  return `<span class="badge">QUEUED</span>`;
}

function dotClass(status) {
  if (status === "completed") return "success";
  if (status === "error") return "error";
  if (status === "running") return "working";
  return "";
}

const cameraStatusSeen = {};
function renderCameraStatus(cameras) {
  $("#processEmpty").hidden = true;
  const list = $("#cameraStatusList");
  list.innerHTML = "";
  cameras.forEach((cam) => {
    const prev = cameraStatusSeen[cam.camera_id];
    if (prev !== cam.status) {
      cameraStatusSeen[cam.camera_id] = cam.status;
      if (cam.status === "completed") {
        logEvent("info", `${cam.label}: pipeline completed`);
        maybeFetchRealTracks(cam.camera_id);
      }
      if (cam.status === "error") logEvent("error", `${cam.label}: ${cam.message}`);
      if (cam.status === "running" && prev === "queued") logEvent("info", `${cam.label}: detection started`);
    }
    const row = document.createElement("div");
    row.className = "camera-status-row";
    row.innerHTML = `
      <div class="camera-status-top">
        <div>
          <span class="pulse-dot ${dotClass(cam.status)}"></span>
          <span class="camera-status-name">${escapeHtml(cam.label)}</span>
        </div>
        ${statusBadge(cam.status)}
      </div>
      <progress max="100" value="${cam.percent}"></progress>
      <div class="camera-status-msg">${escapeHtml(cam.message)} — ${cam.percent}%</div>
    `;
    list.appendChild(row);
  });
}

async function pollProgress() {
  if (!currentSessionId) return;
  try {
    const resp = await fetch(`/api/session/${currentSessionId}/progress`);
    if (!resp.ok) throw new Error("Session not found");
    const data = await resp.json();
    renderCameraStatus(data.cameras);
    $("#overallProgressBar").value = data.overall_percent;
    $("#overallProgressText").textContent = `${data.overall_percent}%`;

    const allDone = data.cameras.every((c) => c.status === "completed" || c.status === "error");
    if (allDone) {
      stopProgressPolling();
      toast("Processing complete — view results");
      loadResults();
    } else {
      progressPollTimer = setTimeout(pollProgress, 1200);
    }
  } catch (err) {
    stopProgressPolling();
    toast(err.message, true);
  }
}
function startProgressPolling() {
  stopProgressPolling();
  pollProgress();
}
function stopProgressPolling() {
  clearTimeout(progressPollTimer);
  progressPollTimer = null;
}

/* ══════════════════════════════════════════════════════════════════════
   04 — RESULTS
   ══════════════════════════════════════════════════════════════════════ */

let lastMatched = [];

function sightingChips(sightings) {
  return sightings
    .map(
      (s) =>
        `<span class="sighting-chip">${escapeHtml(s.camera_label)} · ${s.first_seen_sec}s–${s.last_seen_sec}s</span>`
    )
    .join("");
}

function renderTopFrames(topFrames) {
  if (!topFrames || !topFrames.length) return "";
  const thumbs = topFrames
    .map(
      (f) =>
        `<div class="evidence-thumb" title="Frame ${f.frame} @ ${f.time_sec}s · face match ${(f.similarity * 100).toFixed(0)}%">
           <img src="${escapeHtml(f.url)}" alt="match frame" loading="lazy"/>
           <span>${(f.similarity * 100).toFixed(0)}%</span>
         </div>`
    )
    .join("");
  return `<div class="evidence-strip">${thumbs}</div>`;
}

function renderMatched(matched) {
  const list = $("#matchedList");
  const empty = $("#matchedEmpty");
  list.innerHTML = "";
  empty.hidden = matched.length > 0;
  matched.forEach((m) => {
    const cams = [...new Set(m.sightings.map((s) => s.camera_label))];
    const row = document.createElement("div");
    row.className = "result-row";
    row.innerHTML = `
      <div class="result-left">
        <div class="person-avatar">${initials(m.name)}</div>
        <div>
          <div class="result-name">${escapeHtml(m.name)}</div>
          <div class="result-sub">Seen on ${cams.length} camera${cams.length === 1 ? "" : "s"} · match confidence ${(m.similarity * 100).toFixed(0)}%</div>
          <div>${sightingChips(m.sightings)}</div>
          ${renderTopFrames(m.top_frames)}
        </div>
      </div>
    `;
    list.appendChild(row);
  });
}

function renderUnmatched(unmatched) {
  const list = $("#unmatchedList");
  const empty = $("#unmatchedEmpty");
  list.innerHTML = "";
  empty.hidden = unmatched.length > 0;
  unmatched.forEach((u) => {
    const row = document.createElement("div");
    row.className = "result-row unmatched";
    row.innerHTML = `
      <div class="result-left">
        <div class="person-avatar" style="background:linear-gradient(135deg,#5a6b80,#2c3646);color:#eaf0f7;">?</div>
        <div>
          <div class="result-name">Unknown person · ${escapeHtml(u.track_id)}</div>
          <div class="result-sub">${sightingChips([u])}</div>
        </div>
      </div>
    `;
    list.appendChild(row);
  });
}

async function loadResults() {
  if (!currentSessionId) return;
  try {
    const resp = await fetch(`/api/session/${currentSessionId}/results`);
    if (!resp.ok) throw new Error("Results not available yet");
    const data = await resp.json();

    $("#statDetected").textContent = data.summary.people_detected;
    $("#statMatched").textContent = data.summary.matched;
    $("#statUnknown").textContent = data.summary.unknown;
    $("#statCameras").textContent = data.summary.camera_count;

    lastMatched = data.matched;
    renderMatched(lastMatched);
    renderUnmatched(data.unmatched);
    loadStorage();
    logEvent(
      "info",
      `Results ready: ${data.summary.matched} matched, ${data.summary.unknown} unknown across ${data.summary.camera_count} camera(s)`
    );
  } catch (err) {
    toast(err.message, true);
  }
}

$("#resultsSearch").addEventListener("input", (e) => {
  const q = e.target.value.trim().toLowerCase();
  const filtered = q ? lastMatched.filter((m) => m.name.toLowerCase().includes(q)) : lastMatched;
  renderMatched(filtered);
});

/* ══════════════════════════════════════════════════════════════════════
   05a — STREAMING LOG TERMINAL
   Colorized, monospaced, capped buffer so the DOM never grows unbounded.
   ══════════════════════════════════════════════════════════════════════ */

const LOG_MAX_LINES = 200;
const logBuffer = [];

function timeTag() {
  return new Date().toLocaleTimeString("en-GB", { hour12: false });
}

/**
 * logEvent(level, message)
 * level: "info" | "warn" | "error" | "critical"
 */
function logEvent(level, message) {
  logBuffer.push({ level, message, t: timeTag() });
  if (logBuffer.length > LOG_MAX_LINES) logBuffer.shift();
  renderLogTail();
}

function logLineHtml(entry) {
  const tagText = {
    info: "INFO",
    warn: "WARN",
    error: "ERROR",
    critical: "CRITICAL RE-ID MATCH",
  }[entry.level];
  const isCritical = entry.level === "critical";
  return `<div class="log-line"><span class="log-time">${entry.t}</span> <span class="log-tag ${entry.level}">[${tagText}]</span> <span class="log-msg${isCritical ? " critical-text" : ""}">${escapeHtml(entry.message)}</span></div>`;
}

// Appends only the newest line by default (cheap), full re-render when clearing.
function renderLogTail() {
  const el = $("#logTerminal");
  if (!el) return;
  const entry = logBuffer[logBuffer.length - 1];
  el.insertAdjacentHTML("beforeend", logLineHtml(entry));
  while (el.children.length > LOG_MAX_LINES) el.removeChild(el.firstChild);
  el.scrollTop = el.scrollHeight;
}

function renderLogFull() {
  const el = $("#logTerminal");
  if (!el) return;
  el.innerHTML = logBuffer.map(logLineHtml).join("");
  el.scrollTop = el.scrollHeight;
}

$("#logClearBtn").addEventListener("click", () => {
  logBuffer.length = 0;
  renderLogFull();
  toast("Log cleared");
});

/* ══════════════════════════════════════════════════════════════════════
   05b — HARDWARE TELEMETRY (GPU VRAM / utilization / drop rate / latency)
   Self-contained mock feed — swap `sampleTelemetry()` for a real metrics
   endpoint (e.g. GET /api/telemetry) when the backend exposes one.
   ══════════════════════════════════════════════════════════════════════ */

const TELEMETRY_TOTAL_VRAM_GB = 12;
const latencyHistory = [];
const LATENCY_POINTS = 48;
let telemetryTimer = null;
let telemetryStarted = false;

function sampleTelemetry() {
  const gpuActive = document.querySelectorAll(".camera-status-row .badge.warn").length > 0;
  const baseVram = gpuActive ? 6.4 : 2.1;
  const vramUsed = clamp(baseVram + (Math.random() - 0.5) * 1.4, 0.6, TELEMETRY_TOTAL_VRAM_GB - 0.2);
  const gpuUtil = clamp((gpuActive ? 62 : 8) + (Math.random() - 0.5) * 30, 2, 99);
  const dropRate = clamp(gpuActive ? Math.random() * 2.2 : Math.random() * 0.4, 0, 8);
  const latency = clamp((gpuActive ? 38 : 14) + (Math.random() - 0.5) * 16, 6, 120);
  return { vramUsed, gpuUtil, dropRate, latency };
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function updateTelemetryDom({ vramUsed, gpuUtil, dropRate, latency }) {
  $("#vramValue").textContent = `${vramUsed.toFixed(1)} / ${TELEMETRY_TOTAL_VRAM_GB} GB`;
  $("#vramGauge").style.width = `${(vramUsed / TELEMETRY_TOTAL_VRAM_GB) * 100}%`;
  $("#vramSub").textContent = `device: ${navigator.gpu ? "WebGPU-capable" : "cuda / cpu fallback"}`;

  $("#gpuUtilValue").textContent = `${gpuUtil.toFixed(0)}%`;
  $("#gpuUtilGauge").style.width = `${gpuUtil}%`;

  $("#dropRateValue").textContent = `${dropRate.toFixed(1)}%`;
  $("#dropRateGauge").style.width = `${Math.min(100, dropRate * 10)}%`;

  $("#latencyValue").textContent = `${latency.toFixed(0)} ms`;

  latencyHistory.push(latency);
  if (latencyHistory.length > LATENCY_POINTS) latencyHistory.shift();
  drawLatencyChart();

  if (dropRate > 5) logEvent("warn", `Elevated stream drop rate: ${dropRate.toFixed(1)}%`);
}

function drawLatencyChart() {
  const canvas = $("#latencyChart");
  if (!canvas || latencyHistory.length < 2) return;
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || 600;
  const cssHeight = 72;
  if (canvas.width !== cssWidth * dpr || canvas.height !== cssHeight * dpr) {
    canvas.width = cssWidth * dpr;
    canvas.height = cssHeight * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const max = Math.max(60, ...latencyHistory);
  const stepX = cssWidth / (LATENCY_POINTS - 1);
  const styles = getComputedStyle(document.documentElement);
  const lineColor = styles.getPropertyValue("--cool").trim() || "#4ee3d1";

  ctx.beginPath();
  latencyHistory.forEach((v, i) => {
    const x = i * stepX;
    const y = cssHeight - (v / max) * (cssHeight - 8) - 4;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineWidth = 2;
  ctx.strokeStyle = lineColor;
  ctx.lineJoin = "round";
  ctx.stroke();

  // filled area under the line for a glanceable "load" feel
  ctx.lineTo((latencyHistory.length - 1) * stepX, cssHeight);
  ctx.lineTo(0, cssHeight);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, cssHeight);
  grad.addColorStop(0, lineColor + "55");
  grad.addColorStop(1, lineColor + "00");
  ctx.fillStyle = grad;
  ctx.fill();
}

function telemetryTick() {
  updateTelemetryDom(sampleTelemetry());
}

function startTelemetry() {
  if (telemetryTimer) return;
  telemetryTick();
  telemetryTimer = setInterval(telemetryTick, 1400);
  if (!telemetryStarted) {
    telemetryStarted = true;
    logEvent("info", "Telemetry stream connected");
  }
}
function stopTelemetry() {
  clearInterval(telemetryTimer);
  telemetryTimer = null;
}

/* ══════════════════════════════════════════════════════════════════════
   05c — LIVE DETECTION PREVIEW (real video + canvas overlay)
   The uploaded camera clip plays in a <video> element; a transparent
   <canvas> sits on top of it and is redrawn every animation frame.

   Coordinate pipeline (this is the part that fixes box misalignment):
     1. Boxes are always expressed in the VIDEO'S OWN intrinsic pixel space
        (0..video.videoWidth, 0..video.videoHeight) — exactly what the
        backend's /camera/{id}/tracks endpoint returns (bbox = [x1,y1,x2,y2]
        from the real detector), and what the mock generator produces too.
     2. getContainRect() computes where the video is actually drawn inside
        the canvas box under `object-fit: contain` — i.e. the letterboxed
        rectangle, not the full canvas.
     3. videoBoxToCanvas() scales a video-space box into that rectangle and
        clamps it so it can never extend past the real video content into
        the letterbox bars.
   Until a camera's real tracks are ready, mock tracks fill in (also in
   video-pixel space) so the preview isn't empty — the mode badge in the
   corner of the stage tells you which one you're looking at.
   ══════════════════════════════════════════════════════════════════════ */

const MOCK_SUBJECTS = [
  { track_id: 101, name: "Subject_Alpha" },
  { track_id: 104, name: "Subject_Beta" },
  { track_id: 118, name: null },
  { track_id: 122, name: "Subject_Ophelia" },
];

const sessionVideoUrls = {}; // camera_id -> object URL, set when a session starts
const realTrackCache = {}; // camera_id -> { fps, frames: [{t, boxes}], fetching }
let overlayRaf = null;
let overlayMockTracks = [];
let overlayAlertTimer = null;
let overlayResizeObserver = null;
let activePreviewCamId = null;

function populatePreviewCameras(cameraIds) {
  const select = $("#previewCameraSelect");
  select.innerHTML = "";
  cameraIds.forEach((camId) => {
    const opt = document.createElement("option");
    opt.value = camId;
    opt.textContent = sessionCameraLabels[camId] || camId;
    select.appendChild(opt);
  });
  $("#previewPanel").hidden = cameraIds.length === 0;
  if (cameraIds.length) loadPreviewCamera(cameraIds[0]);
}

function loadPreviewCamera(camId) {
  const url = sessionVideoUrls[camId];
  const video = $("#previewVideo");
  const stage = video.closest(".overlay-stage");
  if (!url || !video) return;

  activePreviewCamId = camId;
  video.src = url;
  video.muted = true;
  video.playsInline = true;
  stage.classList.remove("has-video");

  video.onloadedmetadata = () => {
    stage.classList.add("has-video");
    resizeOverlayCanvas();
    video.play().catch(() => {
      /* autoplay can be blocked before user interaction — Pause button still works */
    });
  };

  seedMockTracks(video);
  updateOverlayModeBadge();
  // if this camera already finished processing (e.g. switching back to it),
  // real detections may already be cached or fetchable
  if (!realTrackCache[camId]) maybeFetchRealTracks(camId);

  $("#previewPauseToggle").textContent = "⏸ Pause";
}

$("#previewCameraSelect").addEventListener("change", (e) => {
  loadPreviewCamera(e.target.value);
  logEvent("info", `Preview switched to ${sessionCameraLabels[e.target.value] || e.target.value}`);
});

$("#previewPauseToggle").addEventListener("click", () => {
  const video = $("#previewVideo");
  if (video.paused) {
    video.play();
    if (!overlayRaf) startOverlay();
    $("#previewPauseToggle").textContent = "⏸ Pause";
  } else {
    video.pause();
    $("#previewPauseToggle").textContent = "▶ Resume";
  }
});

function resizeOverlayCanvas() {
  const canvas = $("#overlayCanvas");
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w === 0 || h === 0) return; // panel hidden — nothing to size yet
  const targetW = Math.round(w * dpr);
  const targetH = Math.round(h * dpr);
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }
}

/**
 * Fetch real per-frame boxes for a camera once its pipeline has finished
 * (the backend only has data once Re-ID has run — see /camera/{id}/tracks).
 * Safe to call speculatively; 409 just means "not ready yet".
 */
async function maybeFetchRealTracks(camId, _retryCount) {
  if (!currentSessionId || realTrackCache[camId]) return;
  realTrackCache[camId] = { fetching: true, fps: 25, frames: [] };
  const retries = _retryCount || 0;
  try {
    const resp = await fetch(`/api/session/${currentSessionId}/camera/${camId}/tracks`);
    if (resp.status === 409) {
      delete realTrackCache[camId];
      if (retries < 5) {
        setTimeout(() => maybeFetchRealTracks(camId, retries + 1), 2000);
      }
      return;
    }
    if (!resp.ok) throw new Error("Could not load detections for this camera");
    const data = await resp.json();
    realTrackCache[camId] = { fetching: false, fps: data.fps || 25, frames: data.frames || [] };
    if (camId === activePreviewCamId) {
      updateOverlayModeBadge();
      logEvent("info", `${sessionCameraLabels[camId] || camId}: live detections loaded into preview`);
    }
  } catch (err) {
    delete realTrackCache[camId];
    if (retries < 3) {
      setTimeout(() => maybeFetchRealTracks(camId, retries + 1), 2000);
    } else {
      logEvent("warn", `Could not load detections for ${sessionCameraLabels[camId] || camId}: ${err.message}`);
    }
  }
}

function updateOverlayModeBadge() {
  const badge = $("#overlayModeBadge");
  if (!badge) return;
  const cache = realTrackCache[activePreviewCamId];
  const isLive = cache && !cache.fetching && cache.frames.length > 0;
  badge.textContent = isLive ? "● LIVE DETECTIONS" : "◌ DEMO OVERLAY (sample data)";
  badge.classList.toggle("live", !!isLive);
}

// ── mock fallback tracks, expressed in video-pixel space like real ones ──

function seedMockTracks(video) {
  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;
  overlayMockTracks = MOCK_SUBJECTS.map((s, i) => ({
    ...s,
    _vw: vw,
    _vh: vh,
    cx: (0.12 + i * (0.7 / MOCK_SUBJECTS.length) + Math.random() * 0.04) * vw,
    cy: (0.35 + Math.random() * 0.3) * vh,
    vx: (Math.random() - 0.5) * vw * 0.0035,
    vy: (Math.random() - 0.5) * vh * 0.002,
    boxW: (0.1 + Math.random() * 0.05) * vw,
    boxH: (0.45 + Math.random() * 0.15) * vh,
    similarity: s.name ? 0.9 + Math.random() * 0.099 : null,
  }));
}

function stepMockTracks() {
  overlayMockTracks.forEach((t) => {
    t.cx += t.vx;
    t.cy += t.vy;
    if (t.cx - t.boxW / 2 < 0 || t.cx + t.boxW / 2 > t._vw) t.vx *= -1;
    if (t.cy - t.boxH / 2 < 0 || t.cy + t.boxH / 2 > t._vh) t.vy *= -1;
  });
  return overlayMockTracks.map((t) => ({
    track_id: t.track_id,
    name: t.name,
    similarity: t.similarity,
    x1: t.cx - t.boxW / 2,
    y1: t.cy - t.boxH / 2,
    x2: t.cx + t.boxW / 2,
    y2: t.cy + t.boxH / 2,
  }));
}

// binary search for the sampled frame nearest video.currentTime
function findNearestFrame(frames, t, tolerance = 0.35) {
  if (!frames.length) return null;
  let lo = 0,
    hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].t < t) lo = mid + 1;
    else hi = mid;
  }
  let best = frames[lo];
  if (lo > 0 && Math.abs(frames[lo - 1].t - t) < Math.abs(best.t - t)) best = frames[lo - 1];
  return Math.abs(best.t - t) <= tolerance ? best : null;
}

function getActiveBoxes(video) {
  const cache = realTrackCache[activePreviewCamId];
  if (cache && !cache.fetching && cache.frames.length > 0) {
    const frame = findNearestFrame(cache.frames, video.currentTime);
    if (!frame) return [];
    return frame.boxes
      .filter((b) => Array.isArray(b.bbox) && b.bbox.length === 4)
      .map((b) => ({
        track_id: b.track_id,
        name: b.name,
        similarity: b.similarity,
        x1: b.bbox[0],
        y1: b.bbox[1],
        x2: b.bbox[2],
        y2: b.bbox[3],
      }));
  }
  return stepMockTracks();
}

// ── coordinate transform: video-intrinsic pixels -> canvas display pixels,
//    accounting for `object-fit: contain` letterboxing ─────────────────────

function getContainRect(mediaW, mediaH, boxW, boxH) {
  if (!mediaW || !mediaH || !boxW || !boxH) return null;
  const mediaRatio = mediaW / mediaH;
  const boxRatio = boxW / boxH;
  let drawW, drawH;
  if (mediaRatio > boxRatio) {
    drawW = boxW;
    drawH = boxW / mediaRatio;
  } else {
    drawH = boxH;
    drawW = boxH * mediaRatio;
  }
  return {
    offsetX: (boxW - drawW) / 2,
    offsetY: (boxH - drawH) / 2,
    drawW,
    drawH,
    scaleX: drawW / mediaW,
    scaleY: drawH / mediaH,
  };
}

function videoBoxToCanvas(box, rect) {
  let px1 = rect.offsetX + box.x1 * rect.scaleX;
  let py1 = rect.offsetY + box.y1 * rect.scaleY;
  let px2 = rect.offsetX + box.x2 * rect.scaleX;
  let py2 = rect.offsetY + box.y2 * rect.scaleY;

  // clamp: a box can never extend past the actual video content into the
  // letterbox bars, no matter what the source coordinates say
  const minX = rect.offsetX,
    maxX = rect.offsetX + rect.drawW;
  const minY = rect.offsetY,
    maxY = rect.offsetY + rect.drawH;
  px1 = Math.min(Math.max(px1, minX), maxX);
  px2 = Math.min(Math.max(px2, minX), maxX);
  py1 = Math.min(Math.max(py1, minY), maxY);
  py2 = Math.min(Math.max(py2, minY), maxY);

  return { x: px1, y: py1, w: Math.max(0, px2 - px1), h: Math.max(0, py2 - py1) };
}

// ── rendering: RED = database match (alert / target), GREEN = unidentified ──

function drawBoundingBox(ctx, px, isMatch, pulse) {
  const color = isMatch ? "#ff3b5c" : "#3ddc7a";
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = isMatch ? 2 + pulse * 1.4 : 2;
  ctx.shadowColor = color;
  ctx.shadowBlur = isMatch ? 10 + pulse * 10 : 6;
  ctx.strokeRect(px.x, px.y, px.w, px.h);

  // corner ticks for a tactical-HUD feel
  const c = 10;
  ctx.lineWidth = 3;
  ctx.shadowBlur = 0;
  [
    [px.x, px.y, c, 0, 0, c],
    [px.x + px.w, px.y, -c, 0, 0, c],
    [px.x, px.y + px.h, c, 0, 0, -c],
    [px.x + px.w, px.y + px.h, -c, 0, 0, -c],
  ].forEach(([x, y, dx1, dy1, dx2, dy2]) => {
    ctx.beginPath();
    ctx.moveTo(x + dx1, y + dy1);
    ctx.lineTo(x, y);
    ctx.lineTo(x + dx2, y + dy2);
    ctx.stroke();
  });
  ctx.restore();
}

function drawPlacard(ctx, px, box, isMatch, pulse, rect) {
  const label = isMatch
    ? `⚠ TARGET — ${box.name} · ID ${box.track_id}${
        box.similarity != null ? ` [Match: ${(box.similarity * 100).toFixed(1)}%]` : ""
      }`
    : `ID ${box.track_id} — Unidentified`;
  const color = isMatch ? "#ff3b5c" : "#3ddc7a";

  ctx.save();
  ctx.font = "700 12px 'JetBrains Mono', monospace";
  const paddingX = 8;
  const textWidth = ctx.measureText(label).width;
  const boxW = textWidth + paddingX * 2;
  const boxH = 20;
  // keep the placard within the actual video content area, same rule as the box itself
  const maxX = rect.offsetX + rect.drawW - boxW - 2;
  const bx = Math.max(rect.offsetX + 2, Math.min(px.x, maxX));
  const by = Math.max(rect.offsetY + 2, px.y - boxH - 4);

  ctx.globalAlpha = isMatch ? 0.75 + pulse * 0.25 : 0.82;
  ctx.fillStyle = isMatch ? "rgba(255, 20, 60, 0.92)" : "rgba(10, 16, 24, 0.85)";
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(bx, by, boxW, boxH, 5) : ctx.rect(bx, by, boxW, boxH);
  ctx.fill();
  ctx.stroke();

  ctx.globalAlpha = 1;
  ctx.fillStyle = isMatch ? "#fff" : "#eaf0f7";
  ctx.textBaseline = "middle";
  ctx.fillText(label, bx + paddingX, by + boxH / 2 + 1);
  ctx.restore();
}

function renderOverlayFrame(ctx, canvas, video) {
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.width / dpr;
  const ch = canvas.height / dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);

  const vw = video.videoWidth,
    vh = video.videoHeight;
  const rect = getContainRect(vw, vh, cw, ch);
  if (!rect) return;

  const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 220);
  const boxes = getActiveBoxes(video);

  boxes.forEach((b) => {
    const px = videoBoxToCanvas(b, rect);
    if (px.w <= 0 || px.h <= 0) return; // fully clamped out of the visible frame
    const isMatch = !!b.name;
    drawBoundingBox(ctx, px, isMatch, pulse);
    drawPlacard(ctx, px, b, isMatch, pulse, rect);
  });
}

function overlayLoop() {
  const canvas = $("#overlayCanvas");
  const video = $("#previewVideo");
  if (!canvas || !video) return;
  if (video.paused || video.ended) {
    overlayRaf = null;
    return;
  }
  const ctx = canvas.getContext("2d");
  renderOverlayFrame(ctx, canvas, video);
  overlayRaf = requestAnimationFrame(overlayLoop);
}

function startOverlay() {
  if (overlayRaf) return;
  resizeOverlayCanvas();
  overlayRaf = requestAnimationFrame(overlayLoop);

  if (!overlayAlertTimer) {
    // periodically log a critical alert line when the active camera has a
    // real, high-confidence database match on screen right now
    overlayAlertTimer = setInterval(() => {
      const cache = realTrackCache[activePreviewCamId];
      if (!cache || cache.fetching || !cache.frames.length) return;
      const video = $("#previewVideo");
      const frame = findNearestFrame(cache.frames, video.currentTime);
      const hit = frame && frame.boxes.find((b) => b.name && b.similarity >= 0.75);
      if (hit) {
        logEvent(
          "critical",
          `${hit.name} re-identified with ${(hit.similarity * 100).toFixed(1)}% confidence (ID ${hit.track_id})`
        );
      }
    }, 4000);
  }
}

function stopOverlay() {
  if (overlayRaf) cancelAnimationFrame(overlayRaf);
  overlayRaf = null;
  clearInterval(overlayAlertTimer);
  overlayAlertTimer = null;
}

function pausePreviewPlayback() {
  const video = $("#previewVideo");
  if (video && !video.paused) video.pause();
  stopOverlay();
}

function resumePreviewPlayback() {
  const video = $("#previewVideo");
  if (!video || !video.src) return;
  resizeOverlayCanvas();
  video.play().catch(() => {});
  if (!video.paused) startOverlay();
}

// keep the overlay canvas pixel-perfect if the panel or window changes size
if (window.ResizeObserver) {
  overlayResizeObserver = new ResizeObserver(() => resizeOverlayCanvas());
  overlayResizeObserver.observe($("#overlayCanvas").closest(".overlay-stage"));
}
window.addEventListener("resize", resizeOverlayCanvas);

$("#previewVideo").addEventListener("play", () => startOverlay());
$("#previewVideo").addEventListener("pause", () => stopOverlay());

/* ══════════════════════════════════════════════════════════════════════
   05d — DATABASE IMAGE GALLERY + LIGHTBOX
   ══════════════════════════════════════════════════════════════════════ */

let lightboxPhotos = [];
let lightboxIndex = 0;
let lightboxName = "";

function openLightbox(name, photos, startIndex) {
  if (!photos || !photos.length) return;
  lightboxPhotos = photos;
  lightboxIndex = startIndex || 0;
  lightboxName = name;
  renderLightbox();
  $("#lightboxModal").hidden = false;
}

function renderLightbox() {
  $("#lightboxTitle").textContent = lightboxName;
  $("#lightboxImage").src = lightboxPhotos[lightboxIndex];
  $("#lightboxImage").alt = `${lightboxName} — photo ${lightboxIndex + 1}`;
  $("#lightboxCounter").textContent = `${lightboxIndex + 1} of ${lightboxPhotos.length}`;
  const multi = lightboxPhotos.length > 1;
  $("#lightboxPrev").hidden = !multi;
  $("#lightboxNext").hidden = !multi;
}

function closeLightbox() {
  $("#lightboxModal").hidden = true;
}

$("#lightboxClose").addEventListener("click", closeLightbox);
$("#lightboxModal").addEventListener("click", (e) => {
  if (e.target.id === "lightboxModal") closeLightbox();
});
$("#lightboxPrev").addEventListener("click", () => {
  lightboxIndex = (lightboxIndex - 1 + lightboxPhotos.length) % lightboxPhotos.length;
  renderLightbox();
});
$("#lightboxNext").addEventListener("click", () => {
  lightboxIndex = (lightboxIndex + 1) % lightboxPhotos.length;
  renderLightbox();
});
document.addEventListener("keydown", (e) => {
  if ($("#lightboxModal").hidden) return;
  if (e.key === "Escape") closeLightbox();
  if (e.key === "ArrowLeft") $("#lightboxPrev").click();
  if (e.key === "ArrowRight") $("#lightboxNext").click();
});

async function restoreSession() {
  if (currentSessionId) return;
  try {
    const resp = await fetch("/api/session/current");
    if (!resp.ok) return;
    const data = await resp.json();
    if (!data.session_id) return;
    currentSessionId = data.session_id;
    sessionCameraLabels = {};
    const camIds = [];
    data.cameras.forEach((c) => {
      sessionCameraLabels[c.camera_id] = c.label;
      camIds.push(c.camera_id);
    });
    populatePreviewCameras(camIds);
    logEvent("info", `Re-attached to session ${currentSessionId} after refresh`);
    const prog = await fetch(`/api/session/${currentSessionId}/progress`).then((r) => r.json());
    const allDone = prog.cameras.every((c) => c.status === "completed" || c.status === "error");
    if (allDone) {
      goToTab("results");
      loadResults();
    } else {
      goToTab("process");
      startProgressPolling();
    }
  } catch (err) {
    logEvent("warn", `Could not restore session: ${err.message}`);
  }
}

/* ══════════════════════════════════════════════════════════════════════
   init
   ══════════════════════════════════════════════════════════════════════ */

loadPersons();
restoreSession();
logEvent("info", "Console initialized — awaiting camera input");
