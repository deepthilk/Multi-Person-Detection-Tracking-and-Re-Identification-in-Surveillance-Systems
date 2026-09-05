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
  if (name === "alerts") loadAlerts();
  if (name === "results" && currentSessionId) loadResults();
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
const frameStrideInput = $("#frameStride");

function clampStride() {
  frameStrideInput.value = Math.max(1, Math.min(10, parseInt(frameStrideInput.value || "3", 10)));
}
frameStrideInput.addEventListener("change", clampStride);
$("#strideMinus").addEventListener("click", () => {
  frameStrideInput.value = Math.max(1, parseInt(frameStrideInput.value || "3", 10) - 1);
  clampStride();
});
$("#stridePlus").addEventListener("click", () => {
  frameStrideInput.value = Math.min(10, parseInt(frameStrideInput.value || "3", 10) + 1);
  clampStride();
});

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
  form.append("stride", frameStrideInput.value || "3");

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
    Object.keys(realTrackCache).forEach((k) => delete realTrackCache[k]);
    cameraStatusSeenClear();
    sessionComplete = false;
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
  form.append("person_id", $("#personId").value.trim() || "");
  form.append("flag", $("#personFlag").value || "normal");
  form.append("details", $("#personDetails").value.trim() || "");

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

// Person avatar: the track thumbnail when we have one, otherwise initials/?
function avatarHtml(hasThumb, fallbackText) {
  const thumb = hasThumb && (hasThumb.thumb_url || hasThumb.sightings?.[0]?.thumb_url);
  if (thumb) return `<img class="person-avatar" src="${thumb}" alt="" loading="lazy" />`;
  return `<div class="person-avatar">${fallbackText}</div>`;
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
        <div class="person-name">${escapeHtml(p.name)} ${flagBadge(p.flag)}</div>
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

// ── watch-list flag badges ────────────────────────────────────────────────
const FLAG_LABEL = { criminal: "CRIMINAL", missing: "MISSING", person_of_interest: "PERSON OF INTEREST", normal: "" };
function flagBadge(flag) {
  if (!flag || flag === "normal") return "";
  return `<span class="flag flag-${escapeHtml(flag)}">${escapeHtml(FLAG_LABEL[flag] || flag)}</span>`;
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
    window.lastPersons = persons;

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
let deleteTarget = null;
function openDeleteModal(name) {
  deleteTarget = name;
  $("#confirmBody").textContent = `This removes "${name}"'s photos and embeddings from the identity database. This can't be undone.`;
  $("#confirmModal").hidden = false;
}
$("#confirmCancel").addEventListener("click", () => ($("#confirmModal").hidden = true));
$("#confirmDelete").addEventListener("click", async () => {
  if (!deleteTarget) return;
  try {
    const resp = await fetch(`/api/persons/${encodeURIComponent(deleteTarget)}`, { method: "DELETE" });
    if (!resp.ok) throw new Error("Delete failed");
    toast(`${deleteTarget} removed`);
    logEvent("warn", `Removed "${deleteTarget}" from the identity database`);
    loadPersons($("#personSearch").value.trim());
  } catch (err) {
    toast(err.message, true);
  } finally {
    $("#confirmModal").hidden = true;
    deleteTarget = null;
  }
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
  const rec = (window.lastPersons || []).find((r) => r.name === name);
  $("#editId").value = rec?.person_id || "";
  $("#editFlag").value = rec?.flag || "normal";
  $("#editDetails").value = rec?.details || "";
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

    const profForm = new FormData();
    profForm.append("person_id", $("#editId").value.trim() || "");
    profForm.append("flag", $("#editFlag").value || "normal");
    profForm.append("details", $("#editDetails").value.trim() || "");
    const profResp = await fetch(`/api/persons/${encodeURIComponent(editTarget)}/profile`, {
      method: "PUT",
      body: profForm,
    });
    if (!profResp.ok) throw new Error("Could not save watch-list profile");

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
function cameraStatusSeenClear() {
  Object.keys(cameraStatusSeen).forEach((k) => delete cameraStatusSeen[k]);
}
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
        // NOTE: overlay tracks are intentionally NOT fetched here — boxes
        // stay off until the whole session (incl. cross-camera unify) is done
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
      if (!data.unified) {
        // cross-camera unification + re-render is still running — keep polling
        progressPollTimer = setTimeout(pollProgress, 1200);
        return;
      }
      stopProgressPolling();
      sessionComplete = true;
      // unification rewrote the reid.jsons with global_ids — refetch every
      // camera's overlay so on-screen labels show the global IDs
      Object.keys(realTrackCache).forEach((camId) => {
        delete realTrackCache[camId];
        maybeFetchRealTracks(camId);
      });
      playCompletionChime();
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

// ── completion chime: short two-note beep when the whole session finishes ─
let audioCtx = null;
function playCompletionChime() {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") audioCtx.resume();
    [880, 1174.66].forEach((freq, i) => {
      const t0 = audioCtx.currentTime + i * 0.18;
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sine";
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, t0);
      gain.gain.exponentialRampToValueAtTime(0.22, t0 + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.4);
      osc.connect(gain).connect(audioCtx.destination);
      osc.start(t0);
      osc.stop(t0 + 0.45);
    });
    logEvent("info", "Completion chime played");
  } catch (e) {
    /* audio unavailable — silent fallback */
  }
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

// ── correction helpers: persist a name onto a track, learn it in the DB ──

async function assignTrackName(cameraId, trackId, name) {
  const body = new FormData();
  body.append("camera_id", cameraId);
  body.append("track_id", String(trackId));
  body.append("name", name || "");
  const resp = await fetch(`/api/session/${currentSessionId}/correct`, { method: "POST", body });
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      detail = (await resp.json()).detail;
    } catch (e) {
      /* keep statusText */
    }
    throw new Error(detail);
  }
  return resp.json();
}

// After a name correction the backend re-renders and updates names, but the
// browser's in-memory overlay cache still holds the OLD per-frame data — drop
// it and refetch so the live box label shows the corrected name immediately.
async function refreshOverlayAfterCorrection() {
  for (const camId of Object.keys(realTrackCache)) delete realTrackCache[camId];
  if (activePreviewCamId) await maybeFetchRealTracks(activePreviewCamId);
}

async function renamePerson(oldName, newName, sightings) {
  // a person has at most one track per camera — correct each camera once
  const perCam = {};
  sightings.forEach((s) => {
    if (!(s.camera_id in perCam)) perCam[s.camera_id] = s;
  });
  for (const s of Object.values(perCam)) {
    await assignTrackName(s.camera_id, s.track_id, newName);
  }
}

function renderCameraOutputs(cameras) {
  const box = $("#cameraOutputs");
  if (!box) return;
  const empty = $("#cameraOutputsEmpty");
  box.innerHTML = "";
  const withVideo = cameras.filter((c) => c.output_url);
  empty.hidden = withVideo.length > 0;
  withVideo.forEach((c) => {
    const item = document.createElement("div");
    item.className = "camera-output";
    item.innerHTML = `
      <div class="camera-output-head">
        <strong>${escapeHtml(c.label)}</strong>
        <span class="result-sub">${c.people
          .map((p) => escapeHtml(p.name || `unknown #${p.track_id}`))
          .join(" · ")}</span>
      </div>
      <div class="output-stage">
        <video controls preload="metadata" src="${c.output_url}"></video>
        <div class="video-fallback">
          <p>This camera's annotated video can't be played by the browser.</p>
          <span>The render needs ffmpeg for browser-compatible encoding — run <code>!apt install -y ffmpeg</code> once in Colab, then re-run the session.</span>
        </div>
      </div>
    `;
    const video = item.querySelector("video");
    video.addEventListener("error", () => item.classList.add("video-error"));
    box.appendChild(item);
  });
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
        ${avatarHtml(m, initials(m.name))}
        <div>
          <div class="result-name">${escapeHtml(m.name)}</div>
          <div class="result-sub">Seen on ${cams.length} camera${cams.length === 1 ? "" : "s"} · match confidence ${(m.similarity * 100).toFixed(0)}%</div>
          <div>${sightingChips(m.sightings)}</div>
        </div>
      </div>
      <button class="btn ghost small" type="button">✎ Rename</button>
    `;
    const btn = row.querySelector("button");
    btn.addEventListener("click", () => {
      row.innerHTML = `
        <div class="result-left" style="align-items:flex-start;flex:1;">
          <div style="width:100%;">
            <div class="result-name">Rename “${escapeHtml(m.name)}”</div>
            <div class="result-sub">Applies to ${m.sightings.length} sighting(s). A new name auto-registers this person for future videos; blank clears to unknown.</div>
            <input class="correction-input" value="${escapeHtml(m.name)}" autocomplete="off" />
            <div style="margin-top:.6rem;display:flex;gap:.5rem;">
              <button class="btn primary small" data-role="save" type="button">Save</button>
              <button class="btn ghost small" data-role="cancel" type="button">Cancel</button>
            </div>
          </div>
        </div>
      `;
      row.querySelector('[data-role="save"]').addEventListener("click", async () => {
        const newName = row.querySelector(".correction-input").value.trim();
        try {
          await renamePerson(m.name, newName, m.sightings);
          toast(newName ? `Renamed to “${newName}” and learned for future videos` : "Cleared name");
          await refreshOverlayAfterCorrection();
          loadResults();
        } catch (err) {
          toast(err.message, true);
        }
      });
      row.querySelector('[data-role="cancel"]').addEventListener("click", loadResults);
    });
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
    const tentPct = u.tentative_similarity != null ? Math.round(u.tentative_similarity * 100) : null;
    const tentHint = u.tentative_name
      ? `<div class="result-sub tentative-hint">may be ${escapeHtml(u.tentative_name)}${tentPct != null ? ` · ${tentPct}%` : ""} (below match threshold)</div>`
      : "";
    row.innerHTML = `
      <div class="result-left">
        ${avatarHtml(u, "?")}
        <div style="flex:1;">
          <div class="result-name">Unknown person · ${escapeHtml(u.track_id)}</div>
          <div class="result-sub">${sightingChips([u])}</div>
          ${tentHint}
          <div style="margin-top:.6rem;display:flex;gap:.5rem;">
            <input class="correction-input" placeholder="Assign name… (new names auto-register)" autocomplete="off" />
            <button class="btn primary small" type="button">Save</button>
          </div>
        </div>
      </div>
    `;
    const saveBtn = row.querySelector("button");
    const input = row.querySelector(".correction-input");
    const doSave = async () => {
      const name = input.value.trim();
      try {
        await assignTrackName(u.camera_id, u.track_id, name);
        toast(name ? `${u.track_id} assigned to “${name}”` : "Cleared to unknown");
        await refreshOverlayAfterCorrection();
        loadResults();
      } catch (err) {
        toast(err.message, true);
      }
    };
    saveBtn.addEventListener("click", doSave);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") doSave();
    });
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
    renderCameraOutputs(data.cameras || []);
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
   Until a camera's real tracks are ready, the preview shows the plain video
   (no fake boxes) — the badge tells you whether real detections are loaded.
   ══════════════════════════════════════════════════════════════════════ */

const sessionVideoUrls = {}; // camera_id -> object URL, set when a session starts
let sessionComplete = false; // boxes are drawn only AFTER the whole session (incl. cross-camera unify) finishes
const realTrackCache = {}; // camera_id -> { fps, frames: [{t, boxes}], fetching }
let overlayRaf = null;
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
      /* autoplay can be blocked before user interaction */
    });
  };

  updateOverlayModeBadge();
  // if this camera already finished processing (e.g. switching back to it),
  // real detections may already be cached or fetchable
  if (!realTrackCache[camId]) maybeFetchRealTracks(camId);
}

$("#previewCameraSelect").addEventListener("change", (e) => {
  loadPreviewCamera(e.target.value);
  logEvent("info", `Preview switched to ${sessionCameraLabels[e.target.value] || e.target.value}`);
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
async function maybeFetchRealTracks(camId) {
  if (!currentSessionId || realTrackCache[camId]) return;
  realTrackCache[camId] = { fetching: true, fps: 25, frames: [] };
  try {
    const resp = await fetch(`/api/session/${currentSessionId}/camera/${camId}/tracks`);
    if (resp.status === 409) {
      // not ready yet — poll again while the session is still active; the
      // retry chain self-terminates once the camera completes and returns 200
      delete realTrackCache[camId];
      setTimeout(() => {
        if (currentSessionId) maybeFetchRealTracks(camId);
      }, 3000);
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
    logEvent("warn", `Could not load detections for ${sessionCameraLabels[camId] || camId}: ${err.message}`);
  }
}

function updateOverlayModeBadge() {
  const badge = $("#overlayModeBadge");
  if (!badge) return;
  const cache = realTrackCache[activePreviewCamId];
  const isLive = cache && !cache.fetching && cache.frames.length > 0;
  badge.textContent = isLive ? "● LIVE DETECTIONS" : "AWAITING PROCESSING";
  badge.classList.toggle("live", !!isLive);
}

// Binary search for the two sampled frames bracketing video time `t`.
// Returns { f0, f1, alpha, gapSec } where f0.t <= t <= f1.t, alpha in [0,1]
// is how far t is between them, and gapSec is the wall-clock distance
// between the brackets. Never returns null while frames exist, so boxes
// never vanish due to sampling.
function findFrameWindow(frames, t) {
  if (!frames.length) return null;
  if (t <= frames[0].t) return { f0: frames[0], f1: null, alpha: 0, gapSec: 0 };
  const last = frames[frames.length - 1];
  if (t >= last.t) return { f0: last, f1: null, alpha: 0, gapSec: 0 };
  let lo = 0,
    hi = frames.length - 1;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].t < t) lo = mid;
    else hi = mid;
  }
  const f0 = frames[lo],
    f1 = frames[hi];
  const span = Math.max(f1.t - f0.t, 1e-6);
  return { f0, f1, alpha: Math.min(1, Math.max(0, (t - f0.t) / span)), gapSec: f1.t - f0.t };
}

// Gaps shorter than this glide the box smoothly (brief occlusion); longer
// gaps hold the box at its last known position instead of sweeping across
// empty space (a person leaving and reappearing far away).
const MAX_GLIDE_GAP_SEC = 1.0;

// Build a per-box interpolation map from one frame's boxes.
function boxForIndex(frames, idx) {
  const boxes = frames[idx] && frames[idx].boxes;
  const out = {};
  if (!boxes) return out;
  for (const b of boxes) {
    if (Array.isArray(b.bbox) && b.bbox.length === 4) out[b.track_id] = b;
  }
  return out;
}

// Linearly interpolate every box between the bracketing sampled frames so
// the box tracks the person continuously (no lag, no snap) and stays
// visible even when the underlying sampling is coarse.
function getActiveBoxes(video) {
  // while the pipeline is still running the preview shows ONLY the raw
  // footage — boxes appear once the full session (incl. unify) is complete
  if (!sessionComplete) return [];
  const cache = realTrackCache[activePreviewCamId];
  if (!cache || cache.fetching || cache.frames.length === 0) return [];
  const win = findFrameWindow(cache.frames, video.currentTime);
  if (!win) return [];

  const glide = win.gapSec <= MAX_GLIDE_GAP_SEC && win.f1;
  const a = boxForIndex(cache.frames, cache.frames.indexOf(win.f0));
  let b = {};
  if (glide) b = boxForIndex(cache.frames, cache.frames.indexOf(win.f1));
  const alpha = glide ? win.alpha : 0;
  const allIds = new Set([...Object.keys(a), ...Object.keys(b)]);
  const result = [];
  for (const id of allIds) {
    const ba = a[id],
      bb = b[id];
    const src = ba || bb;
    // glide smoothly on short gaps; hold the last known position on long ones
    result.push(interpBox(src, bb && glide ? bb : src, alpha, ba, bb, glide));
  }
  return result;
}

function interpBox(a, b, alpha, hasA, hasB, glide) {
  const lerp = (p, q) => p + (q - p) * alpha;
  // For a long gap (no glide), keep the box at the most recent known place
  // instead of interpolating toward a far-away reappearance.
  const effective = glide ? b : (a || b);
  return {
    track_id: a.track_id,
    name: a.name,
    similarity: a.similarity,
    global_id: a.global_id,
    tentative_name: a.tentative_name ?? b.tentative_name,
    x1: lerp(a.bbox[0], effective.bbox[0]),
    y1: lerp(a.bbox[1], effective.bbox[1]),
    x2: lerp(a.bbox[2], effective.bbox[2]),
    y2: lerp(a.bbox[3], effective.bbox[3]),
  };
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

// ── rendering: solid BLACK box with a BLACK label placard. RED text = known
//    person (name / ID), GREEN text = unknown. No blink pulse — a box stays
//    steadily visible so identities can be read at a glance.

const BOX_COLOR = "#000000"; // black box (both known & unknown)
const TEXT_KNOWN_COLOR = "#ff0000"; // red — known person
const TEXT_UNKNOWN_COLOR = "#00ff00"; // green — unknown person

function drawBoundingBox(ctx, px, isMatch) {
  ctx.save();
  // thin light outline underneath so the black box pops on dark backgrounds
  ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
  ctx.lineWidth = 2;
  ctx.strokeRect(px.x, px.y, px.w, px.h);
  ctx.strokeStyle = BOX_COLOR;
  ctx.lineWidth = 4;
  ctx.strokeRect(px.x, px.y, px.w, px.h);
  ctx.restore();
}

function drawPlacard(ctx, px, box, isMatch, rect) {
  const textColor = isMatch ? TEXT_KNOWN_COLOR : TEXT_UNKNOWN_COLOR;
  const idLabel = box.global_id != null ? `GID ${box.global_id}` : `ID ${box.track_id}`;
  const tentPct = box.tentative_similarity != null ? Math.round(box.tentative_similarity * 100) : null;
  const label = isMatch
    ? `${box.name} · ${idLabel}${
        box.similarity != null ? ` [Match: ${(box.similarity * 100).toFixed(1)}%]` : ""
      }`
    : box.tentative_name
    ? `may be ${box.tentative_name}${tentPct != null ? ` (${tentPct}%)` : ""} · ${idLabel}`
    : `${box.global_id != null ? `Global ID ${box.global_id}` : `ID ${box.track_id}`} — Unidentified`;

  ctx.save();
  ctx.font = "700 13px 'JetBrains Mono', monospace";
  const paddingX = 8;
  const textWidth = ctx.measureText(label).width;
  const boxW = textWidth + paddingX * 2;
  const boxH = 22;
  // keep the placard within the actual video content area, same rule as the box itself
  const maxX = rect.offsetX + rect.drawW - boxW - 2;
  const bx = Math.max(rect.offsetX + 2, Math.min(px.x, maxX));
  const by = Math.max(rect.offsetY + 2, px.y - boxH - 4);

  ctx.globalAlpha = 1;
  ctx.fillStyle = BOX_COLOR; // black label background
  ctx.strokeStyle = textColor;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(bx, by, boxW, boxH, 5) : ctx.rect(bx, by, boxW, boxH);
  ctx.fill();
  ctx.stroke();

  ctx.globalAlpha = 1;
  ctx.fillStyle = textColor;
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

  const boxes = getActiveBoxes(video);

  boxes.forEach((b) => {
    const px = videoBoxToCanvas(b, rect);
    if (px.w <= 0 || px.h <= 0) return; // fully clamped out of the visible frame
    const isMatch = !!b.name;
    drawBoundingBox(ctx, px, isMatch);
    drawPlacard(ctx, px, b, isMatch, rect);
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
      const win = findFrameWindow(cache.frames, video.currentTime);
      let hit = null;
      if (win) {
        const check = (f) =>
          f && f.boxes.find((b) => b.name && b.similarity >= 0.75);
        hit = check(win.f0) || (win.f1 && check(win.f1));
      }
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

// ── pause / resume control for the live preview ──────────────────────────
function syncPreviewPauseLabel() {
  const v = $("#previewVideo");
  const btn = $("#previewPauseBtn");
  if (!v || !btn) return;
  btn.textContent = v.paused ? "▶ Play" : "⏸ Pause";
}
$("#previewPauseBtn").addEventListener("click", () => {
  const v = $("#previewVideo");
  if (!v || !v.src) return;
  if (v.paused) v.play().catch(() => {});
  else v.pause();
  logEvent("info", v.paused ? "Preview paused" : "Preview resumed");
});

$("#previewVideo").addEventListener("play", () => { startOverlay(); syncPreviewPauseLabel(); });
$("#previewVideo").addEventListener("pause", () => { stopOverlay(); syncPreviewPauseLabel(); });
$("#previewVideo").addEventListener("loadedmetadata", syncPreviewPauseLabel);

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

/* ══════════════════════════════════════════════════════════════════════
   watch-list alerts feed
   ══════════════════════════════════════════════════════════════════════ */

async function loadAlerts() {
  const list = $("#alertsList");
  const empty = $("#alertsEmpty");
  const count = $("#alertsCount");
  try {
    const resp = await fetch("/api/alerts");
    if (!resp.ok) throw new Error("Failed to load alerts");
    const { alerts } = await resp.json();
    const active = (alerts || []).filter((a) => a.flag && a.flag !== "normal");
    updateAlertsBadge(active.length);
    if (count) count.textContent = active.length
      ? `${active.length} active watch-list alert${active.length === 1 ? "" : "s"}`
      : "";

    list.innerHTML = "";
    empty.hidden = active.length > 0;
    active.forEach((a) => list.appendChild(alertCard(a)));
  } catch (err) {
    toast(err.message, true);
  }
}

function updateAlertsBadge(n) {
  const badge = $("#alertsBadge");
  if (!badge) return;
  badge.hidden = !n;
  badge.textContent = n;
}

function alertCard(a) {
  const el = document.createElement("div");
  const cls = ["alert-card"];
  if (a.flag === "criminal") cls.push("critical");
  el.className = cls.join(" ");
  const flagTxt = FLAG_LABEL[a.flag] || a.flag;
  const when = a.time ? new Date(a.time.replace(" ", "T")).toLocaleString() : "—";
  const sim = a.similarity != null ? ` · ${Math.round(a.similarity * 100)}% match` : "";
  const where = [a.source, a.camera_id].filter(Boolean).join(" · ");
  const track = a.track_id != null ? `track #${a.track_id}` : "";
  const crop = a.crop_url
    ? `<img class="alert-crop" src="${a.crop_url}" alt="" loading="lazy" />`
    : `<div class="alert-crop empty">no&nbsp;photo</div>`;

  el.innerHTML = `
    <div class="alert-head">
      ${crop}
      <div class="alert-body">
        <div class="alert-title"><b>${escapeHtml(a.person)}</b> ${flagBadge(a.flag)}</div>
        <div class="alert-flagtext">${escapeHtml(flagTxt)}</div>
        ${a.details ? `<div class="alert-details">${escapeHtml(a.details)}</div>` : ""}
        <div class="alert-meta">${escapeHtml(where)}${track ? " · " + track : ""}</div>
        <div class="alert-meta">${escapeHtml(when)}${sim}</div>
      </div>
      <button class="btn danger small alert-dismiss" data-dismiss="${escapeHtml(a.id)}">Dismiss</button>
    </div>
  `;
  // click the alert's crop to open a full-size preview in the lightbox
  const cropImg = el.querySelector("img.alert-crop");
  if (cropImg) {
    cropImg.style.cursor = "zoom-in";
    cropImg.addEventListener("click", () =>
      openLightbox(`${a.person} — alert crop`, [a.crop_url], 0)
    );
  }
  return el;
}

$("#alertsList").addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-dismiss]");
  if (!btn) return;
  btn.disabled = true;
  try {
    const resp = await fetch(`/api/alerts/${encodeURIComponent(btn.dataset.dismiss)}`, { method: "DELETE" });
    if (!resp.ok) throw new Error("Could not dismiss alert");
    loadAlerts();
  } catch (err) {
    btn.disabled = false;
    toast(err.message, true);
  }
});

$("#alertsRefresh").addEventListener("click", loadAlerts);
// keep the badge fresh even while the operator is on another tab
setInterval(() => {
  fetch("/api/alerts")
    .then((r) => r.json())
    .then(({ alerts }) => updateAlertsBadge((alerts || []).filter((a) => a.flag && a.flag !== "normal").length))
    .catch(() => {});
}, 15000);

/* ══════════════════════════════════════════════════════════════════════
   init
   ══════════════════════════════════════════════════════════════════════ */

loadPersons();
loadAlerts();
logEvent("info", "Console initialized — awaiting camera input");
