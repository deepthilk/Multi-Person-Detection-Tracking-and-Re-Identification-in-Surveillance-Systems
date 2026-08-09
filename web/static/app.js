/* Surveillance Re-ID Dashboard — frontend logic */
(() => {
  "use strict";

  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];

  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[c]));

  const api = async (path, opts = {}) => {
    const res = await fetch(path, opts);
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try {
        const j = await res.json();
        if (j && j.detail) detail = j.detail;
      } catch (_) {}
      throw new Error(detail);
    }
    return res.json();
  };

  const state = { syncMode: false, leader: null, alerts: [], activeJobs: 0 };

  const updateProcessingBanner = () => {
    const busy = state.activeJobs > 0;
    $$(".processing-banner").forEach((b) => (b.hidden = !busy));
  };
  const bumpJobs = (delta) => {
    state.activeJobs = Math.max(0, state.activeJobs + delta);
    updateProcessingBanner();
  };

  const FLAG_LABEL = { criminal: "CRIMINAL", missing: "MISSING", person_of_interest: "PERSON OF INTEREST", normal: "" };

  const flagBadge = (flag) => {
    if (!flag || flag === "normal") return "";
    return `<span class="flag flag-${esc(flag)}">${esc(FLAG_LABEL[flag] || flag)}</span>`;
  };

  // ── navigation ────────────────────────────────────────────────────────────
  const loaders = {};
  const show = (name) => {
    $$(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.page === name));
    $$(".page").forEach((p) => {
      const on = p.dataset.page === name;
      p.classList.toggle("active", on);
      p.classList.toggle("hidden", !on);
    });
    document.querySelector(".content").scrollTop = 0;
    if (loaders[name]) loaders[name]();
  };
  $$(".nav-btn").forEach((b) => b.addEventListener("click", () => show(b.dataset.page)));
  $$("[data-goto]").forEach((b) => b.addEventListener("click", () => show(b.dataset.goto)));

  const setSys = (ok, text) => {
    const dot = $(".sys-dot");
    dot.classList.toggle("ok", ok);
    dot.classList.toggle("err", !ok);
    $("#sysStatusText").textContent = text;
  };

  // ── job console ───────────────────────────────────────────────────────────
  const jobConsole = (boxEl, jobId, { onDone, onError, onComplete } = {}) => {
    boxEl.classList.remove("hidden");
    boxEl.innerHTML = `
      <div class="job-head">
        <span class="job-status running">RUNNING</span>
        <span class="job-msg">Starting…</span>
      </div>
      <progress max="100" value="0"></progress>
      <pre class="job-console"></pre>`;
    const statusEl = $(".job-status", boxEl);
    const msgEl = $(".job-msg", boxEl);
    const prog = $("progress", boxEl);
    const con = $(".job-console", boxEl);
    let timer = null;
    let cleaned = false;
    bumpJobs(1);
    const clean = () => {
      if (!cleaned) {
        cleaned = true;
        bumpJobs(-1);
      }
    };
    const finish = (stateName) => {
      statusEl.textContent = stateName;
      statusEl.className = `job-status ${stateName.toLowerCase()}`;
    };
    const stop = () => {
      clean();
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
    };
    const poll = async () => {
      try {
        const job = await api(`/api/jobs/${jobId}`);
        prog.value = job.percent || 0;
        msgEl.textContent = job.message || job.error || "";
        con.innerHTML = job.log
          .map((l) => {
            const cls = /MATCH|FOUND|identified/.test(l)
              ? "l-match"
              : /STEP \d|Detecting|tracking|clustering|cross-camera/.test(l)
              ? "l-step"
              : "";
            return `<span class="${cls}">${esc(l)}</span>`;
          })
          .join("\n");
        con.scrollTop = con.scrollHeight;
        if (job.status === "done") {
          stop();
          finish("DONE");
          if (onDone) onDone(job);
          if (onComplete) onComplete();
          return;
        }
        if (job.status === "error") {
          stop();
          finish("ERROR");
          msgEl.textContent = job.error || job.message || "Failed";
          if (onError) onError(job);
          if (onComplete) onComplete();
          return;
        }
        timer = setTimeout(poll, 1000);
      } catch (err) {
        stop();
        finish("ERROR");
        msgEl.textContent = err.message;
        if (onError) onError(err);
        if (onComplete) onComplete();
      }
    };
    poll();
  };

  // result card HTML for a processed video / live cam run
  const resultCardHtml = (result) => {
    const persons = result.persons || [];
    const alerts = (result.alert_count || 0) > 0;
    let html = "";
    if (result.output_url) {
      html += `
        <div class="card">
          <div class="card-head"><h3>Processed feed · ${esc(result.label || "")}${result.camera_id ? ` · ${esc(result.camera_id)}` : ""}</h3></div>
          <video src="${result.output_url}?t=${Date.now()}" controls muted preload="metadata" playsinline></video>
        </div>`;
    }
    html += `<div class="card">
      <div class="card-head">
        <h3>Identified persons</h3>
        ${alerts ? `<span class="badge badge-alert">⚠ ${result.alert_count} alert(s) triggered</span>` : '<span class="badge">0 alerts</span>'}
      </div>`;
    if (!persons.length) {
      html += '<div class="empty-note">No registered persons were confidently identified in this clip (tracks not matching the database are ignored).</div>';
    } else {
      html += `<div class="person-grid compact">
        ${persons.map((p) => `
          <div class="person-card">
            ${p.crop_url ? `<img class="person-thumb clickable" src="${p.crop_url}?t=${Date.now()}" alt="" data-photo="${esc(p.crop_url)}" data-title="${esc(p.name)}">` : '<div class="person-thumb empty">no crop</div>'}
            <div class="person-info">
              <h4>${esc(p.name)} ${flagBadge(p.flag)}</h4>
              <div class="meta">similarity ${(p.similarity || 0).toFixed(3)} · ${p.frames} frame(s) · via ${esc(p.source)}</div>
            </div>
          </div>`).join("")}
      </div>`;
    }
    html += "</div>";
    return html;
  };

  // single rendered result (live cam / one-off video)
  const renderProcessedResults = (box, result) => {
    box.classList.remove("hidden");
    box.innerHTML = resultCardHtml(result);
    wirePhotos(box);
  };

  // processed videos pile up here, one by one, as each finishes
  const appendProcessedResult = (result) => {
    const list = $("#videoResultsList");
    list.classList.remove("hidden");
    const card = document.createElement("div");
    card.innerHTML = resultCardHtml(result);
    list.prepend(card);
    wirePhotos(list);
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };

  // photo lightbox (click any data-photo image)
  const openLightbox = (url, title) => {
    $("#lightboxTitle").textContent = title || "Photo";
    $("#lightboxBody").innerHTML = `<img class="lightbox-img" src="${url}?t=${Date.now()}" alt="">`;
    $("#lightbox").showModal();
  };
  const wirePhotos = (root) => {
    $$("[data-photo]", root).forEach((img) => {
      img.addEventListener("click", () => openLightbox(img.dataset.photo, img.dataset.title || "Photo"));
    });
  };
  $("#lightboxClose").addEventListener("click", () => $("#lightbox").close());
  $("#lightbox").addEventListener("click", (e) => {
    if (e.target === $("#lightbox")) $("#lightbox").close();
  });

  // ── OVERVIEW ──────────────────────────────────────────────────────────────
  const cameraCardHtml = (c) => {
    const src = c.play_url
      ? `<video src="${esc(c.play_url)}?t=${Date.now()}" controls muted preload="metadata" playsinline></video>`
      : '<div class="cam-empty">no video yet — run the pipeline or process it on Video Processing</div>';
    const badges = [
      c.source_exists ? '<span class="chip ok">online</span>' : '<span class="chip miss">no source</span>',
      c.rendered ? '<span class="chip ok">rendered</span>' : "",
    ].join("");
    const idents = (c.identified || []).join(", ");
    const stats = [
      `<span><b>${c.people_frames || 0}</b> frames w/ people</span>`,
      `<span><b>${(c.identified || []).length}</b> identified${idents ? ` · ${esc(idents)}` : ""}</span>`,
    ].join(" · ");
    return `<div class="cam-card">
      ${src}
      <div class="cam-label">
        <div class="cam-label-left"><b>${esc(c.camera_id)}</b>${badges}</div>
        <span title="${esc(c.source || "")}">${esc(c.source_name || c.source || "—")}</span>
      </div>
      <div class="cam-stats">${stats}</div>
    </div>`;
  };

  loaders.overview = async () => {
    const statsEl = $("#overviewStats");
    statsEl.innerHTML = '<div class="card"><span class="muted">Loading system status…</span></div>';
    try {
      const [o, alerts] = await Promise.all([api("/api/overview"), api("/api/alerts").catch(() => ({ alerts: [] }))]);
      state.alerts = alerts.alerts || [];
      updateAlertsBadge();
      const tiles = [
        { v: o.cameras.length, l: "Cameras" },
        { v: o.registered_persons.length, l: "Registered Persons" },
        { v: o.global_identities_count, l: "Global Identities" },
        { v: o.global_named_count || 0, l: "Named Persons" },
        { v: (alerts.alerts || []).length, l: "Active Alerts", danger: (alerts.alerts || []).length > 0 },
        { v: (o.models.device || "cpu").toUpperCase(), l: "Inference Device" },
      ];
      statsEl.innerHTML = tiles
        .map((t) => `<div class="stat-tile${t.danger ? " danger" : ""}"><div class="stat-value">${t.v}</div><div class="stat-label">${t.l}</div></div>`)
        .join("");

      $("#overviewCameras").innerHTML = o.cameras.length
        ? o.cameras.map(cameraCardHtml).join("")
        : '<div class="cam-empty">No cameras configured</div>';

      $("#overviewPersons").innerHTML = o.registered_persons.length
        ? o.registered_persons.map((n) => `<span class="chip ok">${esc(n)}</span>`).join("")
        : '<span class="muted">None registered yet — add people on the Registration page.</span>';

      const identities = o.identities || [];
      $("#overviewIdentities").innerHTML = identities.length
        ? identities.map((g) => `
            <div class="ident-row">
              <div class="ident-main">
                <b>${esc(g.name || "Unnamed person")}</b>
                <span class="muted">GID ${g.global_id}</span>
                ${flagBadge(g.flag)}
              </div>
              <div class="ident-meta">
                ${(g.cameras || []).map((cc) => `<span class="cam-mini seen">${esc(cc)}</span>`).join(" ")}
                ${g.similarity ? `<span class="muted">sim ${Number(g.similarity).toFixed(3)}</span>` : ""}
                ${g.name_source ? `<span class="muted">via ${esc(g.name_source)}</span>` : ""}
              </div>
            </div>`).join("")
        : '<span class="muted">No identities yet — run the integrated pipeline first.</span>';

      setSys(true, `API ready · ${o.cameras.length} cameras · ${o.global_identities_count} identities · ${o.registered_persons.length} persons`);
    } catch (err) {
      setSys(false, "API error");
      statsEl.innerHTML = `<div class="card"><span class="status state-error">${esc(err.message)}</span></div>`;
    }
  };

  // ── LIVE CAM ──────────────────────────────────────────────────────────────
  loaders.live = () => {};

  $("#liveBtn").addEventListener("click", async () => {
    const btn = $("#liveBtn");
    btn.disabled = true;
    $("#liveResults").classList.add("hidden");
    try {
      const { job_id } = await api("/api/live", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ duration: parseInt($("#liveDuration").value, 10) || 10 }),
      });
      jobConsole($("#liveJob"), job_id, {
        onDone: (job) => {
          renderProcessedResults($("#liveResults"), job.result || {});
          loadAlertsBadge();
        },
      });
    } catch (err) {
      alert(err.message);
    } finally {
      btn.disabled = false;
    }
  });

  // ── VIDEO PROCESSING (upload + pipeline + results) ────────────────────────
  const dz = $("#dropZone");
  const pendingFiles = () => state.pendingFiles || (state.pendingFiles = []);
  const setVideoBtn = () => {
    const n = pendingFiles().length;
    $("#videoFileLabel").textContent = n ? `${n} video(s) selected` : "no files selected";
    $("#videoBtn").disabled = n === 0;
    $("#videoBtn").textContent = n ? `▶ Run Pipeline (${n})` : "▶ Run Pipeline";
  };
  const renderSelectedVideos = () => {
    const box = $("#selectedVideos");
    box.innerHTML = pendingFiles()
      .map((f, i) => `
        <div class="sel-video">
          <video src="${URL.createObjectURL(f)}" muted preload="metadata" playsinline></video>
          <div class="sel-video-info">
            <b>${esc(f.name)}</b>
            <span class="muted">${(f.size / 1e6).toFixed(1)} MB</span>
          </div>
          <button class="btn ghost small" data-remove="${i}" title="Remove">✕</button>
        </div>`)
      .join("");
    $$("[data-remove]", box).forEach((b) =>
      b.addEventListener("click", () => {
        pendingFiles().splice(parseInt(b.dataset.remove, 10), 1);
        $("#videoFile").value = "";
        renderSelectedVideos();
        setVideoBtn();
      })
    );
    setVideoBtn();
  };

  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => {
      e.preventDefault();
      dz.classList.add("over");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => {
      e.preventDefault();
      dz.classList.remove("over");
    })
  );
  dz.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) {
      $("#videoFile").files = e.dataTransfer.files;
      $("#videoFile").dispatchEvent(new Event("change"));
    }
  });
  dz.addEventListener("click", () => $("#videoFile").click());

  $("#videoFile").addEventListener("change", (e) => {
    state.pendingFiles = [...e.target.files];
    renderSelectedVideos();
  });

  $("#videoBtn").addEventListener("click", async () => {
    const files = pendingFiles();
    if (!files.length) return;
    const btn = $("#videoBtn");
    btn.disabled = true;
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    const jobsBox = $("#videoJobs");
    jobsBox.classList.remove("hidden");
    let pending = files.length;
    const doneOne = () => {
      pending -= 1;
      if (pending <= 0) btn.disabled = false;
    };
    try {
      const { jobs } = await api("/api/videos/process", { method: "POST", body: fd });
      jobs.forEach(({ job_id }) => {
        const box = document.createElement("div");
        box.className = "jobs-box";
        jobsBox.appendChild(box);
        jobConsole(box, job_id, {
          onDone: (job) => {
            appendProcessedResult(job.result || {});
            loadAlertsBadge();
            doneOne();
          },
          onError: doneOne,
        });
      });
      state.pendingFiles = [];
      $("#videoFile").value = "";
      renderSelectedVideos();
    } catch (err) {
      alert(err.message);
      btn.disabled = false;
    }
  });

  // ── ALERTS ────────────────────────────────────────────────────────────────
  loaders.alerts = loadAlerts;

  async function loadAlerts() {
    const box = $("#alertsList");
    box.innerHTML = '<span class="muted">Loading alerts…</span>';
    try {
      const data = await api("/api/alerts");
      state.alerts = data.alerts || [];
      updateAlertsBadge();
      if (!state.alerts.length) {
        box.innerHTML = '<div class="empty-note">No alerts yet. Flag a person as criminal / missing / person of interest on the Registration page — they will show up here whenever they are recognised.</div>';
        return;
      }
      box.innerHTML = state.alerts
        .map((a) => {
          const cams = (a.cameras && a.cameras.length)
            ? a.cameras.map((c) => {
                const n = (a.cameras_frames && a.cameras_frames[c]) || "";
                return `<span class="cam-mini seen">${esc(c)}${n ? ` · ${n} fr` : ""}</span>`;
              }).join(" ")
            : `<span class="cam-mini seen">${a.camera_id ? esc(a.camera_id) : "cam"}</span>`;
          const crop = a.crop_url
            ? `<img class="alert-crop clickable" src="${esc(a.crop_url)}?t=${Date.now()}" alt="" data-photo="${esc(a.crop_url)}" data-title="${esc(a.person)}" onerror="this.style.display='none'">`
            : '<div class="alert-crop empty">no photo</div>';
          const when = a.time
            ? `<div class="alert-time"><b>Identified</b> · ${esc(a.time)}</div>`
            : "";
          const details = a.details ? `<div class="alert-details">${esc(a.details)}</div>` : "";
          const sim = a.similarity ? `similarity <b>${Number(a.similarity).toFixed(3)}</b>` : "";
          const meta = [
            `seen on ${cams}`,
            a.frames ? `${a.frames} frame(s)` : "",
            sim,
            `source · ${esc(a.source || "unknown")}`,
          ].filter(Boolean).join(" · ");
          return `<div class="alert-card">
            <div class="alert-main">
              ${crop}
              <div>
                <div class="alert-name">${esc(a.person)} ${flagBadge(a.flag)}${a.gid ? ` <span class="muted">GID ${a.gid}</span>` : ""}</div>
                ${when}
                <div class="alert-meta">${meta}</div>
                ${details}
              </div>
            </div>
            ${a.id ? `<button class="btn ghost small danger" data-dismiss-alert="${esc(a.id)}" title="Mark as handled and remove this alert">✓ Handled</button>` : ""}
          </div>`;
        })
        .join("");
      $$("[data-dismiss-alert]", box).forEach((b) =>
        b.addEventListener("click", async () => {
          const id = b.dataset.dismissAlert;
          if (!confirm("Mark this alert as handled and remove it?")) return;
          try {
            await api(`/api/alerts/${encodeURIComponent(id)}`, { method: "DELETE" });
            await loadAlerts();
            loadAlertsBadge();
          } catch (err) {
            alert(err.message);
          }
        })
      );
      wirePhotos(box);
    } catch (err) {
      box.innerHTML = `<div class="card"><span class="status state-error">${esc(err.message)}</span></div>`;
    }
  }

  $("#alertsRefresh").addEventListener("click", loadAlerts);

  async function loadAlertsBadge() {
    try {
      const data = await api("/api/alerts");
      state.alerts = data.alerts || [];
      updateAlertsBadge();
    } catch (_) {}
  }

  function updateAlertsBadge() {
    const n = (state.alerts || []).length;
    const badge = $("#alertsBadge");
    if (n > 0) {
      badge.textContent = n;
      badge.hidden = false;
    } else {
      badge.hidden = true;
    }
  }

  // ── REGISTRATION ──────────────────────────────────────────────────────────
  loaders.persons = () => {
    loadPersons();
    loadSearchSelect();
  };

  async function loadPersons() {
    const list = $("#personsList");
    list.innerHTML = '<span class="muted">Loading…</span>';
    try {
      const { persons } = await api("/api/persons");
      if (!persons.length) {
        list.innerHTML = '<div class="empty-note">No persons registered yet. Use the form above.</div>';
        return;
      }
      list.innerHTML = persons
        .map(
          (p) => `
          <div class="person-card">
            ${p.thumbnail ? `<img class="person-thumb" src="${p.thumbnail}" alt="${esc(p.name)}">` : '<div class="person-thumb empty">no photo</div>'}
            <div class="person-info">
              <h4>${esc(p.name)} ${flagBadge(p.flag)}</h4>
              ${p.person_id ? `<div class="meta">ID ${esc(p.person_id)}</div>` : ""}
              <div class="meta">${p.num_images} photo(s) · ${p.num_faces} face encoding(s) · ${p.searchable ? "searchable" : "not searchable"}</div>
              ${p.details ? `<div class="alert-details">${esc(p.details)}</div>` : ""}
            </div>
            ${p.images.length ? `<div class="person-images">${p.images.map((u) => `<img src="${u}" alt="">`).join("")}</div>` : ""}
            <div class="person-actions">
              <button class="btn ghost small" data-edit="${esc(p.name)}">Edit</button>
              <button class="btn ghost small danger" data-delete="${esc(p.name)}">Delete</button>
            </div>
          </div>`
        )
        .join("");
      $$("[data-delete]", list).forEach((b) =>
        b.addEventListener("click", async () => {
          const name = b.dataset.delete;
          if (!confirm(`Delete '${name}' from the identity database?`)) return;
          try {
            await api(`/api/persons/${encodeURIComponent(name)}`, { method: "DELETE" });
            await loadPersons();
            await loadSearchSelect();
          } catch (err) {
            alert(err.message);
          }
        })
      );
      $$("[data-edit]", list).forEach((b) =>
        b.addEventListener("click", () => {
          const p = persons.find((x) => x.name === b.dataset.edit);
          if (p) openEditModal(p);
        })
      );
    } catch (err) {
      list.innerHTML = `<div class="card"><span class="status state-error">${esc(err.message)}</span></div>`;
    }
  }

  // edit modal
  let editingName = null;
  function openEditModal(p) {
    editingName = p.name;
    $("#editModalTitle").textContent = `Edit · ${p.name}`;
    $("#editId").value = p.person_id || "";
    $("#editFlag").value = p.flag || "normal";
    $("#editDetails").value = p.details || "";
    $("#editModal").showModal();
  }
  $("#editModalClose").addEventListener("click", () => $("#editModal").close());
  $("#editCancel").addEventListener("click", () => $("#editModal").close());
  $("#editModal").addEventListener("click", (e) => {
    if (e.target === $("#editModal")) $("#editModal").close();
  });
  $("#editSave").addEventListener("click", async () => {
    if (!editingName) return;
    const payload = {
      person_id: $("#editId").value.trim() || null,
      flag: $("#editFlag").value,
      details: $("#editDetails").value.trim(),
    };
    try {
      await api(`/api/persons/${encodeURIComponent(editingName)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      $("#editModal").close();
      await loadPersons();
      loadAlertsBadge();
    } catch (err) {
      alert(err.message);
    }
  });

  $("#registerFiles").addEventListener("change", (e) => {
    const n = e.target.files.length;
    $("#registerFilesCount").textContent = n ? `${n} photo(s) selected` : "no photos selected";
  });

  $("#registerForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = $("#registerName").value.trim();
    const files = $("#registerFiles").files;
    if (!name || !files.length) {
      alert("Enter a person name and choose at least one photo.");
      return;
    }
    const fd = new FormData();
    fd.append("name", name);
    fd.append("person_id", $("#registerId").value.trim());
    fd.append("flag", $("#registerFlag").value);
    fd.append("details", $("#registerDetails").value.trim());
    for (const f of files) fd.append("files", f);
    const btn = $("button[type=submit]", e.target);
    btn.disabled = true;
    try {
      const { job_id } = await api("/api/persons", { method: "POST", body: fd });
      jobConsole($("#registerJob"), job_id, {
        onDone: async () => {
          await loadPersons();
          await loadSearchSelect();
          $("#registerName").value = "";
          $("#registerId").value = "";
          $("#registerDetails").value = "";
          $("#registerFiles").value = "";
          $("#registerFilesCount").textContent = "no photos selected";
        },
      });
    } catch (err) {
      alert(err.message);
    } finally {
      btn.disabled = false;
    }
  });

  $("#rebuildFaceDbBtn").addEventListener("click", async () => {
    try {
      const { job_id } = await api("/api/persons/rebuild-face-db", { method: "POST" });
      jobConsole($("#rebuildJob"), job_id, {
        onDone: () => {
          loadPersons();
          loadSearchSelect();
        },
      });
    } catch (err) {
      alert(err.message);
    }
  });

  // ── FACE SEARCH ───────────────────────────────────────────────────────────
  async function loadSearchSelect() {
    try {
      const { persons } = await api("/api/search/persons");
      const sel = $("#searchPerson");
      sel.innerHTML =
        '<option value="">Select a registered person…</option>' +
        persons.map((p) => `<option value="${esc(p.name)}">${esc(p.name)} (${p.num_faces} face encodings)</option>`).join("");
    } catch (_) {}
  }

  $("#searchForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = $("#searchPerson").value;
    if (!name) return;
    const payload = { name, render: $("#searchRender").checked, min_frames: 3 };
    try {
      const { job_id } = await api("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      jobConsole($("#searchJob"), job_id, { onDone: (job) => renderSearchResults(job.result) });
    } catch (err) {
      alert(err.message);
    }
  });

  $("#searchPerson").addEventListener("change", async (e) => {
    const name = e.target.value;
    if (!name) {
      $("#searchResults").innerHTML = "";
      return;
    }
    try {
      const data = await api(`/api/search/results/${encodeURIComponent(name)}`);
      if (data.exists && data.report) renderSearchResults(data.report);
      else $("#searchResults").innerHTML = '<span class="muted">No saved results for this person yet — run a search.</span>';
    } catch (_) {}
  });

  function renderSearchResults(report) {
    const box = $("#searchResults");
    if (!report) {
      box.innerHTML = "";
      return;
    }
    const head = `
      <div class="search-summary">
        <span class="badge badge-output">${esc(report.name)}</span>
        <span class="muted">face distance threshold ${report.threshold} · min ${report.min_frames} frames · ${report.videos_searched.length} video(s) searched</span>
      </div>`;
    if (!report.matches || !report.matches.length) {
      box.innerHTML = head + '<div class="empty-note">No confirmed appearances found in the searched videos.</div>';
      return;
    }
    box.innerHTML =
      head +
      report.matches
        .map((m) => {
          const segLines = m.segments
            .map(
              (s) =>
                `<div class="muted" style="margin:2px 0">frames ${s.first_frame}–${s.last_frame} · ${s.first_time} → ${s.last_time} (${s.duration_seconds}s) · similarity <b style="color:var(--accent)">${s.best_similarity.toFixed(3)}</b> · ${s.n_frames} frame(s)</div>`
            )
            .join("");
          const sheet = m.contact_url
            ? `<a class="sheet-link" href="${m.contact_url}" target="_blank" rel="noopener" title="Open the full-size verification sheet in a new tab">
                 <img src="${m.contact_url}?t=${Date.now()}" alt="verification contact sheet">
                 <span class="seg-meta">verification sheet — click to enlarge</span>
               </a>`
            : "";
          const media = m.highlight_url
            ? `<div class="seg-video">
                 <video src="${m.highlight_url}?t=${Date.now()}" controls muted preload="metadata" playsinline></video>
                 ${sheet}
                 <div class="seg-meta"><span>video: <b>${esc(m.video)}</b></span><span>fps: ${m.fps}</span></div>
               </div>`
            : "";
          return `<div class="seg-card">
            <div class="seg-head"><h3>${esc(m.video)}</h3><span class="badge">${m.segments.length} appearance(s)</span></div>
            ${segLines}
            ${media ? `<div class="seg-videos">${media}</div>` : ""}
          </div>`;
        })
        .join("");
  }

  // ── boot ──────────────────────────────────────────────────────────────────
  setInterval(loadAlertsBadge, 20000);
  show("overview");
})();
