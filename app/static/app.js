const state = {
  cameras: [],
  incidents: [],
  selectedCamera: null,
  mediaMode: "image",
  file: null,
  image: null,
  polygon: [],
  currentStatus: "",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function toast(message, type = "") {
  const item = document.createElement("div");
  item.className = `toast ${type}`;
  item.textContent = message;
  $("#toastStack").append(item);
  setTimeout(() => item.remove(), 4200);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch (_) {}
    throw new Error(message);
  }
  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("json") ? response.json() : response;
}

function formatDate(value) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
  }).format(new Date(value));
}

function notificationLabel(status = "") {
  const value = String(status).toLowerCase();
  if (value === "sent") return "Delivered";
  if (value === "disabled" || value === "not_configured") return "Not configured";
  if (value === "no_subscribers") return "No subscribed recipients";
  if (value.startsWith("failed")) return "Delivery pending";
  return value ? value.replaceAll("_", " ") : "Not requested";
}

function escapeHtml(value = "") {
  const node = document.createElement("div");
  node.textContent = value;
  return node.innerHTML;
}

function go(view) {
  const titles = {
    overview: ["Operations", "Facility overview"],
    analyse: ["Computer vision", "Analyse media"],
    cameras: ["Configuration", "Protected exits"],
    settings: ["Runtime", "System connections"],
  };
  $$(".view").forEach(el => el.classList.toggle("active", el.id === `view-${view}`));
  $$(".nav-item").forEach(el => el.classList.toggle("active", el.dataset.view === view));
  $("#pageEyebrow").textContent = titles[view][0];
  $("#pageTitle").textContent = titles[view][1];
  if (view === "overview") loadOverview();
  if (view === "cameras") renderCameras();
}

async function loadHealth() {
  try {
    const health = await api("/api/health");
    $("#systemText").textContent = health.vision?.enabled
      ? `${health.vision.model} · local vision`
      : `${health.detector} · local`;
    $("#visionRuntimeText").textContent = health.vision?.enabled
      ? health.vision.model
      : health.detector;
    $("#agentRuntimeText").textContent = health.nemo_agent?.enabled
      ? `${health.nemo_agent.model} · NeMo`
      : "Deterministic fallback";
    $("#telegramRuntime").classList.remove("error");
    $("#telegramRuntime").classList.toggle("warning", !health.telegram_configured);
    $("#telegramRuntimeText").textContent = !health.telegram_configured
      ? "Not configured"
      : `${health.telegram_recipients} recipient${health.telegram_recipients === 1 ? "" : "s"} configured`;
    $("#detectorName").textContent = health.detector === "grounding_dino"
      ? "Grounding DINO"
      : health.vision?.enabled ? `${health.vision.model} · multimodal vision` : "Demo spatial detector";
    $("#detectorStatus").textContent = health.detector === "grounding_dino"
      ? "Open-vocabulary NVIDIA-ready detector selected. It loads on first analysis."
      : health.vision?.enabled
        ? `${health.vision.model} reasons about uploaded images; the demo detector handles drawn exit zones.`
        : "Zero-download test provider. Switch DETECTOR_PROVIDER for GPU inference.";
    $("#llmName").textContent = health.nemo_agent?.enabled
      ? `${health.llm.model} · NeMo agent`
      : health.llm.enabled ? health.llm.model : "Deterministic fallback";
    $("#llmStatus").textContent = health.nemo_agent?.enabled
      ? `Local Ollama reasoning orchestrated by the ${health.nemo_agent.model} NeMo workflow.`
      : health.llm.enabled
        ? `OpenAI-compatible local endpoint: ${health.llm.base_url}`
        : "Enable a local Ollama, vLLM, or NVIDIA NIM endpoint in .env when ready.";
    $("#telegramName").textContent = health.telegram_configured
      ? `Telegram configured · ${health.telegram_recipients} recipient${health.telegram_recipients === 1 ? "" : "s"}`
      : "Telegram not configured";
    $("#telegramStatus").textContent = health.telegram_configured
      ? "Annotated evidence is routed to subscribed recipients through @SmartFacilityAssistant_bot."
      : "Add the bot token and recipient ID in .env, then message @SmartFacilityAssistant_bot with /start.";
    $("#telegramTest").disabled = !health.telegram_configured;
  } catch (error) {
    $("#systemText").textContent = "API unavailable";
    toast(error.message, "error");
  }
}

async function loadCameras() {
  state.cameras = await api("/api/cameras");
  if (!state.selectedCamera && state.cameras.length) state.selectedCamera = state.cameras[0];
  const select = $("#cameraSelect");
  select.innerHTML = state.cameras.map(camera =>
    `<option value="${camera.id}">${escapeHtml(camera.facility)} · ${escapeHtml(camera.zone)}</option>`
  ).join("");
  if (state.selectedCamera) {
    select.value = state.selectedCamera.id;
    populateCameraControls(state.selectedCamera);
  }
  renderCameras();
}

function populateCameraControls(camera) {
  $("#overlapInput").value = camera.minimum_overlap;
  $("#durationInput").value = camera.persistence_seconds;
  $("#classesInput").value = camera.blocked_classes.join(", ");
}

async function loadOverview() {
  try {
    const [stats, incidents] = await Promise.all([
      api("/api/stats"),
      api(`/api/incidents${state.currentStatus ? `?status=${state.currentStatus}` : ""}`),
    ]);
    state.incidents = incidents;
    $("#statOpen").textContent = stats.open;
    $("#statAck").textContent = stats.acknowledged;
    $("#statCameras").textContent = stats.cameras;
    $("#statFalse").textContent = stats.false_alarm;
    renderIncidents();
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderIncidents() {
  const list = $("#incidentList");
  if (!state.incidents.length) {
    list.innerHTML = `<div class="empty-state"><strong>No incidents in this view</strong><br><small>Use the synthetic demo frame to exercise the complete workflow.</small></div>`;
    return;
  }
  list.innerHTML = state.incidents.map(incident => `
    <div class="incident-row" data-incident="${incident.id}">
      ${incident.evidence_image
        ? `<img class="evidence-thumb" src="/${incident.evidence_image}" alt="">`
        : `<span class="evidence-thumb"></span>`}
      <div class="incident-main"><strong>${escapeHtml(incident.zone)}</strong><small>${escapeHtml(incident.id)}</small></div>
      <div class="incident-data"><strong>${escapeHtml(incident.object_type)}</strong><small>${Math.round(incident.confidence * 100)}% confidence</small></div>
      <div class="incident-data"><strong>${Math.round(incident.overlap * 100)}% overlap</strong><small>${incident.duration_seconds.toFixed(1)}s persistent</small></div>
      <div class="incident-data"><strong>${formatDate(incident.created_at)}</strong><small>${escapeHtml(incident.facility)}</small></div>
      <span class="badge ${incident.status}">${incident.status.replace("_", " ")}</span>
    </div>
  `).join("");
  $$(".incident-row").forEach(row => row.addEventListener("click", () => openIncident(row.dataset.incident)));
}

async function openIncident(id) {
  try {
    const incident = await api(`/api/incidents/${id}`);
    $("#drawerContent").innerHTML = `
      ${incident.evidence_image ? `<img class="drawer-image" src="/${incident.evidence_image}" alt="Incident evidence">` : ""}
      <div class="drawer-body">
        <p class="eyebrow mint">${escapeHtml(incident.id)}</p>
        <h2>${escapeHtml(incident.zone)}</h2>
        <p>${escapeHtml(incident.summary)}</p>
        <div class="drawer-data">
          <div><span>Status</span><strong>${escapeHtml(incident.status.replace("_", " "))}</strong></div>
          <div><span>Severity</span><strong>${escapeHtml(incident.severity)}</strong></div>
          <div><span>Detected</span><strong>${escapeHtml(incident.object_type)}</strong></div>
          <div><span>Confidence</span><strong>${Math.round(incident.confidence * 100)}%</strong></div>
          <div><span>ROI overlap</span><strong>${Math.round(incident.overlap * 100)}%</strong></div>
          <div><span>Notification</span><strong>${escapeHtml(notificationLabel(incident.telegram_status))}</strong></div>
        </div>
        <div class="sop-box">
          <span>Grounded recommendation · ${escapeHtml(incident.sop_title)}</span>
          <strong>Recommended first action</strong>
          <p>${escapeHtml(incident.recommended_action)}</p>
        </div>
        <div class="drawer-actions">
          <button class="button bright" data-action="acknowledge">Acknowledge</button>
          <button class="button ghost" data-action="false-alarm">False alarm</button>
          <button class="button subtle" data-action="close">Close</button>
        </div>
      </div>`;
    $("#incidentDrawer").classList.add("open");
    $("#drawerBackdrop").classList.add("open");
    $$(".drawer-actions button").forEach(button => button.addEventListener("click", async () => {
      try {
        await api(`/api/incidents/${id}/${button.dataset.action}`, { method: "POST" });
        toast(`Incident ${button.dataset.action.replace("-", " ")}`);
        closeDrawer();
        loadOverview();
      } catch (error) { toast(error.message, "error"); }
    }));
  } catch (error) { toast(error.message, "error"); }
}

function closeDrawer() {
  $("#incidentDrawer").classList.remove("open");
  $("#drawerBackdrop").classList.remove("open");
}

function renderCameras() {
  const grid = $("#cameraGrid");
  if (!grid) return;
  grid.innerHTML = state.cameras.map(camera => `
    <article class="panel camera-card">
      <div class="camera-card-top"><span class="camera-card-icon">◉</span><button class="icon-button" data-edit-camera="${camera.id}">•••</button></div>
      <h3>${escapeHtml(camera.name)}</h3>
      <p>${escapeHtml(camera.facility)} · ${escapeHtml(camera.zone)}</p>
      <dl>
        <div><dt>Exit polygon</dt><dd>${camera.exit_zone.length} points</dd></div>
        <div><dt>Persistence</dt><dd>${camera.persistence_seconds}s</dd></div>
        <div><dt>Min. overlap</dt><dd>${Math.round(camera.minimum_overlap * 100)}%</dd></div>
        <div><dt>State</dt><dd>${camera.enabled ? "Enabled" : "Disabled"}</dd></div>
      </dl>
    </article>
  `).join("");
  $$("[data-edit-camera]").forEach(button => button.addEventListener("click", () => editCamera(button.dataset.editCamera)));
}

const canvas = $("#roiCanvas");
const ctx = canvas.getContext("2d");

function drawCanvas() {
  if (!state.image) return;
  canvas.width = state.image.naturalWidth || state.image.videoWidth;
  canvas.height = state.image.naturalHeight || state.image.videoHeight;
  ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
  if (state.polygon.length) {
    ctx.beginPath();
    state.polygon.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    if (state.polygon.length >= 3) ctx.closePath();
    ctx.fillStyle = "rgba(65, 225, 143, .15)";
    ctx.fill();
    ctx.strokeStyle = "#8cf0bd";
    ctx.lineWidth = Math.max(2, canvas.width / 450);
    ctx.stroke();
    state.polygon.forEach(([x, y], index) => {
      ctx.beginPath();
      ctx.arc(x, y, Math.max(5, canvas.width / 180), 0, Math.PI * 2);
      ctx.fillStyle = "#c7ff77";
      ctx.fill();
      ctx.fillStyle = "#07100d";
      ctx.font = `bold ${Math.max(10, canvas.width / 90)}px sans-serif`;
      ctx.fillText(String(index + 1), x + 8, y - 8);
    });
  }
  updateZoneStatus();
}

function updateZoneStatus() {
  const ready = state.polygon.length >= 3;
  $("#zoneStatus").classList.toggle("ready", ready);
  $("#zoneStatus").innerHTML = ready
    ? `<span>◆</span><div><strong>Protected-exit mode</strong><small>${state.polygon.length} points · object overlap rules apply</small></div>`
    : state.mediaMode === "image"
      ? `<span>✦</span><div><strong>Automatic scene reasoning</strong><small>Local vision model checks visible objects and signs</small></div>`
      : `<span>◇</span><div><strong>No exit zone drawn</strong><small>Video monitoring requires at least three points</small></div>`;
  $("#analyseLabel").textContent = ready
    ? (state.mediaMode === "image" ? "Analyse exit zone" : "Analyse video")
    : "Reason about scene";
  $("#analyseButton").disabled = !(state.file && (ready || state.mediaMode === "image"));
}

function showImage(blob, filename = "image.jpg") {
  const image = new Image();
  image.onload = () => {
    state.image = image;
    state.polygon = [];
    canvas.style.display = "block";
    $("#stagePlaceholder").style.display = "none";
    $("#drawHint").style.display = "block";
    drawCanvas();
    URL.revokeObjectURL(image.src);
  };
  image.src = URL.createObjectURL(blob);
  state.file = new File([blob], filename, { type: blob.type || "image/jpeg" });
}

canvas.addEventListener("click", event => {
  if (!state.image) return;
  const rect = canvas.getBoundingClientRect();
  state.polygon.push([
    Math.round((event.clientX - rect.left) * canvas.width / rect.width),
    Math.round((event.clientY - rect.top) * canvas.height / rect.height),
  ]);
  drawCanvas();
});

async function runAnalysis() {
  if (!state.file || !state.selectedCamera) return;
  const useSceneReasoning = state.mediaMode === "image" && state.polygon.length < 3;
  const button = $("#analyseButton");
  button.disabled = true;
  $("#analyseLabel").textContent = useSceneReasoning ? "Reasoning locally…" : (state.mediaMode === "image" ? "Analysing…" : "Uploading…");
  const form = new FormData();
  form.append("file", state.file);
  form.append("camera_id", state.selectedCamera.id);
  if (useSceneReasoning) {
    try {
      const result = await api("/api/analyse/scene", { method: "POST", body: form });
      renderSceneResult(result);
    } catch (error) {
      toast(error.message, "error");
    } finally {
      button.disabled = false;
      updateZoneStatus();
    }
    return;
  }
  const updatedCamera = {
    ...state.selectedCamera,
    exit_zone: state.polygon,
    blocked_classes: $("#classesInput").value.split(",").map(value => value.trim()).filter(Boolean),
    minimum_overlap: Number($("#overlapInput").value),
    persistence_seconds: Number($("#durationInput").value),
  };
  delete updatedCamera.id;
  delete updatedCamera.created_at;
  form.append("camera_id", state.selectedCamera.id);
  form.append("exit_zone", JSON.stringify(state.polygon));
  try {
    state.selectedCamera = await api(`/api/cameras/${state.selectedCamera.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updatedCamera),
    });
    const cameraIndex = state.cameras.findIndex(item => item.id === state.selectedCamera.id);
    if (cameraIndex >= 0) state.cameras[cameraIndex] = state.selectedCamera;
    if (state.mediaMode === "image") {
      const result = await api("/api/analyse/image", { method: "POST", body: form });
      renderAnalysisResult(result);
    } else {
      const job = await api("/api/analyse/video", { method: "POST", body: form });
      renderJob(job);
      pollJob(job.id);
    }
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    updateZoneStatus();
  }
}

function renderSceneResult(result) {
  const evidence = result.evidence.length
    ? `<ul>${result.evidence.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
    : "<p>No supporting observations returned.</p>";
  const objects = result.visible_objects.length
    ? `<div class="detection-chips">${result.visible_objects.map(item => `<span class="chip">${escapeHtml(item)}</span>`).join("")}</div>`
    : "";
  $("#analysisResult").innerHTML = `
    <article class="panel result-card">
      ${result.annotated_image ? `<img src="${result.annotated_image}?t=${Date.now()}" alt="Annotated violation">` : ""}
      <div class="result-copy">
        <p class="eyebrow ${result.violation ? "" : "mint"}">${escapeHtml(result.model)} · local vision</p>
        <h3>${result.violation ? `${escapeHtml(result.category)} issue detected` : "No visible violation detected"}</h3>
        <p>${escapeHtml(result.summary)}</p>
        <p><strong>${Math.round(result.confidence * 100)}% confidence</strong></p>
        ${evidence}
        ${objects}
        <p class="privacy-copy">${result.violation
          ? `Incident recorded · Notification: ${escapeHtml(notificationLabel(result.telegram_status))}.`
          : "No incident or alert was created."}</p>
        ${result.incidents?.length ? `<button class="button bright" data-open-scene="${result.incidents[0]}">Review incident</button>` : ""}
      </div>
    </article>`;
  const open = $("[data-open-scene]");
  if (open) open.addEventListener("click", () => openIncident(open.dataset.openScene));
}

function renderAnalysisResult(result) {
  const blocked = result.detections.filter(item => item.is_blocking);
  $("#analysisResult").innerHTML = `
    <article class="panel result-card">
      <img src="${result.annotated_image}?t=${Date.now()}" alt="Annotated analysis result">
      <div class="result-copy">
        <p class="eyebrow ${blocked.length ? "" : "mint"}">${escapeHtml(result.provider)} analysis</p>
        <h3>${blocked.length ? `${blocked.length} obstruction${blocked.length > 1 ? "s" : ""} detected` : "Exit zone is clear"}</h3>
        <p>${blocked.length
          ? `The spatial rule created ${result.incidents.length} incident record(s). Open the incident queue to review the SOP-grounded response.`
          : "Objects may have been detected, but none met the configured class and overlap rule."}</p>
        <div class="detection-chips">${result.detections.length
          ? result.detections.map(item => `<span class="chip ${item.is_blocking ? "blocking" : ""}">${escapeHtml(item.label)} · ${Math.round(item.confidence * 100)}% · ${Math.round(item.overlap * 100)}% ROI</span>`).join("")
          : `<span class="chip">No objects detected</span>`}</div>
        ${result.incidents.length ? `<button class="button bright" data-open-result="${result.incidents[0]}">Review incident</button>` : ""}
      </div>
    </article>`;
  const open = $("[data-open-result]");
  if (open) open.addEventListener("click", () => openIncident(open.dataset.openResult));
  toast(blocked.length ? "Incident workflow completed" : "Analysis completed");
}

function renderJob(job) {
  $("#analysisResult").innerHTML = `
    <article class="panel result-copy">
      <p class="eyebrow mint">Video job · ${escapeHtml(job.id)}</p>
      <h3 id="jobTitle">Processing uploaded video</h3>
      <div class="progress-shell"><div class="progress-bar" id="jobProgress" style="width:${job.progress}%"></div></div>
      <p id="jobMessage">${escapeHtml(job.message || "Queued for local processing")}</p>
    </article>`;
}

async function pollJob(id) {
  try {
    const job = await api(`/api/jobs/${id}`);
    $("#jobProgress").style.width = `${job.progress}%`;
    $("#jobMessage").textContent = job.message;
    if (job.status === "completed") {
      $("#jobTitle").textContent = "Video analysis complete";
      toast(job.message);
      if (job.incidents.length) {
        const button = document.createElement("button");
        button.className = "button bright";
        button.textContent = "Review incident";
        button.onclick = () => openIncident(job.incidents[0]);
        $("#jobMessage").after(button);
      }
    } else if (job.status === "failed") {
      $("#jobTitle").textContent = "Video analysis failed";
      toast(job.message, "error");
    } else {
      setTimeout(() => pollJob(id), 1000);
    }
  } catch (error) { toast(error.message, "error"); }
}

function editCamera(id = "") {
  const camera = state.cameras.find(item => item.id === id);
  $("#cameraDialogTitle").textContent = camera ? "Edit protected exit" : "Add protected exit";
  $("#cameraId").value = camera?.id || "";
  $("#cameraName").value = camera?.name || "";
  $("#cameraFacility").value = camera?.facility || "";
  $("#cameraZone").value = camera?.zone || "";
  $("#cameraRtsp").value = camera?.rtsp_url || "";
  $("#cameraOverlap").value = camera?.minimum_overlap ?? .25;
  $("#cameraPersistence").value = camera?.persistence_seconds ?? 5;
  $("#cameraDialog").showModal();
}

async function saveCamera(event) {
  event.preventDefault();
  const id = $("#cameraId").value;
  const existing = state.cameras.find(camera => camera.id === id);
  const payload = {
    name: $("#cameraName").value,
    facility: $("#cameraFacility").value,
    zone: $("#cameraZone").value,
    rtsp_url: $("#cameraRtsp").value || null,
    exit_zone: existing?.exit_zone || [],
    blocked_classes: existing?.blocked_classes || ["vehicle", "trolley", "pallet", "box", "large object"],
    confidence_threshold: existing?.confidence_threshold || .35,
    minimum_overlap: Number($("#cameraOverlap").value),
    persistence_seconds: Number($("#cameraPersistence").value),
    alert_cooldown_seconds: existing?.alert_cooldown_seconds || 300,
    enabled: true,
  };
  try {
    await api(id ? `/api/cameras/${id}` : "/api/cameras", {
      method: id ? "PUT" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    $("#cameraDialog").close();
    await loadCameras();
    toast(id ? "Camera updated" : "Camera added");
  } catch (error) { toast(error.message, "error"); }
}

function wireEvents() {
  $$(".nav-item").forEach(button => button.addEventListener("click", () => go(button.dataset.view)));
  $$("[data-go]").forEach(button => button.addEventListener("click", () => go(button.dataset.go)));
  $("#refreshOverview").addEventListener("click", loadOverview);
  $$(".filter").forEach(button => button.addEventListener("click", () => {
    $$(".filter").forEach(item => item.classList.remove("active"));
    button.classList.add("active");
    state.currentStatus = button.dataset.status;
    loadOverview();
  }));
  $("#drawerClose").addEventListener("click", closeDrawer);
  $("#drawerBackdrop").addEventListener("click", closeDrawer);
  $("#cameraSelect").addEventListener("change", event => {
    state.selectedCamera = state.cameras.find(camera => camera.id === event.target.value);
    populateCameraControls(state.selectedCamera);
  });
  $("#mediaInput").addEventListener("change", event => {
    const file = event.target.files[0];
    if (!file) return;
    state.file = file;
    if (state.mediaMode === "image") showImage(file, file.name);
    else {
      const video = document.createElement("video");
      video.muted = true;
      video.preload = "metadata";
      video.onloadeddata = () => {
        video.currentTime = Math.min(.1, video.duration || .1);
      };
      video.onseeked = () => {
        const preview = document.createElement("canvas");
        preview.width = video.videoWidth;
        preview.height = video.videoHeight;
        preview.getContext("2d").drawImage(video, 0, 0);
        preview.toBlob(blob => {
          const image = new Image();
          image.onload = () => {
            state.image = image;
            state.polygon = state.selectedCamera?.exit_zone || [];
            canvas.style.display = "block";
            $("#stagePlaceholder").style.display = "none";
            $("#drawHint").style.display = "block";
            drawCanvas();
            URL.revokeObjectURL(image.src);
          };
          image.src = URL.createObjectURL(blob);
        }, "image/jpeg", .9);
        URL.revokeObjectURL(video.src);
      };
      video.src = URL.createObjectURL(file);
    }
  });
  $("#demoButton").addEventListener("click", async () => {
    try {
      state.mediaMode = "image";
      setMediaMode("image");
      const response = await fetch("/api/demo/frame");
      showImage(await response.blob(), "smart-facility-demo.jpg");
      state.polygon = [[250, 100], [850, 100], [850, 680], [250, 680]];
      setTimeout(drawCanvas, 30);
    } catch (error) { toast(error.message, "error"); }
  });
  $("#resetPolygon").addEventListener("click", () => { state.polygon = []; drawCanvas(); });
  $("#analyseButton").addEventListener("click", runAnalysis);
  $$("[data-media]").forEach(button => button.addEventListener("click", () => setMediaMode(button.dataset.media)));
  $("#newCameraButton").addEventListener("click", () => editCamera());
  $("#cameraForm").addEventListener("submit", saveCamera);
  $("#telegramTest").addEventListener("click", async () => {
    try {
      const result = await api("/api/telegram/test", { method: "POST" });
      toast(`Telegram: ${result.status}`);
    } catch (error) { toast(error.message, "error"); }
  });
}

function setMediaMode(mode) {
  state.mediaMode = mode;
  $$("[data-media]").forEach(button => button.classList.toggle("active", button.dataset.media === mode));
  $("#mediaInput").accept = mode === "image" ? "image/*" : "video/*";
  $("#analyseLabel").textContent = mode === "image" ? "Analyse image" : "Analyse video";
  if (mode === "video" && state.selectedCamera?.exit_zone?.length >= 3) {
    state.polygon = state.selectedCamera.exit_zone;
  }
  updateZoneStatus();
}

function connectEvents() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/api/events`);
  socket.onmessage = event => {
    const payload = JSON.parse(event.data);
    if (payload.type === "incident.created") {
      toast(`New incident: ${payload.incident_id}`);
      loadOverview();
    }
  };
  socket.onclose = () => setTimeout(connectEvents, 3000);
}

async function init() {
  wireEvents();
  await Promise.all([loadHealth(), loadCameras(), loadOverview()]);
  connectEvents();
}

init();
