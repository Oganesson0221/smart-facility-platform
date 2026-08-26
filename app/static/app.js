const state = {
  cameras: [],
  incidents: [],
  selectedCamera: null,
  mediaMode: "image",
  file: null,
  image: null,
  polygon: [],
  currentStatus: "",
  runtime: {
    samEnabled: false,
    samReady: false,
    samDetail: "",
  },
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

function formatDateTime(value) {
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

function notificationLabel(status = "") {
  const value = String(status).toLowerCase();
  if (value === "sent") return "Delivered";
  if (value === "disabled" || value === "not_configured") return "Not configured";
  if (value === "no_subscribers") return "No subscribed recipients";
  if (value.startsWith("partial")) return "Partially delivered";
  if (value.startsWith("failed")) return "Delivery failed";
  return value ? value.replaceAll("_", " ") : "Not requested";
}

function notificationDetail(status = "") {
  const value = String(status || "").trim();
  if (!value) return "No Telegram alert was requested.";
  if (value === "sent") return "Telegram alert delivered.";
  if (value.startsWith("sent to ")) return `Telegram alert ${value}.`;
  if (value.startsWith("partial:")) return `Telegram alert partially delivered: ${value.slice(8).trim()}.`;
  if (value.startsWith("failed:")) return `Telegram delivery failed: ${value.slice(7).trim()}.`;
  if (value === "disabled" || value === "not_configured") {
    return "Telegram alerting is not configured.";
  }
  if (value === "no_subscribers") return "Telegram has no subscribed recipients.";
  return `Telegram status: ${value}.`;
}

function visionValidationLabel(validation) {
  if (!validation) return "Not run";
  const confidence = validation.confidence == null ? "" : ` · ${formatPercent(validation.confidence)}`;
  if (validation.confirmed === true) return `Confirmed${confidence}`;
  if (validation.confirmed === false) return `Not confirmed${confidence}`;
  if (validation.accepted && validation.mode === "disabled") return "Accepted · validation disabled";
  if (validation.accepted) return "Accepted through fallback";
  return validation.mode === "unavailable" ? "Validation unavailable" : "Not confirmed";
}

function fireExitVisionValidationMarkup(validations = []) {
  if (!validations.length) {
    return `<p class="scene-section-copy">No candidate reached Nemotron validation.</p>`;
  }
  return `
    <div class="detection-detail-list">
      ${validations.map(validation => {
        const evidence = Array.isArray(validation.visible_evidence)
          ? validation.visible_evidence.filter(Boolean)
          : [];
        return `
          <div class="detection-detail-row ${validation.accepted ? "blocking" : ""}">
            <div class="detection-detail-main">
              <strong>${escapeHtml(validation.label || "candidate")}</strong>
              <small>${escapeHtml(visionValidationLabel(validation))} · ${escapeHtml(validation.model || "Nemotron")}</small>
            </div>
            <p class="detection-detail-copy">${escapeHtml(validation.summary || "No validation summary returned.")}</p>
            ${evidence.length ? `<ul class="scene-evidence-list">${evidence.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : ""}
          </div>`;
      }).join("")}
    </div>`;
}

function cameraPolygon(camera) {
  return (camera?.exit_zone || []).map(([x, y]) => [Number(x), Number(y)]);
}

function formatPercent(value = 0) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

function formatOptionalPercent(value) {
  return value == null || value === "" ? "n/a" : formatPercent(value);
}

function segmentationMethodLabel(method = "") {
  const value = String(method || "").toLowerCase();
  if (value === "sam_mask") return "SAM mask";
  if (value === "sam_rejected") return "SAM rejected";
  return "YOLO bounding box";
}

function zoneModeLabel(mode = "") {
  return String(mode || "").toLowerCase() === "full_frame"
    ? "Whole frame"
    : "Drawn polygon";
}

function escapeHtml(value = "") {
  const node = document.createElement("div");
  node.textContent = value;
  return node.innerHTML;
}

function detailCard(title, body, open = false) {
  return `
    <details class="detail-card"${open ? " open" : ""}>
      <summary>${escapeHtml(title)}</summary>
      <div class="detail-body">${body}</div>
    </details>`;
}

function compactDetail(detail = "") {
  const value = String(detail || "").trim();
  if (!value) return "Waiting for runtime.";
  return value.length > 96 ? `${value.slice(0, 93)}...` : value;
}

function setRuntimeNode(selector, title, hint, tone = "") {
  const node = $(selector);
  node.classList.remove("error", "warning");
  if (tone) node.classList.add(tone);
  node.querySelector("strong").textContent = title;
  const hintNode = node.querySelector("em");
  if (hintNode) hintNode.textContent = hint;
}

function detectionText(item) {
  return `${item.label} ${Math.round(item.confidence * 100)}%`;
}

function sceneDetectionSummary(detections = []) {
  if (!detections.length) {
    return "YOLO did not identify any supported objects above the current threshold.";
  }
  const labels = detections.slice(0, 4).map(detectionText);
  const extra = detections.length > labels.length
    ? ` and ${detections.length - labels.length} more`
    : "";
  return `YOLO identified ${labels.join(", ")}${extra}.`;
}

function sceneWorkflowMarkup(phase) {
  const detectState = phase === "detect" ? "active" : phase === "vision" || phase === "complete" ? "done" : "";
  const visionState = phase === "vision" ? "active" : phase === "complete" ? "done" : "";
  const detectText = phase === "idle"
    ? "Runs first when analysis starts"
    : phase === "detect"
      ? "Running on this upload"
      : "Detected objects captured";
  const visionText = phase === "idle"
    ? "Runs after YOLO detections are ready"
    : phase === "detect"
      ? "Waiting for YOLO output"
      : phase === "vision"
        ? "Reasoning with the local vision model now"
        : "Local vision reasoning complete";
  return `
    <div class="analysis-steps">
      <div class="analysis-step ${detectState}">
        <b>1</b>
        <div class="analysis-step-copy">
          <span class="analysis-step-title">YOLO detection</span>
          <small>${escapeHtml(detectText)}</small>
        </div>
      </div>
      <div class="analysis-step ${visionState}">
        <b>2</b>
        <div class="analysis-step-copy">
          <span class="analysis-step-title">Local vision reasoning</span>
          <small>${escapeHtml(visionText)}</small>
        </div>
      </div>
    </div>`;
}

function sceneDetectionListMarkup(detections = []) {
  if (!detections.length) {
    return `
      <div class="scene-detection-list">
        <div class="scene-detection-row empty">
          <strong>No supported YOLO detections</strong>
          <span>0%</span>
        </div>
      </div>`;
  }
  return `
    <div class="scene-detection-list">
      ${detections.map(item => `
        <div class="scene-detection-row">
          <strong>${escapeHtml(item.label)}</strong>
          <span>${Math.round(item.confidence * 100)}%</span>
        </div>
      `).join("")}
    </div>`;
}

function fireExitWorkflowMarkup(result, blocked) {
  const fullFrame = String(result.zone_mode || "").toLowerCase() === "full_frame";
  const samUsed = (result.detections || []).some(item => item.spatial_method === "sam_mask");
  const fallbacks = (result.detections || []).filter(item =>
    item.spatial_method !== "sam_mask" && (item.fallback_reason || item.segmentation_state === "fallback")
  ).length;
  const validations = result.vision_validations || [];
  const accepted = validations.filter(item => item.accepted).length;
  const rejected = validations.length - accepted;
  const validationDetail = !blocked.length
    ? "Multimodal validation was skipped because no deterministic candidate passed"
    : !validations.length
      ? "No Nemotron decision was returned"
      : rejected
        ? `${rejected} candidate${rejected === 1 ? " was" : "s were"} not confirmed by Nemotron`
        : `${accepted} candidate${accepted === 1 ? " was" : "s were"} confirmed by Nemotron`;
  const steps = [
    ["YOLO detection", `${(result.detections || []).length} candidate${(result.detections || []).length === 1 ? "" : "s"} scored locally`],
    [fullFrame ? "Full-frame pre-check" : "Fire-exit zone pre-check", fullFrame
      ? "No polygon was drawn, so the whole image became the comparison zone for relevant obstruction classes"
      : "Only relevant obstruction classes near the polygon were considered for segmentation"],
    ["SAM segmentation", samUsed
      ? fullFrame
        ? "YOLO boxes prompted SAM for relevant obstruction classes across the full image"
        : "YOLO boxes prompted SAM on zone candidates"
      : "No candidate required SAM or segmentation fell back"],
    ["Mask overlap calculation", samUsed ? "Mask-based intrusion and blockage metrics were computed" : "Existing YOLO bounding-box overlap stayed active"],
    ["Persistence", state.mediaMode === "video" ? "Track persistence remains required before incident creation" : "Still-image path does not require persistence"],
    ["Local vision validation", validationDetail],
    ["SOP retrieval", result.incidents.length ? "The grounded NeMo SOP workflow ran for the confirmed incident" : "SOP retrieval was skipped because no incident was confirmed"],
    ["Incident notification", result.incidents.length
      ? `Incident evidence stored · ${notificationDetail(result.telegram_status)}`
      : rejected
        ? "No incident or Telegram alert was created because Nemotron did not confirm the candidate"
        : "No incident was created"],
  ];
  return `
    <div class="analysis-steps">
      ${steps.map(([title, detail], index) => `
        <div class="analysis-step done">
          <b>${index + 1}</b>
          <div class="analysis-step-copy">
            <span class="analysis-step-title">${escapeHtml(title)}</span>
            <small>${escapeHtml(detail)}</small>
          </div>
        </div>
      `).join("")}
      ${fallbacks ? `
        <div class="analysis-step">
          <b>i</b>
          <div class="analysis-step-copy">
            <span class="analysis-step-title">Fallback detail</span>
            <small>${fallbacks} detection${fallbacks === 1 ? "" : "s"} used the YOLO bounding-box path.</small>
          </div>
        </div>
      ` : ""}
    </div>`;
}

function fireExitDetectionListMarkup(detections = []) {
  if (!detections.length) {
    return `
      <div class="detection-detail-list">
        <div class="detection-detail-row">
          <div class="detection-detail-main">
            <strong>No detections</strong>
            <small>The detector did not return any supported objects above the current threshold.</small>
          </div>
        </div>
      </div>`;
  }
  return `
    <div class="detection-detail-list">
      ${detections.map(item => {
        const box = (item.yolo_box || item.box || []).join(", ");
        const track = item.track_id ?? "n/a";
        const summary = item.segmentation
          ? `SAM model ${item.sam_model || "sam"} segmented the YOLO ${item.label} detection from box [${box}] into a ${item.sam_polygon?.length || 0}-point polygon.`
          : item.spatial_method === "sam_rejected"
            ? `Segmentation failed in fail-closed mode. Reason: ${item.fallback_reason || "unknown"}.`
            : item.fallback_reason
              ? `Segmentation unavailable. Falling back to the YOLO bounding-box overlap path. Reason: ${item.fallback_reason}.`
              : `YOLO box [${box}] remained the spatial method for this detection.`;
        return `
          <div class="detection-detail-row ${item.is_blocking ? "blocking" : ""}">
            <div class="detection-detail-main">
              <strong>${escapeHtml(item.label)}</strong>
              <small>${formatPercent(item.confidence)} confidence · track ${escapeHtml(String(track))} · ${escapeHtml(segmentationMethodLabel(item.spatial_method))}</small>
            </div>
            <div class="detection-detail-metrics">
              <span>YOLO box [${escapeHtml(box)}]</span>
              <span>Inside zone ${formatPercent(item.object_intrusion_ratio)}</span>
              <span>Exit blocked ${formatPercent(item.exit_blockage_ratio)}</span>
              ${item.mask_zone_iou != null ? `<span>Mask IoU ${formatPercent(item.mask_zone_iou)}</span>` : ""}
              ${item.sam_inference_ms != null ? `<span>SAM ${Math.round(item.sam_inference_ms)} ms</span>` : ""}
            </div>
            <p class="detection-detail-copy">${escapeHtml(summary)}</p>
          </div>`;
      }).join("")}
    </div>`;
}

function renderSceneReady() {
  if (state.mediaMode !== "image" || !state.file || state.polygon.length >= 3) return;
  const samActive = state.runtime.samEnabled && state.runtime.samReady;
  $("#analysisResult").innerHTML = `
    <article class="panel result-copy scene-progress-card">
      <p class="eyebrow mint">${samActive ? "Full-frame workflow ready" : "Full-frame fallback ready"}</p>
      <h3>${samActive ? "YOLO, SAM, and local vision are ready" : "YOLO is ready and SAM will fall back if needed"}</h3>
      <div class="scene-metric-grid">
        <div class="scene-metric">
          <span>Zone basis</span>
          <strong>Whole frame</strong>
        </div>
        <div class="scene-metric">
          <span>Segmentation</span>
          <strong>${samActive ? "SAM active" : "YOLO fallback"}</strong>
        </div>
      </div>
      <p>${samActive
        ? "Run the image as-is, or draw a polygon if you want to limit the decision area."
        : "Run the image as-is. If SAM is not available, the app keeps the box-overlap path active."}</p>
      ${detailCard("What will run", sceneWorkflowMarkup("idle"))}
      <p class="privacy-copy">Draw a polygon only when you want to restrict analysis to a specific clearance area.</p>
    </article>`;
}

function analysisLegendMarkup(fullFrame) {
  return `
    <div class="detection-chips">
      <span class="chip">Rectangle: YOLO detection</span>
      <span class="chip">Filled contour: SAM object mask</span>
      <span class="chip">${fullFrame ? "Green frame: analysis zone" : "Green polygon: fire-exit clearance zone"}</span>
    </div>`;
}

function fireExitWorkflowPreviewMarkup(result) {
  const fullFrame = String(result.zone_mode || "").toLowerCase() === "full_frame";
  const samUsed = (result.detections || []).some(item => item.spatial_method === "sam_mask");
  const fallbacks = (result.detections || []).filter(item =>
    item.spatial_method !== "sam_mask" && (item.fallback_reason || item.segmentation_state === "fallback")
  ).length;
  const blocked = Number(result.blocking_candidates || 0);
  const steps = [
    ["YOLO detection", `${(result.detections || []).length} candidate${(result.detections || []).length === 1 ? "" : "s"} identified locally`],
    [fullFrame ? "Full-frame pre-check" : "Fire-exit zone pre-check", fullFrame
      ? "The whole image is being used as the comparison zone because no polygon was drawn"
      : "Only detections near the drawn zone were considered for segmentation"],
    ["SAM segmentation", samUsed
      ? "Relevant YOLO boxes were refined into SAM masks and polygons"
      : fallbacks
        ? `SAM was unavailable for ${fallbacks} candidate${fallbacks === 1 ? "" : "s"}, so the YOLO box fallback stayed active`
      : "No candidate required SAM or segmentation fell back to the YOLO box path"],
    ["Mask overlap calculation", blocked
      ? `${blocked} candidate${blocked === 1 ? "" : "s"} met the deterministic intrusion/blockage thresholds`
      : "No detection met the deterministic obstruction thresholds"],
    ["Local vision validation", result.will_validate_with_vision
      ? "Confirmed candidates are being sent to the multimodal validator now"
      : blocked
        ? "Multimodal validation is disabled, so the workflow will finalize directly"
        : "Multimodal validation is skipped because no obstruction was confirmed"],
    ["Incident notification", blocked
      ? "Telegram and SOP steps will run after validation and incident creation"
      : "No alert will be sent unless a blocking candidate is confirmed"],
  ];
  return `
    <div class="analysis-steps">
      ${steps.map(([title, detail], index) => `
        <div class="analysis-step ${index < 4 ? "done" : index === 4 ? "active" : ""}">
          <b>${index + 1}</b>
          <div class="analysis-step-copy">
            <span class="analysis-step-title">${escapeHtml(title)}</span>
            <small>${escapeHtml(detail)}</small>
          </div>
        </div>
      `).join("")}
    </div>`;
}

function renderAnalysisPreview(result) {
  const fullFrame = String(result.zone_mode || "").toLowerCase() === "full_frame";
  const blocked = Number(result.blocking_candidates || 0);
  $("#analysisResult").innerHTML = `
    <article class="panel result-card scene-result-card">
      <img src="${result.annotated_image}?t=${Date.now()}" alt="Deterministic YOLO and SAM preview">
      <div class="result-copy">
        <p class="eyebrow mint">${escapeHtml(result.provider)} deterministic preview</p>
        <h3>${blocked
          ? "Deterministic checks found a candidate"
          : "Deterministic checks completed"}</h3>
        <div class="scene-metric-grid">
          <div class="scene-metric">
            <span>Zone basis</span>
            <strong>${escapeHtml(zoneModeLabel(result.zone_mode))}</strong>
          </div>
          <div class="scene-metric">
            <span>Detections</span>
            <strong>${(result.detections || []).length}</strong>
          </div>
          <div class="scene-metric">
            <span>Blocking candidates</span>
            <strong>${blocked}</strong>
          </div>
          <div class="scene-metric">
            <span>Next step</span>
            <strong>${escapeHtml(result.next_step || "Finalization")}</strong>
          </div>
        </div>
        <p class="result-summary-copy">${blocked
          ? "YOLO and SAM finished the deterministic pass. Multimodal validation is next."
          : "YOLO and SAM finished the deterministic pass. No blocking candidate needs escalation yet."}</p>
        ${detailCard("Workflow details", fireExitWorkflowPreviewMarkup(result), true)}
        ${detailCard("Detection details", fireExitDetectionListMarkup(result.detections || []))}
        ${detailCard("Overlay legend", analysisLegendMarkup(fullFrame))}
        <p class="privacy-copy">${result.will_validate_with_vision
          ? "Deterministic CV completed on this machine. Multimodal validation is running now for the confirmed candidate set."
          : "Deterministic CV completed on this machine. No multimodal validation request is needed for the current result."}</p>
      </div>
    </article>`;
}

function buildImageAnalysisForm(useFullFrame, previewToken = "") {
  const form = new FormData();
  form.append("file", state.file);
  form.append("camera_id", state.selectedCamera.id);
  form.append("exit_zone", useFullFrame ? "" : JSON.stringify(state.polygon));
  if (previewToken) form.append("preview_token", previewToken);
  return form;
}

function flushUiFrame() {
  return new Promise(resolve => requestAnimationFrame(() => resolve()));
}

function go(view) {
  const titles = {
    overview: ["Operations", "Facility overview"],
    analyse: ["Computer vision", "Analyse media"],
    cameras: ["Configuration", "Fire-exit zones"],
    settings: ["Runtime", "System connections"],
  };
  $$(".view").forEach(el => el.classList.toggle("active", el.id === `view-${view}`));
  $$(".nav-item").forEach(el => {
    const active = el.dataset.view === view;
    el.classList.toggle("active", active);
    if (active) el.setAttribute("aria-current", "page");
    else el.removeAttribute("aria-current");
  });
  $("#pageEyebrow").textContent = titles[view][0];
  $("#pageTitle").textContent = titles[view][1];
  if (view === "overview") loadOverview();
  if (view === "cameras") renderCameras();
}

async function loadHealth() {
  try {
    const health = await api("/api/health");
    state.runtime.samEnabled = Boolean(health.sam?.enabled);
    state.runtime.samReady = Boolean(health.sam?.ready);
    state.runtime.samDetail = String(health.sam?.detail || "");
    const visionEnabled = Boolean(health.vision?.enabled);
    const visionReachable = Boolean(health.vision?.reachable);
    const visionModelAvailable = health.vision?.model_available !== false;
    const nemoEnabled = Boolean(health.nemo_agent?.enabled);
    const nemoReachable = Boolean(health.nemo_agent?.reachable);
    const switchyardEnabled = Boolean(health.switchyard?.enabled);
    const switchyardReachable = Boolean(health.switchyard?.reachable);
    const telegramConfigured = Boolean(health.telegram_configured);
    $("#systemText").textContent = visionEnabled
      ? `${health.vision.model} · local stack`
      : `${health.detector} · local stack`;
    if (!visionEnabled) {
      setRuntimeNode("#visionRuntime", "Detector only", health.detector);
    } else if (!visionReachable) {
      setRuntimeNode("#visionRuntime", "Vision offline", compactDetail(health.vision?.detail), "warning");
    } else if (!visionModelAvailable) {
      setRuntimeNode("#visionRuntime", "Vision mismatch", compactDetail(health.vision?.detail), "warning");
    } else {
      setRuntimeNode("#visionRuntime", "Vision online", health.vision.model);
    }
    if (!nemoEnabled) {
      setRuntimeNode("#agentRuntime", "NeMo disabled", "Deterministic fallback");
    } else if (!nemoReachable) {
      setRuntimeNode("#agentRuntime", "NeMo offline", "Using local fallback until the NeMo server is up", "warning");
    } else {
      setRuntimeNode("#agentRuntime", "NeMo online", health.nemo_agent.model);
    }
    if (!switchyardEnabled) {
      setRuntimeNode("#switchyardRuntime", "Routing disabled", "NeMo uses its configured model directly", "warning");
    } else if (!switchyardReachable) {
      setRuntimeNode("#switchyardRuntime", "Switchyard offline", compactDetail(health.switchyard?.detail), "warning");
    } else if (health.switchyard?.route_available === false) {
      setRuntimeNode("#switchyardRuntime", "Route missing", compactDetail(health.switchyard?.detail), "warning");
    } else {
      setRuntimeNode("#switchyardRuntime", "Stage Router online", health.switchyard.route);
    }
    if (!telegramConfigured) {
      setRuntimeNode("#telegramRuntime", "Alerting optional", "Telegram not configured", "warning");
    } else {
      setRuntimeNode(
        "#telegramRuntime",
        "Alerting ready",
        `${health.telegram_recipients} recipient${health.telegram_recipients === 1 ? "" : "s"} configured`,
      );
    }
    const runtimeIssues = [
      visionEnabled && (!visionReachable || !visionModelAvailable),
      switchyardEnabled && !switchyardReachable,
      nemoEnabled && !nemoReachable,
      !telegramConfigured,
    ].filter(Boolean).length;
    const runtimeDisclosure = $(".runtime-disclosure");
    runtimeDisclosure.classList.toggle("warning", runtimeIssues > 0);
    $("#runtimeSummaryText").textContent = runtimeIssues
      ? `${runtimeIssues} service${runtimeIssues === 1 ? "" : "s"} need attention`
      : "All four services ready";
    $("#detectorName").textContent = health.detector === "yolo"
      ? health.sam?.enabled ? "Ultralytics YOLO + SAM 2" : "Ultralytics YOLO"
      : health.detector === "grounding_dino"
        ? "Grounding DINO"
        : health.vision?.enabled ? `${health.vision.model} · multimodal vision` : "Demo spatial detector";
    $("#detectorStatus").textContent = health.detector === "yolo"
      ? health.sam?.enabled
        ? `Pretrained YOLO remains the first-stage detector. ${health.sam.provider.toUpperCase()} ${health.sam.model_size} refines zone candidates with box-prompted segmentation when enabled.`
        : "Pretrained YOLO is the active first-stage detector. The saved fire-exit clearance polygon remains the source of truth."
      : health.detector === "grounding_dino"
        ? "Open-vocabulary NVIDIA-ready detector selected. It loads on first analysis."
        : health.vision?.enabled
          ? `${health.vision.model} reasons about uploaded images; the spatial detector evaluates drawn fire-exit clearance zones.`
          : "Zero-download test provider. Switch DETECTOR_PROVIDER for GPU inference.";
    $("#llmName").textContent = health.nemo_agent?.enabled
      ? health.nemo_agent?.reachable
        ? `${health.switchyard?.route || health.llm.model} · NeMo agent`
        : "NeMo unavailable"
      : health.llm.enabled ? health.llm.model : "Deterministic fallback";
    $("#llmStatus").textContent = health.nemo_agent?.enabled
      ? health.nemo_agent?.reachable
        ? `Local OpenAI-compatible reasoning orchestrated by the ${health.nemo_agent.model} NeMo workflow.`
        : `The NeMo workflow is offline, so the app is using the local SOP fallback path. Start ./scripts/run-nemo-agent.sh after both vLLM servers are available. Detail: ${health.nemo_agent?.detail || "unknown"}.`
      : health.llm.enabled
        ? `OpenAI-compatible local endpoint: ${health.llm.base_url}`
        : "Enable a local vLLM or other OpenAI-compatible endpoint in .env when ready.";
    $("#telegramName").textContent = health.telegram_configured
      ? `Telegram configured · ${health.telegram_recipients} recipient${health.telegram_recipients === 1 ? "" : "s"}`
      : "Telegram not configured";
    $("#telegramStatus").textContent = health.telegram_configured
      ? "Annotated evidence is routed to subscribed recipients through @SmartFacilityAssistant_bot."
      : "Add the bot token and recipient ID in .env, then message @SmartFacilityAssistant_bot with /start.";
    $("#telegramTest").disabled = !health.telegram_configured;
  } catch (error) {
    $("#systemText").textContent = "API unavailable";
    $("#runtimeSummaryText").textContent = "Status unavailable";
    setRuntimeNode("#switchyardRuntime", "Health unavailable", "Could not load routing status", "warning");
    $(".runtime-disclosure").classList.add("warning");
    setRuntimeNode("#visionRuntime", "Health unavailable", "Could not load runtime status", "warning");
    setRuntimeNode("#agentRuntime", "Health unavailable", "Could not load runtime status", "warning");
    setRuntimeNode("#telegramRuntime", "Health unavailable", "Could not load runtime status", "warning");
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
  list.innerHTML = state.incidents.map(incident => {
    const metadata = incident.incident_metadata || {};
    const identifier = incident.vehicle_identifier || metadata.vehicle_identifier || "";
    const identifierType = incident.vehicle_identifier_type || metadata.vehicle_identifier_type || "";
    const identifierLabel = identifier
      ? `${identifier}${identifierType && identifierType !== "none" ? ` · ${identifierType.replaceAll("_", " ")}` : ""}`
      : `${Math.round(incident.confidence * 100)}% YOLO confidence`;
    const duration = Number(metadata.blocked_duration_seconds ?? incident.duration_seconds ?? 0);
    return `
      <div class="incident-row" data-incident="${incident.id}" role="button" tabindex="0" aria-label="Review incident ${escapeHtml(incident.id)}">
        ${incident.evidence_image
          ? `<img class="evidence-thumb" src="/${incident.evidence_image}" alt="Evidence for incident ${escapeHtml(incident.id)}">`
          : `<span class="evidence-thumb"></span>`}
        <div class="incident-main"><strong>${escapeHtml(incident.zone)}</strong><small>${escapeHtml(incident.id)}</small></div>
        <div class="incident-data"><strong>${escapeHtml(incident.object_type)}</strong><small>${escapeHtml(identifierLabel)}</small></div>
        <div class="incident-data"><strong>${formatPercent(incident.object_intrusion_ratio ?? incident.overlap)}</strong><small>${escapeHtml(segmentationMethodLabel(incident.spatial_method || metadata.spatial_method))} · ${duration > 0 ? `${duration.toFixed(1)}s` : "snapshot"}</small></div>
        <div class="incident-data"><strong>${formatDate(incident.created_at)}</strong><small>${escapeHtml(incident.facility)}</small></div>
        <span class="badge ${incident.status}">${incident.status.replace("_", " ")}</span>
      </div>`;
  }).join("");
  $$(".incident-row").forEach(row => {
    row.addEventListener("click", () => openIncident(row.dataset.incident));
    row.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openIncident(row.dataset.incident);
      }
    });
  });
}

async function openIncident(id) {
  try {
    const incident = await api(`/api/incidents/${id}`);
    const metadata = incident.incident_metadata || {};
    const spatialMethod = incident.spatial_method || metadata.spatial_method || "yolo_box_fallback";
    const samPolygon = incident.sam_polygon || metadata.sam_polygon || [];
    const zoneMode = metadata.zone_mode || "polygon";
    const vehicleIdentifier = incident.vehicle_identifier || metadata.vehicle_identifier || "";
    const vehicleIdentifierType = incident.vehicle_identifier_type || metadata.vehicle_identifier_type || "";
    const vehicleIdentifierConfidence = incident.vehicle_identifier_confidence ?? metadata.vehicle_identifier_confidence;
    const yoloBox = Array.isArray(metadata.yolo_box) ? metadata.yolo_box.join(", ") : "n/a";
    const durationSeconds = Number(metadata.blocked_duration_seconds ?? incident.duration_seconds ?? 0);
    const durationLabel = durationSeconds > 0 ? `${durationSeconds.toFixed(1)} seconds` : "Image snapshot (not timed)";
    const validation = metadata.vision_validation || {};
    const validationConfirmed = validation.confirmed === true
      ? "Confirmed"
      : validation.confirmed === false
        ? "Rejected"
        : validation.mode === "disabled"
          ? "Disabled"
          : validation.mode === "unavailable"
            ? "Unavailable"
            : "Not recorded";
    const validationConfidence = validation.confidence == null ? "n/a" : formatPercent(validation.confidence);
    const validationEvidence = Array.isArray(validation.visible_evidence)
      ? validation.visible_evidence.filter(Boolean)
      : [];
    const validationEvidenceMarkup = validationEvidence.length
      ? `<ul class="scene-evidence-list">${validationEvidence.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
      : `<p>No additional visible-evidence notes were stored.</p>`;
    const segmentationNote = samPolygon.length
      ? `${samPolygon.length} polygon points · ${incident.sam_model || metadata.sam_model || "SAM"}`
      : metadata.segmentation_fallback_reason
        ? `Fallback reason: ${metadata.segmentation_fallback_reason}`
        : "No segmentation metadata stored for this incident.";
    $("#drawerContent").innerHTML = `
      ${incident.evidence_image ? `<img class="drawer-image" src="/${incident.evidence_image}" alt="Incident evidence">` : ""}
      <div class="drawer-body">
        <div class="drawer-heading">
          <p class="eyebrow mint">${escapeHtml(incident.id)}</p>
          <h2>${escapeHtml(incident.zone)}</h2>
          <p class="drawer-location">${escapeHtml(incident.facility)} · ${escapeHtml(formatDateTime(incident.created_at))}</p>
          <p class="drawer-summary">${escapeHtml(incident.summary)}</p>
        </div>

        <div class="incident-summary-grid">
          <div><span>Status</span><strong class="summary-status ${incident.status}">${escapeHtml(incident.status.replace("_", " "))}</strong></div>
          <div><span>Detected object</span><strong>${escapeHtml(incident.object_type)} · ${Math.round(incident.confidence * 100)}%</strong></div>
          <div><span>Exit blocked</span><strong>${formatOptionalPercent(incident.exit_blockage_ratio ?? metadata.exit_blockage_ratio)}</strong></div>
          <div><span>Vehicle identifier</span><strong>${escapeHtml(vehicleIdentifier || "Not readable")}</strong></div>
        </div>

        <section class="response-callout">
          <span>Recommended first action · ${escapeHtml(incident.sop_title)}</span>
          <p>${escapeHtml(incident.recommended_action)}</p>
        </section>

        <details class="drawer-section" open>
          <summary><span><strong>Operational context</strong><small>Location, overlap, timing, and alert delivery</small></span></summary>
          <div class="drawer-section-body">
            <div class="drawer-data">
              <div><span>Severity</span><strong>${escapeHtml(incident.severity)}</strong></div>
              <div><span>Blocked duration</span><strong>${escapeHtml(durationLabel)}</strong></div>
              <div><span>Object inside zone</span><strong>${formatPercent(incident.object_intrusion_ratio ?? incident.overlap)}</strong></div>
              <div><span>Mask / zone IoU</span><strong>${formatOptionalPercent(incident.mask_zone_iou ?? metadata.mask_zone_iou)}</strong></div>
              <div><span>Nemotron review</span><strong>${escapeHtml(validationConfirmed)} · ${escapeHtml(validationConfidence)}</strong></div>
              <div><span>Notification</span><strong>${escapeHtml(notificationLabel(incident.telegram_status))}</strong></div>
            </div>
            <p class="drawer-note">${escapeHtml(notificationDetail(incident.telegram_status))}</p>
          </div>
        </details>

        <details class="drawer-section">
          <summary><span><strong>Detection evidence</strong><small>YOLO, SAM, zone, and tracking measurements</small></span></summary>
          <div class="drawer-section-body">
            <div class="drawer-data">
              <div><span>YOLO confidence</span><strong>${Math.round(incident.confidence * 100)}%</strong></div>
              <div><span>YOLO box</span><strong>${escapeHtml(yoloBox)}</strong></div>
              <div><span>Zone basis</span><strong>${escapeHtml(zoneModeLabel(zoneMode))}</strong></div>
              <div><span>Spatial method</span><strong>${escapeHtml(segmentationMethodLabel(spatialMethod))}</strong></div>
              <div><span>First seen</span><strong>${escapeHtml(formatDateTime(incident.first_seen))}</strong></div>
              <div><span>Last seen</span><strong>${escapeHtml(formatDateTime(incident.last_seen))}</strong></div>
              <div><span>Track ID</span><strong>${escapeHtml(String(metadata.track_id ?? "n/a"))}</strong></div>
              <div><span>SAM inference</span><strong>${incident.sam_inference_ms != null ? `${Math.round(incident.sam_inference_ms)} ms` : "n/a"}</strong></div>
              <div><span>SAM score</span><strong>${incident.sam_score != null ? formatPercent(incident.sam_score) : "n/a"}</strong></div>
              <div><span>Mask area</span><strong>${metadata.mask_area_pixels != null ? `${Number(metadata.mask_area_pixels).toLocaleString()} px` : "n/a"}</strong></div>
            </div>
            <div class="sop-box compact-sop-box">
              <span>Segmentation detail</span>
              <strong>${escapeHtml(segmentationMethodLabel(spatialMethod))}</strong>
              <p>${escapeHtml(segmentationNote)}</p>
            </div>
          </div>
        </details>

        <details class="drawer-section">
          <summary><span><strong>Vision review</strong><small>Nemotron observations and vehicle evidence</small></span></summary>
          <div class="drawer-section-body">
            ${vehicleIdentifier ? `
              <div class="sop-box compact-sop-box">
                <span>Vehicle identifier · vision-assisted · ${formatOptionalPercent(vehicleIdentifierConfidence)}</span>
                <strong>${escapeHtml(vehicleIdentifier)}${vehicleIdentifierType && vehicleIdentifierType !== "none" ? ` · ${escapeHtml(vehicleIdentifierType.replaceAll("_", " "))}` : ""}</strong>
                <p>Verify this read against the saved image before operational use.</p>
              </div>
            ` : ""}
            <div class="sop-box compact-sop-box">
              <span>Nemotron · ${escapeHtml(validationConfirmed)}${validation.confidence != null ? ` · ${formatPercent(validation.confidence)}` : ""}</span>
              <strong>${escapeHtml(validation.summary || "Vision validation details")}</strong>
              ${validationEvidenceMarkup}
            </div>
          </div>
        </details>

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

function incidentIdFromHash() {
  const match = String(location.hash || "").match(/^#incident=(INC-[A-Za-z0-9-]+)$/i);
  return match ? decodeURIComponent(match[1]) : "";
}

function renderCameras() {
  const grid = $("#cameraGrid");
  if (!grid) return;
  grid.innerHTML = state.cameras.map(camera => `
    <article class="panel camera-card">
      <div class="camera-card-top"><span class="camera-card-label">Camera</span><button class="icon-button" data-edit-camera="${camera.id}" aria-label="Edit ${escapeHtml(camera.name)}">•••</button></div>
      <h3>${escapeHtml(camera.name)}</h3>
      <p>${escapeHtml(camera.facility)} · ${escapeHtml(camera.zone)}</p>
      <dl>
        <div><dt>Clearance zone</dt><dd>${camera.exit_zone.length} points</dd></div>
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
  const samActive = state.runtime.samEnabled && state.runtime.samReady;
  $("#zoneStatus").classList.toggle("ready", ready);
  $("#zoneStatus").innerHTML = ready
    ? `<div><strong>Clearance zone active</strong><small>${state.polygon.length} points · deterministic rules in effect</small></div>`
    : state.mediaMode === "image"
      ? samActive
        ? `<div><strong>Full-frame analysis</strong><small>YOLO prompts SAM on relevant detections across the whole image</small></div>`
        : `<div><strong>Full-frame fallback</strong><small>SAM is not ready${state.runtime.samDetail ? `: ${escapeHtml(state.runtime.samDetail)}` : ""}. YOLO box overlap stays active.</small></div>`
      : `<div><strong>Zone required for video</strong><small>Draw at least three points to run video monitoring</small></div>`;
  $("#analyseLabel").textContent = ready
    ? (state.mediaMode === "image" ? "Analyse clearance zone" : "Analyse video")
    : state.mediaMode === "image"
      ? "Analyse full frame"
      : "Run video analysis";
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
    renderSceneReady();
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
  const useFullFrame = state.mediaMode === "image" && state.polygon.length < 3;
  const button = $("#analyseButton");
  button.disabled = true;
  $("#analyseLabel").textContent = state.mediaMode === "image" ? "Analysing…" : "Uploading…";
  const updatedCamera = {
    ...state.selectedCamera,
    exit_zone: useFullFrame ? cameraPolygon(state.selectedCamera) : state.polygon,
    blocked_classes: $("#classesInput").value.split(",").map(value => value.trim()).filter(Boolean),
    minimum_overlap: Number($("#overlapInput").value),
    persistence_seconds: Number($("#durationInput").value),
  };
  delete updatedCamera.id;
  delete updatedCamera.created_at;
  try {
    state.selectedCamera = await api(`/api/cameras/${state.selectedCamera.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updatedCamera),
    });
    const cameraIndex = state.cameras.findIndex(item => item.id === state.selectedCamera.id);
    if (cameraIndex >= 0) state.cameras[cameraIndex] = state.selectedCamera;
    if (state.mediaMode === "image") {
      $("#analyseLabel").textContent = "Running YOLO + SAM…";
      const preview = await api("/api/analyse/image/preview", {
        method: "POST",
        body: buildImageAnalysisForm(useFullFrame),
      });
      renderAnalysisPreview(preview);
      await flushUiFrame();
      $("#analyseLabel").textContent = preview.will_validate_with_vision
        ? "Validating with the multimodal model…"
        : preview.blocking_candidates
          ? "Finalizing incident…"
          : "Completing analysis…";
      const result = await api("/api/analyse/image", {
        method: "POST",
        body: buildImageAnalysisForm(useFullFrame, preview.preview_token || ""),
      });
      renderAnalysisResult(result);
    } else {
      const job = await api("/api/analyse/video", {
        method: "POST",
        body: buildImageAnalysisForm(false),
      });
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

function renderSceneProgress({ eyebrow, title, message, detections = [], phase }) {
  $("#analysisResult").innerHTML = `
    <article class="panel result-copy scene-progress-card">
      <p class="eyebrow mint">${escapeHtml(eyebrow)}</p>
      <h3>${escapeHtml(title)}</h3>
      ${sceneWorkflowMarkup(phase)}
      <section class="scene-section">
        <span class="section-tag">Status</span>
        <p class="scene-section-copy">${escapeHtml(message)}</p>
      </section>
      <section class="scene-section">
        <span class="section-tag">YOLO detections</span>
        <p class="scene-section-copy">${escapeHtml(sceneDetectionSummary(detections))}</p>
        ${sceneDetectionListMarkup(detections)}
      </section>
      <p class="privacy-copy">${phase === "detect"
        ? "Local first-stage detection is running on this machine."
        : "Grounded YOLO detections are now being passed to the local vision model."}</p>
    </article>`;
}

function renderSceneResult(result) {
  const yoloSummary = (result.scene_detections || []).length
    ? `${sceneDetectionSummary(result.scene_detections)} These grounded detections were sent to the local vision model for evaluation.`
    : "YOLO did not identify any supported objects above the current threshold. The local vision model evaluated the upload with an empty detection list.";
  const evidence = result.evidence.length
    ? `<ul class="scene-evidence-list">${result.evidence.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
    : `<p class="scene-section-copy">No supporting observations returned.</p>`;
  const objects = result.visible_objects.length
    ? `<div class="detection-chips">${result.visible_objects.map(item => `<span class="chip">${escapeHtml(item)}</span>`).join("")}</div>`
    : `<p class="scene-section-copy">No additional visible objects were returned.</p>`;
  $("#analysisResult").innerHTML = `
    <article class="panel result-card scene-result-card">
      ${result.annotated_image ? `<img src="${result.annotated_image}?t=${Date.now()}" alt="Annotated violation">` : ""}
      <div class="result-copy">
        <p class="eyebrow ${result.violation ? "" : "mint"}">${escapeHtml(result.provider || "detector")} · YOLO grounding · ${escapeHtml(result.model)} vision</p>
        <h3>${result.violation ? `${escapeHtml(result.category)} issue detected` : "No visible violation detected"}</h3>
        <div class="scene-metric-grid">
          <div class="scene-metric">
            <span>Confidence</span>
            <strong>${Math.round(result.confidence * 100)}%</strong>
          </div>
          <div class="scene-metric">
            <span>Pipeline</span>
            <strong>YOLO -> Local vision</strong>
          </div>
          ${result.incident_created_at ? `
            <div class="scene-metric">
              <span>Incident time</span>
              <strong>${escapeHtml(formatDateTime(result.incident_created_at))}</strong>
            </div>
          ` : ""}
        </div>
        <p class="result-summary-copy">${escapeHtml(result.summary)}</p>
        ${detailCard("Workflow details", sceneWorkflowMarkup("complete"), true)}
        ${detailCard("YOLO detections", `<p class="scene-section-copy">${escapeHtml(yoloSummary)}</p>${sceneDetectionListMarkup(result.scene_detections || [])}`)}
        ${detailCard("Visible evidence", evidence)}
        ${detailCard("Other visible objects", objects)}
        ${result.recommended_action ? `
          <section class="scene-section scene-section-action">
            <span class="section-tag">Recommended first action</span>
            <p class="scene-lead">${escapeHtml(result.recommended_action)}</p>
          </section>
        ` : ""}
        <p class="privacy-copy">${result.violation
          ? `Incident recorded · ${escapeHtml(notificationDetail(result.telegram_status))}`
          : "No incident or alert was created."}</p>
        ${result.incidents?.length ? `<button class="button bright" data-open-scene="${result.incidents[0]}">Review incident</button>` : ""}
      </div>
    </article>`;
  const open = $("[data-open-scene]");
  if (open) open.addEventListener("click", () => openIncident(open.dataset.openScene));
}

function renderAnalysisResult(result) {
  const blocked = result.detections.filter(item => item.is_blocking);
  const validations = result.vision_validations || [];
  const rejectedValidations = validations.filter(item => !item.accepted);
  const primaryValidation = validations[0] || null;
  const primary = blocked[0] || result.detections[0] || null;
  const spatialMethod = primary ? segmentationMethodLabel(primary.spatial_method) : "None";
  const fullFrame = String(result.zone_mode || "").toLowerCase() === "full_frame";
  $("#analysisResult").innerHTML = `
    <article class="panel result-card scene-result-card">
      <img src="${result.annotated_image}?t=${Date.now()}" alt="Annotated analysis result">
      <div class="result-copy">
        <p class="eyebrow ${blocked.length ? "" : "mint"}">${escapeHtml(result.provider)} ${fullFrame ? "full-frame workflow" : "fire-exit workflow"}</p>
        <h3>${result.incidents.length
          ? `${result.incidents.length} incident${result.incidents.length > 1 ? "s" : ""} confirmed`
          : rejectedValidations.length
            ? `${rejectedValidations.length} candidate${rejectedValidations.length > 1 ? "s" : ""} not confirmed`
          : blocked.length
            ? `${blocked.length} obstruction candidate${blocked.length > 1 ? "s" : ""} detected`
          : fullFrame
            ? "No blocking object met the full-frame thresholds"
            : "Fire-exit clearance zone is clear"}</h3>
        <div class="scene-metric-grid">
          <div class="scene-metric">
            <span>Zone basis</span>
            <strong>${escapeHtml(zoneModeLabel(result.zone_mode))}</strong>
          </div>
          <div class="scene-metric">
            <span>Primary method</span>
            <strong>${escapeHtml(spatialMethod)}</strong>
          </div>
          <div class="scene-metric">
            <span>Detections</span>
            <strong>${result.detections.length}</strong>
          </div>
          <div class="scene-metric">
            <span>Incidents</span>
            <strong>${result.incidents.length}</strong>
          </div>
          <div class="scene-metric">
            <span>Notification</span>
            <strong>${escapeHtml(notificationLabel(result.telegram_status || ""))}</strong>
          </div>
          <div class="scene-metric">
            <span>Nemotron</span>
            <strong>${escapeHtml(visionValidationLabel(primaryValidation))}</strong>
          </div>
        </div>
        <p class="result-summary-copy">${result.incidents.length
          ? `YOLO and SAM identified the obstruction, Nemotron confirmed it, and ${result.incidents.length} incident record(s) were created. ${escapeHtml(notificationDetail(result.telegram_status))}`
          : rejectedValidations.length
            ? `YOLO and SAM identified an obstruction candidate, but Nemotron did not confirm it: ${escapeHtml(primaryValidation?.summary || "no confirmation reason was returned")}. No incident or Telegram alert was created.`
          : blocked.length
            ? "YOLO and SAM identified an obstruction candidate, but no final incident decision was returned."
          : "Objects may have been detected, but none met the configured class, intrusion, blockage, and persistence rules."}</p>
        ${detailCard("Workflow details", fireExitWorkflowMarkup(result, blocked), true)}
        ${validations.length ? detailCard("Nemotron decision", fireExitVisionValidationMarkup(validations), true) : ""}
        ${detailCard(
          "Detection details",
          `<p class="scene-section-copy">${fullFrame
            ? "The evidence image shows the full-frame border, YOLO box, and SAM contour where segmentation succeeded."
            : "The evidence image shows the fire-exit polygon, YOLO box, and SAM contour where segmentation succeeded."}</p>${fireExitDetectionListMarkup(result.detections)}`
        )}
        ${detailCard("Overlay legend", analysisLegendMarkup(fullFrame))}
        ${result.incidents.length ? `<button class="button bright" data-open-result="${result.incidents[0]}">Review incident</button>` : ""}
      </div>
    </article>`;
  const open = $("[data-open-result]");
  if (open) open.addEventListener("click", () => openIncident(open.dataset.openResult));
  toast(result.incidents.length
    ? "Incident confirmed and notification processed"
    : rejectedValidations.length
      ? "Candidate was not confirmed by Nemotron"
      : "Analysis completed");
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
  $("#cameraDialogTitle").textContent = camera ? "Edit fire-exit clearance zone" : "Add fire-exit clearance zone";
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
  const runtimeDisclosure = $(".runtime-disclosure");
  runtimeDisclosure.addEventListener("toggle", () => {
    $(".runtime-summary-action").textContent = runtimeDisclosure.open ? "Hide status" : "View status";
  });
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
    if (state.image && state.mediaMode === "video") {
      state.polygon = cameraPolygon(state.selectedCamera);
      drawCanvas();
    }
    if (state.image) renderSceneReady();
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
            state.polygon = cameraPolygon(state.selectedCamera);
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
    state.polygon = cameraPolygon(state.selectedCamera);
  } else if (mode === "image") {
    state.polygon = [];
  }
  if (state.image) drawCanvas();
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
  const linkedIncident = incidentIdFromHash();
  if (linkedIncident) await openIncident(linkedIncident);
  window.addEventListener("hashchange", () => {
    const incidentId = incidentIdFromHash();
    if (incidentId) openIncident(incidentId);
  });
  connectEvents();
}

init();
