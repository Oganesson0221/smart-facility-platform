const $ = selector => document.querySelector(selector);

function text(value, fallback = "—") {
  return value === undefined || value === null || value === "" ? fallback : String(value);
}

function setCard(cardSelector, stateSelector, detailSelector, status, title, detail) {
  const card = $(cardSelector);
  card.classList.toggle("ready", status === "ready");
  card.classList.toggle("error", status === "error");
  $(stateSelector).textContent = title;
  $(detailSelector).textContent = detail;
}

function endpointState(endpoint, fallbackName) {
  if (!endpoint?.reachable) {
    return ["error", `${fallbackName} offline`, text(endpoint?.detail, "Endpoint unavailable")];
  }
  if (!endpoint.model_available) {
    return ["error", `${fallbackName} mismatch`, text(endpoint.detail)];
  }
  return ["ready", fallbackName, `${endpoint.model} · ${endpoint.base_url}`];
}

function renderDecisions(stageRouter = {}) {
  const decisions = stageRouter.routing_decisions || {};
  const rows = [];
  Object.entries(decisions).forEach(([source, detail]) => {
    const targets = detail?.targets || {};
    Object.entries(targets).forEach(([target, count]) => rows.push({ source, target, count }));
  });
  rows.sort((a, b) => b.count - a.count || a.source.localeCompare(b.source));
  const root = $("#decisionList");
  root.replaceChildren();
  if (!rows.length) {
    const empty = document.createElement("p");
    empty.textContent = "No routing decisions recorded yet.";
    root.append(empty);
    return;
  }
  rows.forEach(row => {
    const item = document.createElement("div");
    item.className = "decision-row";
    const source = document.createElement("strong");
    source.textContent = row.source;
    const target = document.createElement("span");
    target.textContent = row.target;
    const count = document.createElement("b");
    count.textContent = `${row.count} call${row.count === 1 ? "" : "s"}`;
    item.append(source, target, count);
    root.append(item);
  });
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  let payload = {};
  try { payload = await response.json(); } catch (_) { /* empty error body */ }
  if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
  return payload;
}

async function loadStatus() {
  $("#refreshRouting").disabled = true;
  try {
    const status = await request("/api/switchyard/status");
    $("#routeId").textContent = text(status.route, "Route not configured");
    if (!status.enabled) {
      setCard("#switchyardCard", "#switchyardState", "#switchyardDetail", "error", "Switchyard disabled", status.detail);
    } else if (!status.reachable) {
      setCard("#switchyardCard", "#switchyardState", "#switchyardDetail", "error", "Switchyard offline", status.detail);
    } else if (status.route_available === false) {
      setCard("#switchyardCard", "#switchyardState", "#switchyardDetail", "error", "Route unavailable", status.detail);
    } else {
      setCard("#switchyardCard", "#switchyardState", "#switchyardDetail", "ready", "Switchyard online", `${status.route} · ${status.base_url}`);
    }

    const efficient = endpointState(status.efficient_endpoint, "Qwen ready");
    setCard("#efficientCard", "#efficientState", "#efficientDetail", ...efficient);
    const capable = endpointState(status.capable_endpoint, "Nemotron ready");
    setCard("#capableCard", "#capableState", "#capableDetail", ...capable);

    const latest = status.latest_routing;
    $("#latestModel").textContent = text(latest?.model, "No routed request yet");
    $("#latestTier").textContent = text(latest?.tier);
    $("#latestTime").textContent = latest?.ts
      ? `${latest.ts} · ${text(latest.total_tokens, 0)} tokens`
      : "Routing records appear after a completed request.";
    renderDecisions(status.stage_router);
  } catch (error) {
    setCard("#switchyardCard", "#switchyardState", "#switchyardDetail", "error", "Status unavailable", error.message);
  } finally {
    $("#refreshRouting").disabled = false;
  }
}

async function runDiagnostic(button) {
  const scenario = button.dataset.scenario;
  const output = $("#diagnosticOutput");
  document.querySelectorAll(".diagnostic").forEach(item => { item.disabled = true; });
  output.textContent = `Routing ${scenario} trajectory…`;
  try {
    const result = await request(`/api/switchyard/diagnostics/${scenario}`, { method: "POST" });
    output.textContent = [
      `scenario=${result.scenario}`,
      `route=${result.route}`,
      `selected_model=${result.selected_model}`,
      `decision_source=${result.decision_sources.join(",") || "not isolated in stats delta"}`,
      `latency_ms=${result.latency_ms}`,
      result.response_preview ? `response=${result.response_preview}` : "",
    ].filter(Boolean).join("\n");
    await loadStatus();
  } catch (error) {
    output.textContent = `FAILED: ${error.message}`;
  } finally {
    document.querySelectorAll(".diagnostic").forEach(item => { item.disabled = false; });
  }
}

$("#refreshRouting").addEventListener("click", loadStatus);
document.querySelectorAll(".diagnostic").forEach(button => {
  button.addEventListener("click", () => runDiagnostic(button));
});
loadStatus();
setInterval(loadStatus, 15000);
