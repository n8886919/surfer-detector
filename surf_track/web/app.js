const ACTIONS = {
  w: { field: "chasing_wave", label: "追浪" },
  e: { field: "takeoff", label: "起乘" },
  r: { field: "surfing", label: "衝浪" },
};
const SHARER_MEMORY_KEY = "surftrack.lastSharerName";
const TRAIN_HOST_MEMORY_KEY = "surftrack.trainHost";

function rememberedSharerName() {
  try {
    return localStorage.getItem(SHARER_MEMORY_KEY) || "";
  } catch (_) {
    return "";
  }
}

function rememberSharerName(value) {
  try {
    const cleaned = value.trim();
    if (cleaned) localStorage.setItem(SHARER_MEMORY_KEY, cleaned);
    else localStorage.removeItem(SHARER_MEMORY_KEY);
  } catch (_) {
    // Browser storage can be disabled; dataset creation still works normally.
  }
}

function rememberedTrainHost() {
  try {
    return localStorage.getItem(TRAIN_HOST_MEMORY_KEY) || "";
  } catch (_) {
    return "";
  }
}

function rememberTrainHost(value) {
  try {
    const cleaned = value.trim();
    if (cleaned) localStorage.setItem(TRAIN_HOST_MEMORY_KEY, cleaned);
    else localStorage.removeItem(TRAIN_HOST_MEMORY_KEY);
  } catch (_) {
    // Browser storage can be disabled; the host just has to be retyped each session.
  }
}

const SELECTED_SHARERS_MEMORY_KEY = "surftrack.trainSharers";
const UNNAMED_SHARER = "舊資料未填分享者";

function rememberedSelectedSharers() {
  try {
    const parsed = JSON.parse(localStorage.getItem(SELECTED_SHARERS_MEMORY_KEY) || "[]");
    return new Set(Array.isArray(parsed) ? parsed.filter((name) => typeof name === "string") : []);
  } catch (_) {
    return new Set();
  }
}

function rememberSelectedSharers() {
  try {
    localStorage.setItem(SELECTED_SHARERS_MEMORY_KEY, JSON.stringify([...state.selectedSharers]));
  } catch (_) {
    // Browser storage can be disabled; the selection just resets next session.
  }
}

const state = {
  config: null,
  workspace: { datasets: [], jobs: [], training_runs: [] },
  selectedSharers: new Set(),
  sharerListSignature: null,
  scan: null,
  labelImages: [],
  currentImage: null,
  imageHistory: [],
  annotations: [],
  selectedBoxId: null,
  draft: null,
  resize: null,
  cropRegion: null,
  cropDraft: null,
  cropMode: false,
  saveTimer: null,
  loadedLabelDatasetId: null,
  loadedLabelImageCount: -1,
  workspaceLoading: false,
  zoomScale: 1,
  zoomX: 0,
  zoomY: 0,
  previewKey: null,
  previewLoading: false,
};

const elements = {
  navItems: [...document.querySelectorAll("[data-page]")],
  pages: [...document.querySelectorAll("[data-page-panel]")],
  driveForm: document.querySelector("#drive-form"),
  driveUrl: document.querySelector("#drive-url"),
  sourceNote: document.querySelector("#source-note"),
  sourceNoteText: document.querySelector("#source-note-text"),
  facebookGuard: document.querySelector("#facebook-guard"),
  clearUrl: document.querySelector("#clear-url"),
  scanButton: document.querySelector("#scan-button"),
  scanStatus: document.querySelector("#scan-status"),
  statusTitle: document.querySelector("#status-title"),
  statusMessage: document.querySelector("#status-message"),
  resultsPanel: document.querySelector("#results-panel"),
  sourceName: document.querySelector("#source-name"),
  videoCount: document.querySelector("#video-count"),
  totalSize: document.querySelector("#total-size"),
  videoTable: document.querySelector("#video-table"),
  nextStepPanel: document.querySelector("#next-step-panel"),
  nextStepSummary: document.querySelector("#next-step-summary"),
  datasetName: document.querySelector("#dataset-name"),
  sharerName: document.querySelector("#sharer-name"),
  createDatasetButton: document.querySelector("#create-dataset-button"),
  datasetCreateStatus: document.querySelector("#dataset-create-status"),
  privateDriveState: document.querySelector("#private-drive-state"),
  connectDriveButton: document.querySelector("#connect-drive-button"),
  driveSetupNote: document.querySelector("#drive-setup-note"),
  ingestDatasetSelect: document.querySelector("#ingest-dataset-select"),
  startIngestButton: document.querySelector("#start-ingest-button"),
  ingestHelp: document.querySelector("#ingest-help"),
  jobsList: document.querySelector("#jobs-list"),
  appVersion: document.querySelector("#app-version"),
  toast: document.querySelector("#toast"),
  labelDatasetSelect: document.querySelector("#label-dataset-select"),
  sharerList: document.querySelector("#sharer-list"),
  sharerSummary: document.querySelector("#sharer-summary"),
  trainHost: document.querySelector("#train-host"),
  startTrainButton: document.querySelector("#start-train-button"),
  trainStatus: document.querySelector("#train-status"),
  previewStatus: document.querySelector("#preview-status"),
  previewNote: document.querySelector("#preview-note"),
  predictionGrid: document.querySelector("#prediction-grid"),
  labelProgressCount: document.querySelector("#label-progress-count"),
  autosaveState: document.querySelector("#autosave-state"),
  labelImagePosition: document.querySelector("#label-image-position"),
  cropRegionButton: document.querySelector("#crop-region-button"),
  clearRegionButton: document.querySelector("#clear-region-button"),
  previousImageButton: document.querySelector("#previous-image-button"),
  randomImageButton: document.querySelector("#random-image-button"),
  labelStage: document.querySelector("#label-stage"),
  labelEmpty: document.querySelector("#label-empty"),
  imageFrame: document.querySelector("#image-frame"),
  labelImage: document.querySelector("#label-image"),
  annotationLayer: document.querySelector("#annotation-layer"),
  selectedBoxStatus: document.querySelector("#selected-box-status"),
  actionButtons: [...document.querySelectorAll("[data-action]")],
};

function selectPage(pageName) {
  elements.navItems.forEach((item) => item.classList.toggle("active", item.dataset.page === pageName));
  elements.pages.forEach((page) => {
    const active = page.dataset.pagePanel === pageName;
    page.classList.toggle("active", active);
    page.hidden = !active;
  });
  history.replaceState(null, "", `#${pageName}`);
  if (pageName === "label") elements.labelStage.focus({ preventScroll: true });
  if (pageName === "train") loadTrainingPreview();
}

function setStatus(kind, title, message) {
  elements.scanStatus.dataset.state = kind;
  elements.statusTitle.textContent = title;
  elements.statusMessage.textContent = message;
  elements.scanStatus.querySelector(".status-icon").textContent =
    kind === "error" ? "!" : kind === "loading" ? "…" : kind === "warning" ? "⌾" : "✓";
}

function setLoading(loading) {
  elements.scanButton.disabled = loading;
  elements.scanButton.classList.toggle("loading", loading);
  elements.scanButton.querySelector("span").textContent = loading ? "正在檢查…" : "檢查來源";
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (!value) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** index).toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
}

function formatDuration(milliseconds) {
  if (milliseconds === null || milliseconds === undefined) return "—";
  const seconds = Math.floor(Number(milliseconds) / 1000);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
    : `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => elements.toast.classList.remove("visible"), 4200);
}

function renderScan(result) {
  state.scan = result;
  elements.nextStepPanel.hidden = true;
  if (result.state === "link_ready") {
    elements.resultsPanel.classList.add("hidden");
    setStatus("ready", "公開連結已辨識", result.message);
    showToast("連結已接收；安裝專案依賴後即可直接列出公開影片。");
    return;
  }
  if (result.state === "auth_required") {
    elements.resultsPanel.classList.add("hidden");
    elements.facebookGuard.hidden = false;
    elements.driveUrl.value = result.source.canonical_url;
    const removed = Number(result.source?.tracking_parameters_removed || 0);
    setStatus("warning", "Facebook 貼文已辨識", `${result.message} 已清除 ${removed} 個非必要網址參數。`);
    showToast("沒有讀取密碼、cookie 或 Facebook 貼文內容。");
    return;
  }

  const videos = result.videos || [];
  const summary = result.summary || {};
  elements.sourceName.textContent = result.source?.name || "Google Drive";
  elements.videoCount.textContent = String(summary.video_count || 0);
  elements.totalSize.textContent = formatBytes(summary.total_bytes);
  elements.videoTable.replaceChildren();

  for (const video of videos) {
    const row = document.createElement("tr");
    const resolution = video.width && video.height ? `${video.width} × ${video.height}` : "—";
    const values = [video.name, resolution, formatDuration(video.duration_ms), formatBytes(video.size_bytes)];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    }
    elements.videoTable.appendChild(row);
  }

  if (!videos.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.textContent = "這個來源中沒有找到支援的影片檔。";
    row.appendChild(cell);
    elements.videoTable.appendChild(row);
  }

  elements.resultsPanel.classList.remove("hidden");
  setStatus("ready", `掃描完成 · ${videos.length} 支影片`, `已檢查 ${summary.items_scanned || 0} 個來源項目。`);
  if (result.state === "scanned" && videos.length) {
    elements.datasetName.value = result.source?.name || `surf-dataset-${new Date().toISOString().slice(0, 10)}`;
    elements.sharerName.value = rememberedSharerName();
    elements.nextStepSummary.textContent = `${videos.length} 支影片已找到。現在只需要填分享者／來源名稱。`;
    elements.datasetCreateStatus.textContent = "請先填分享者。";
    elements.nextStepPanel.hidden = false;
    updateDatasetCreateState();
    elements.nextStepPanel.scrollIntoView({ behavior: "smooth", block: "center" });
    window.setTimeout(() => elements.sharerName.focus({ preventScroll: true }), 250);
  }
}

function updateSourceHint() {
  const isFacebook = /(^|\.)facebook\.com\//i.test(elements.driveUrl.value.trim().replace(/^https?:\/\//i, ""));
  elements.facebookGuard.hidden = !isFacebook;
  elements.sourceNoteText.textContent = isFacebook
    ? "Facebook 來源會先清除追蹤參數，再要求安全登入。"
    : "Drive 必須設為知道連結的任何人都能檢視及下載。";
}

function updateDatasetCreateState() {
  const ready = state.scan?.state === "scanned" &&
    !state.scan?.datasetCreated &&
    Number(state.scan?.summary?.video_count || 0) > 0 &&
    Boolean(elements.datasetName.value.trim()) &&
    Boolean(elements.sharerName.value.trim());
  elements.createDatasetButton.disabled = !ready;
  if (ready) elements.datasetCreateStatus.textContent = "可以建立。";
  else if (!elements.sharerName.value.trim()) elements.datasetCreateStatus.textContent = "請先填分享者。";
  else elements.datasetCreateStatus.textContent = "Dataset 名稱不可為空。";
}

async function createDataset() {
  updateDatasetCreateState();
  if (elements.createDatasetButton.disabled) return;
  const source = state.scan?.source || {};
  elements.createDatasetButton.disabled = true;
  elements.createDatasetButton.textContent = "建立中…";
  elements.datasetCreateStatus.textContent = "正在保存來源資訊。";
  try {
    const response = await fetch("/api/v1/datasets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: elements.datasetName.value.trim(),
        source_provider: source.provider,
        source_url: source.canonical_url,
        source_title: source.name || "",
        sharer_name: elements.sharerName.value.trim(),
        source_video_count: Number(state.scan?.summary?.video_count || 0),
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error?.message || "無法建立 Dataset。");
    rememberSharerName(elements.sharerName.value);
    state.scan.datasetCreated = true;
    elements.createDatasetButton.textContent = "Dataset 已建立";
    elements.datasetCreateStatus.textContent = state.config?.private_drive_connected
      ? "下一步：到下方按「開始處理」。"
      : "下一步：連接私人 Drive。";
    await loadWorkspace();
    showToast(`已建立 ${payload.dataset.name}，分享者：${payload.dataset.sharer_name}`);
  } catch (error) {
    elements.createDatasetButton.textContent = "建立 Dataset";
    elements.datasetCreateStatus.textContent = error.message;
    updateDatasetCreateState();
  }
}

async function inspectSource(event) {
  event.preventDefault();
  const url = elements.driveUrl.value.trim();
  if (!url) {
    setStatus("error", "缺少來源連結", "請先貼上 Drive 或 Facebook 連結。");
    elements.driveUrl.focus();
    return;
  }

  setLoading(true);
  setStatus("loading", "正在檢查來源", "辨識連結類型並套用安全規則…");
  elements.resultsPanel.classList.add("hidden");
  try {
    const response = await fetch("/api/v1/sources/inspect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error?.message || "檢查失敗，請稍後再試。");
    renderScan(payload);
  } catch (error) {
    setStatus("error", "無法檢查這個來源", error.message);
  } finally {
    setLoading(false);
  }
}

function jobStateLabel(job) {
  return {
    queued: "等待中",
    running: "處理中",
    paused: "可繼續",
    completed: "完成",
    failed: "失敗",
  }[job.state] || job.state;
}

function sharerLabel(dataset) {
  return dataset.sharer_name || UNNAMED_SHARER;
}

function sharerGroups() {
  const groups = new Map();
  for (const dataset of state.workspace.datasets) {
    const name = sharerLabel(dataset);
    const group = groups.get(name) || { name, datasets: 0, images: 0, labeled: 0 };
    group.datasets += 1;
    group.images += Number(dataset.image_count || 0);
    group.labeled += Number(dataset.labeled_image_count || 0);
    groups.set(name, group);
  }
  return [...groups.values()].sort((first, second) => second.labeled - first.labeled);
}

function updateSharerSummary() {
  const chosen = sharerGroups().filter((group) => state.selectedSharers.has(group.name));
  const labeled = chosen.reduce((total, group) => total + group.labeled, 0);
  elements.sharerSummary.textContent = chosen.length
    ? `已選 ${chosen.length} 位 · ${labeled} 張已標圖片`
    : "尚未勾選分享者";
}

function renderSharerList() {
  const groups = sharerGroups();
  // loadWorkspace polls every 2s; only rebuild when the sharers actually changed, so a
  // click is never swallowed by a re-render.
  const signature = groups.map((g) => `${g.name}:${g.datasets}:${g.images}:${g.labeled}`).join("|");
  if (signature === state.sharerListSignature) {
    updateSharerSummary();
    return;
  }
  state.sharerListSignature = signature;
  for (const name of [...state.selectedSharers]) {
    if (!groups.some((group) => group.name === name)) state.selectedSharers.delete(name);
  }

  elements.sharerList.replaceChildren();
  if (!groups.length) {
    const empty = document.createElement("div");
    empty.className = "empty-row";
    empty.innerHTML = "<strong>尚未建立 Dataset</strong><small>先在 Data 頁建立並處理資料集。</small>";
    elements.sharerList.appendChild(empty);
    updateSharerSummary();
    return;
  }

  for (const group of groups) {
    const row = document.createElement("label");
    row.className = "sharer-row";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = state.selectedSharers.has(group.name);
    row.classList.toggle("checked", box.checked);
    box.addEventListener("change", () => {
      if (box.checked) state.selectedSharers.add(group.name);
      else state.selectedSharers.delete(group.name);
      row.classList.toggle("checked", box.checked);
      rememberSelectedSharers();
      updateSharerSummary();
      renderLatestTrainingRun();
    });
    const text = document.createElement("div");
    const strong = document.createElement("strong");
    strong.textContent = group.name;
    const small = document.createElement("small");
    small.textContent = `${group.datasets} 個 Dataset · ${group.labeled}/${group.images} 張已標`;
    text.append(strong, small);
    const count = document.createElement("span");
    count.className = "job-state";
    count.textContent = `${group.labeled} 張`;
    row.append(box, text, count);
    elements.sharerList.appendChild(row);
  }
  updateSharerSummary();
}

function renderJobs() {
  const datasetNames = new Map(state.workspace.datasets.map((dataset) => [dataset.id, dataset.name]));
  elements.jobsList.replaceChildren();
  if (!state.workspace.jobs.length) {
    const empty = document.createElement("div");
    empty.className = "empty-row";
    empty.innerHTML = "<strong>目前沒有處理工作</strong><small>建立處理工作後，這裡會顯示下載、抽圖與上傳進度。</small>";
    elements.jobsList.appendChild(empty);
    return;
  }

  for (const job of state.workspace.jobs) {
    const percentage = Math.round(Number(job.progress || 0) * 100);
    const row = document.createElement("div");
    row.className = "job-row";
    const name = document.createElement("div");
    name.className = "job-name";
    const strong = document.createElement("strong");
    strong.textContent = datasetNames.get(job.dataset_id)
      || (job.dataset_id ? "（資料集已刪除）" : "（舊工作未記錄資料集）");
    const small = document.createElement("small");
    const kind = job.job_type === "ingest" ? "下載與抽圖" : job.job_type;
    small.textContent = `${kind} · ${job.message || job.step}`;
    name.append(strong, small);
    const progress = document.createElement("div");
    progress.innerHTML = `<div class="job-progress-track"><i style="width:${percentage}%"></i></div><div class="job-progress-meta"><span>${job.completed_items} / ${job.total_items}</span><span>${percentage}%</span></div>`;
    const status = document.createElement("span");
    status.className = "job-state";
    status.textContent = jobStateLabel(job);
    row.append(name, progress, status);
    elements.jobsList.appendChild(row);
  }
}

function populateDatasetSelectors() {
  const datasets = state.workspace.datasets;
  if (!rememberedSharerName() && datasets[0]?.sharer_name) {
    rememberSharerName(datasets[0].sharer_name);
  }
  const previousLabel = elements.labelDatasetSelect.value;
  elements.labelDatasetSelect.replaceChildren();
  if (!datasets.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "尚未建立 Dataset";
    elements.labelDatasetSelect.appendChild(option);
    elements.labelDatasetSelect.disabled = true;
  } else {
    elements.labelDatasetSelect.disabled = false;
    for (const dataset of datasets) {
      const total = Number(dataset.image_count || 0);
      const labeled = Number(dataset.labeled_image_count || 0);
      const percentage = total ? Math.round(labeled / total * 100) : 0;
      const labelOption = document.createElement("option");
      labelOption.value = dataset.id;
      labelOption.textContent = `${dataset.name} · ${sharerLabel(dataset)} · ${labeled}/${total} 已標 (${percentage}%)`;
      elements.labelDatasetSelect.appendChild(labelOption);
    }
  }
  if (datasets.some((dataset) => dataset.id === previousLabel)) elements.labelDatasetSelect.value = previousLabel;

  const previousIngest = elements.ingestDatasetSelect.value;
  elements.ingestDatasetSelect.replaceChildren();
  if (!datasets.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "尚未建立 Dataset";
    elements.ingestDatasetSelect.appendChild(option);
    elements.ingestDatasetSelect.disabled = true;
  } else {
    elements.ingestDatasetSelect.disabled = false;
    for (const dataset of datasets) {
      const option = document.createElement("option");
      option.value = dataset.id;
      option.textContent = `${dataset.name} · ${dataset.source_video_count} 支影片`;
      elements.ingestDatasetSelect.appendChild(option);
    }
    if (datasets.some((dataset) => dataset.id === previousIngest)) {
      elements.ingestDatasetSelect.value = previousIngest;
    }
  }

  const selectedLabelDataset = datasets.find((dataset) => dataset.id === elements.labelDatasetSelect.value);
  if (selectedLabelDataset && (
    state.loadedLabelDatasetId !== selectedLabelDataset.id ||
    (state.loadedLabelImageCount === 0 && Number(selectedLabelDataset.image_count) > 0)
  )) {
    loadLabelImages(selectedLabelDataset.id);
  }
  updateIngestButton();
}

function updateIngestButton() {
  const datasetId = elements.ingestDatasetSelect.value;
  const dataset = state.workspace.datasets.find((item) => item.id === datasetId);
  const job = state.workspace.jobs.find((item) => item.dataset_id === datasetId && item.job_type === "ingest");
  if (!dataset) {
    elements.startIngestButton.disabled = true;
    elements.startIngestButton.textContent = "開始處理";
    return;
  }
  if (dataset.source_provider !== "google_drive") {
    elements.startIngestButton.disabled = true;
    elements.startIngestButton.textContent = "尚未支援 Facebook";
    return;
  }
  if (!state.config?.private_drive_connected) {
    elements.startIngestButton.disabled = true;
    elements.startIngestButton.textContent = "請先連接 Drive";
    return;
  }
  const labels = {
    queued: "準備中…",
    running: "處理中…",
    paused: "繼續處理",
    failed: "重試處理",
    completed: "處理完成",
  };
  elements.startIngestButton.textContent = labels[job?.state] || "開始處理";
  elements.startIngestButton.disabled = ["queued", "running", "completed"].includes(job?.state);
}

async function startIngest() {
  const datasetId = elements.ingestDatasetSelect.value;
  if (!datasetId || elements.startIngestButton.disabled) return;
  elements.startIngestButton.disabled = true;
  elements.startIngestButton.textContent = "啟動中…";
  try {
    const response = await fetch(`/api/v1/datasets/${datasetId}/ingest`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error?.message || "無法開始處理。");
    showToast("已開始處理；可以關閉頁面，之後再回來看進度。");
    await loadWorkspace();
  } catch (error) {
    showToast(error.message);
    updateIngestButton();
  }
}

function renderLatestTrainingRun() {
  const run = state.workspace.training_runs[0];
  elements.startTrainButton.disabled = !state.selectedSharers.size
    || !elements.trainHost.value.trim()
    || ["queued", "running"].includes(run?.state);
  elements.startTrainButton.textContent = ["queued", "running"].includes(run?.state)
    ? `訓練中 ${run.epoch} / ${run.total_epochs}`
    : "Train 小模型";
  if (!run) return;
  const metrics = run.metrics || {};
  const stateLabels = { queued: "等待開始", running: `訓練中 ${run.epoch} / ${run.total_epochs}`, completed: "完成", failed: "失敗" };
  document.querySelector("#latest-run-state").textContent = stateLabels[run.state] || run.state;
  if (metrics.error) elements.trainStatus.textContent = metrics.error;
  const mappings = {
    "#metric-recall": metrics.detector_recall,
    "#metric-map50": metrics.detector_map50,
    "#metric-action-f1": metrics.action_macro_f1,
    "#metric-hz": metrics.orin_hz,
    "#metric-chasing": metrics.chasing_wave_f1,
    "#metric-takeoff": metrics.takeoff_f1,
    "#metric-surfing": metrics.surfing_f1,
  };
  for (const [selector, value] of Object.entries(mappings)) {
    if (value === undefined || value === null) continue;
    document.querySelector(selector).textContent = selector === "#metric-hz" ? `${Number(value).toFixed(1)} Hz` : Number(value).toFixed(2);
  }
  const actionValues = [metrics.chasing_wave_f1, metrics.takeoff_f1, metrics.surfing_f1];
  document.querySelectorAll(".action-metrics progress").forEach((bar, index) => {
    bar.value = Number(actionValues[index] || 0);
  });
}

async function startTraining() {
  const host = elements.trainHost.value.trim();
  const sharers = [...state.selectedSharers];
  if (!sharers.length || !host || elements.startTrainButton.disabled) return;
  elements.startTrainButton.disabled = true;
  elements.startTrainButton.textContent = "啟動中…";
  try {
    const response = await fetch("/api/v1/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ host, sharers }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error?.message || "無法開始訓練。");
    rememberTrainHost(host);
    elements.trainStatus.textContent = `已在 ${host} 的 GPU 上背景訓練，可以切到其他頁面。`;
    state.previewKey = null;
    await loadWorkspace();
  } catch (error) {
    elements.trainStatus.textContent = error.message;
    showToast(error.message);
    elements.startTrainButton.disabled = false;
    elements.startTrainButton.textContent = "Train 小模型";
  }
}

function renderTrainingPreview(payload) {
  elements.predictionGrid.replaceChildren();
  elements.previewNote.textContent = payload.message;
  for (const item of payload.images) {
    const card = document.createElement("figure");
    card.className = "prediction-card";
    const imageWrap = document.createElement("div");
    imageWrap.className = "prediction-image";
    const image = document.createElement("img");
    image.src = item.image_url;
    image.alt = `${item.split} 辨識預覽`;
    image.loading = "lazy";
    imageWrap.appendChild(image);
    for (const box of item.boxes) {
      const element = document.createElement("div");
      element.className = "prediction-box";
      element.style.left = `${Number(box.x) * 100}%`;
      element.style.top = `${Number(box.y) * 100}%`;
      element.style.width = `${Number(box.width) * 100}%`;
      element.style.height = `${Number(box.height) * 100}%`;
      const label = document.createElement("span");
      const probability = box.probabilities;
      label.textContent = `追${Math.round(probability.chasing_wave * 100)} 起${Math.round(probability.takeoff * 100)} 衝${Math.round(probability.surfing * 100)}`;
      element.appendChild(label);
      imageWrap.appendChild(element);
    }
    const caption = document.createElement("figcaption");
    caption.append(item.split.toUpperCase(), item.image_id.slice(-6));
    card.append(imageWrap, caption);
    elements.predictionGrid.appendChild(card);
  }
  elements.previewStatus.textContent = `${payload.images.length} 張`;
}

async function loadTrainingPreview() {
  const run = state.workspace.training_runs.find((item) => item.state === "completed");
  if (!run || state.previewLoading) return;
  const key = run.id;
  if (state.previewKey === key) return;
  state.previewLoading = true;
  elements.previewStatus.textContent = "辨識中…";
  try {
    const response = await fetch("/api/v1/training-preview");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error?.message || "無法產生辨識預覽。");
    renderTrainingPreview(payload);
    state.previewKey = key;
  } catch (error) {
    elements.previewStatus.textContent = "無法顯示";
    elements.previewNote.textContent = error.message;
  } finally {
    state.previewLoading = false;
  }
}

async function loadWorkspace() {
  if (state.workspaceLoading) return;
  state.workspaceLoading = true;
  try {
    const response = await fetch("/api/v1/workspace");
    if (!response.ok) throw new Error("workspace unavailable");
    state.workspace = await response.json();
    renderJobs();
    populateDatasetSelectors();
    renderSharerList();
    renderLatestTrainingRun();
    if (document.querySelector("#train-page").classList.contains("active")) loadTrainingPreview();
  } catch (_) {
    showToast("無法載入本機工作狀態。");
  } finally {
    state.workspaceLoading = false;
  }
}

async function loadConfig() {
  try {
    const response = await fetch("/api/v1/config");
    if (!response.ok) return;
    state.config = await response.json();
    elements.appVersion.textContent = `v${state.config.version}`;
    const connected = Boolean(state.config.private_drive_connected);
    elements.privateDriveState.textContent = connected ? "已連接" : "未連接";
    elements.privateDriveState.classList.toggle("connected", connected);
    elements.connectDriveButton.textContent = connected ? "Drive 已連接" : "連接狀態";
    elements.connectDriveButton.disabled = connected;
    if (connected) elements.driveSetupNote.hidden = true;
    updateIngestButton();
    if (!state.config.drive_scan_configured) showToast("UI 已就緒；目前是 Drive 連結驗證模式。");
  } catch (_) {
    setStatus("error", "服務未連線", "請確認 SurfTrack Python 服務仍在執行。");
  }
}

/* Label UI */
function newBoxId() {
  return typeof crypto.randomUUID === "function" ? `box_${crypto.randomUUID()}` : `box_${Date.now()}_${Math.random()}`;
}

function selectedAnnotation() {
  return state.annotations.find((box) => box.id === state.selectedBoxId) || null;
}

function actionLabels(box) {
  return Object.values(ACTIONS).filter((action) => box[action.field]).map((action) => action.label);
}

function hasAction(box) {
  return Object.values(ACTIONS).some((action) => Boolean(box[action.field]));
}

function hasIncompleteBoxes() {
  return state.annotations.some((box) => !hasAction(box));
}

function renderActionState() {
  const selected = selectedAnnotation();
  for (const button of elements.actionButtons) {
    button.classList.toggle("active", Boolean(selected?.[button.dataset.action]));
  }
  if (!selected) {
    elements.selectedBoxStatus.textContent = "尚未選取人物框";
    return;
  }
  const index = state.annotations.findIndex((box) => box.id === selected.id) + 1;
  const labels = actionLabels(selected);
  elements.selectedBoxStatus.textContent = `框 ${index} · ${labels.length ? labels.join(" + ") : "尚未設定狀態"}`;
}

function makeBoxElement(box, { draft = false } = {}) {
  const element = document.createElement("button");
  element.type = "button";
  element.className = `annotation-box${box.id === state.selectedBoxId ? " selected" : ""}${draft ? " draft" : ""}`;
  element.style.left = `${box.x * 100}%`;
  element.style.top = `${box.y * 100}%`;
  element.style.width = `${box.width * 100}%`;
  element.style.height = `${box.height * 100}%`;
  if (!draft) {
    const labels = actionLabels(box);
    if (labels.length) {
      const label = document.createElement("span");
      label.className = "annotation-box-label";
      label.textContent = labels.join(" + ");
      element.appendChild(label);
    }
    for (const handle of ["n", "ne", "e", "se", "s", "sw", "w", "nw"]) {
      const resizeHandle = document.createElement("span");
      resizeHandle.className = "resize-handle";
      resizeHandle.dataset.handle = handle;
      resizeHandle.addEventListener("pointerdown", (event) => beginResize(event, box.id, handle));
      element.appendChild(resizeHandle);
    }
    element.addEventListener("pointerdown", (event) => event.stopPropagation());
    element.addEventListener("click", (event) => {
      event.stopPropagation();
      state.selectedBoxId = box.id;
      renderAnnotations();
      elements.labelStage.focus({ preventScroll: true });
    });
  }
  return element;
}

function makeCropRegionElement(region, { draft = false } = {}) {
  const element = document.createElement("div");
  element.className = `crop-region${draft ? " draft" : ""}`;
  element.style.left = `${region.x * 100}%`;
  element.style.top = `${region.y * 100}%`;
  element.style.width = `${region.width * 100}%`;
  element.style.height = `${region.height * 100}%`;
  return element;
}

function renderAnnotations() {
  elements.annotationLayer.replaceChildren();
  if (state.cropRegion && !state.cropDraft) elements.annotationLayer.appendChild(makeCropRegionElement(state.cropRegion));
  if (state.cropDraft) elements.annotationLayer.appendChild(makeCropRegionElement(state.cropDraft.box, { draft: true }));
  for (const box of state.annotations) elements.annotationLayer.appendChild(makeBoxElement(box));
  if (state.draft) elements.annotationLayer.appendChild(makeBoxElement(state.draft.box, { draft: true }));
  renderActionState();
}

function pointInsideCrop(point) {
  if (!state.cropRegion) return true;
  return point.x >= state.cropRegion.x && point.y >= state.cropRegion.y &&
    point.x <= state.cropRegion.x + state.cropRegion.width &&
    point.y <= state.cropRegion.y + state.cropRegion.height;
}

function clampPointToCrop(point) {
  if (!state.cropRegion) return point;
  return {
    x: Math.min(state.cropRegion.x + state.cropRegion.width, Math.max(state.cropRegion.x, point.x)),
    y: Math.min(state.cropRegion.y + state.cropRegion.height, Math.max(state.cropRegion.y, point.y)),
  };
}

function pointerPosition(event) {
  const rect = elements.annotationLayer.getBoundingClientRect();
  return {
    x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
    y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
  };
}

function beginBox(event) {
  if (!state.currentImage || event.button !== 0 || event.target !== elements.annotationLayer) return;
  event.preventDefault();
  const start = pointerPosition(event);
  if (state.cropMode) {
    state.cropDraft = {
      pointerId: event.pointerId,
      start,
      box: { x: start.x, y: start.y, width: 0, height: 0 },
    };
    elements.annotationLayer.setPointerCapture(event.pointerId);
    renderAnnotations();
    return;
  }
  if (!pointInsideCrop(start)) {
    showToast("bbox 必須畫在選定區域內。");
    return;
  }
  state.draft = {
    pointerId: event.pointerId,
    start,
    box: { id: "draft", x: start.x, y: start.y, width: 0, height: 0 },
  };
  elements.annotationLayer.setPointerCapture(event.pointerId);
  renderAnnotations();
}

function beginResize(event, boxId, handle) {
  const box = state.annotations.find((item) => item.id === boxId);
  if (!box || event.button !== 0) return;
  event.preventDefault();
  event.stopPropagation();
  state.selectedBoxId = boxId;
  state.resize = {
    pointerId: event.pointerId,
    boxId,
    handle,
    original: { x: box.x, y: box.y, width: box.width, height: box.height },
  };
  elements.annotationLayer.setPointerCapture(event.pointerId);
  renderAnnotations();
}

function resizeBox(event) {
  const operation = state.resize;
  if (!operation || event.pointerId !== operation.pointerId) return false;
  const box = state.annotations.find((item) => item.id === operation.boxId);
  if (!box) return false;
  const point = clampPointToCrop(pointerPosition(event));
  const original = operation.original;
  const right = original.x + original.width;
  const bottom = original.y + original.height;
  const minimum = 0.008;
  if (operation.handle.includes("w")) {
    box.x = Math.min(point.x, right - minimum);
    box.width = right - box.x;
  }
  if (operation.handle.includes("e")) {
    box.width = Math.max(minimum, point.x - original.x);
  }
  if (operation.handle.includes("n")) {
    box.y = Math.min(point.y, bottom - minimum);
    box.height = bottom - box.y;
  }
  if (operation.handle.includes("s")) {
    box.height = Math.max(minimum, point.y - original.y);
  }
  renderAnnotations();
  return true;
}

function moveBox(event) {
  if (state.cropDraft && event.pointerId === state.cropDraft.pointerId) {
    const point = pointerPosition(event);
    state.cropDraft.box = {
      x: Math.min(state.cropDraft.start.x, point.x),
      y: Math.min(state.cropDraft.start.y, point.y),
      width: Math.abs(point.x - state.cropDraft.start.x),
      height: Math.abs(point.y - state.cropDraft.start.y),
    };
    renderAnnotations();
    return;
  }
  if (resizeBox(event)) return;
  if (!state.draft || event.pointerId !== state.draft.pointerId) return;
  const point = clampPointToCrop(pointerPosition(event));
  state.draft.box = {
    id: "draft",
    x: Math.min(state.draft.start.x, point.x),
    y: Math.min(state.draft.start.y, point.y),
    width: Math.abs(point.x - state.draft.start.x),
    height: Math.abs(point.y - state.draft.start.y),
  };
  renderAnnotations();
}

function finishBox(event) {
  if (state.cropDraft && event.pointerId === state.cropDraft.pointerId) {
    const crop = state.cropDraft.box;
    state.cropDraft = null;
    if (crop.width >= 0.02 && crop.height >= 0.02) {
      state.cropRegion = crop;
      state.cropMode = false;
      updateCropControls();
      queueSave();
    } else {
      showToast("區域太小，請重新拖曳。");
    }
    renderAnnotations();
    return;
  }
  if (state.resize && event.pointerId === state.resize.pointerId) {
    state.resize = null;
    queueSave();
    renderAnnotations();
    return;
  }
  if (!state.draft || event.pointerId !== state.draft.pointerId) return;
  const box = state.draft.box;
  state.draft = null;
  if (box.width >= 0.008 && box.height >= 0.008) {
    const annotation = {
      ...box,
      id: newBoxId(),
      chasing_wave: false,
      takeoff: false,
      surfing: false,
    };
    state.annotations.push(annotation);
    state.selectedBoxId = annotation.id;
    setAutosaveState("saved", "等待 W/E/R");
  }
  renderAnnotations();
}

function toggleAction(field) {
  const selected = selectedAnnotation();
  if (!selected) {
    showToast("請先拉一個框或點選既有框。");
    return;
  }
  selected[field] = !selected[field];
  renderAnnotations();
  queueSave();
}

function setAutosaveState(mode, text) {
  elements.autosaveState.classList.toggle("saving", mode === "saving");
  elements.autosaveState.classList.toggle("error", mode === "error");
  elements.autosaveState.lastChild.textContent = ` ${text}`;
}

function queueSave() {
  window.clearTimeout(state.saveTimer);
  setAutosaveState("saving", "保存中…");
  state.saveTimer = window.setTimeout(saveCurrentAnnotations, 350);
}

async function saveCurrentAnnotations() {
  window.clearTimeout(state.saveTimer);
  if (!state.currentImage) return true;
  setAutosaveState("saving", "保存中…");
  try {
    const response = await fetch(`/api/v1/images/${state.currentImage.id}/annotations`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        annotations: state.annotations.filter(hasAction),
        crop_region: state.cropRegion,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error?.message || "保存失敗");
    state.currentImage.annotation_count = payload.annotation_count;
    state.currentImage.complete_for_detection = payload.complete_for_detection;
    state.currentImage.reviewed = true;
    state.cropRegion = payload.crop_region;
    setAutosaveState("saved", hasIncompleteBoxes() ? "等待 W/E/R" : "已保存");
    updateLabelProgress();
    return true;
  } catch (error) {
    setAutosaveState("error", "保存失敗");
    showToast(error.message);
    return false;
  }
}

function updateLabelProgress() {
  const labeled = state.labelImages.filter((image) => Number(image.annotation_count) > 0).length;
  const reviewed = state.labelImages.filter((image) => image.reviewed || Number(image.annotation_count) > 0).length;
  elements.labelProgressCount.textContent = `${labeled} 已標 · ${reviewed} 已看 / ${state.labelImages.length}`;
  if (!state.currentImage) {
    elements.labelImagePosition.textContent = "— / —";
    return;
  }
  const index = state.labelImages.findIndex((image) => image.id === state.currentImage.id);
  elements.labelImagePosition.textContent = `${index + 1} / ${state.labelImages.length}`;
}

function updateImageNavigation() {
  const hasImages = state.labelImages.length > 0;
  elements.randomImageButton.disabled = !hasImages;
  elements.previousImageButton.disabled = state.imageHistory.length === 0;
  elements.cropRegionButton.disabled = !hasImages;
  updateCropControls();
}

function updateCropControls() {
  elements.cropRegionButton.classList.toggle("active", state.cropMode);
  elements.cropRegionButton.textContent = state.cropMode ? "拖曳區域…" : "區域";
  elements.clearRegionButton.disabled = !state.currentImage || !state.cropRegion;
}

function toggleCropMode() {
  if (!state.currentImage) return;
  if (state.annotations.length) {
    showToast("重選區域前請先刪除這張圖的 bbox。");
    return;
  }
  state.cropMode = !state.cropMode;
  state.cropDraft = null;
  updateCropControls();
  renderAnnotations();
  elements.labelStage.focus({ preventScroll: true });
}

function clearCropRegion() {
  if (state.annotations.length) {
    showToast("恢復全圖前請先刪除這張圖的 bbox。");
    return;
  }
  state.cropRegion = null;
  state.cropMode = false;
  state.cropDraft = null;
  updateCropControls();
  renderAnnotations();
  queueSave();
}

function applyZoom() {
  const inverse = 1 / state.zoomScale;
  elements.imageFrame.style.setProperty("--zoom-border", `${inverse}px`);
  elements.imageFrame.style.setProperty("--zoom-handle-edge", `${10 * inverse}px`);
  elements.imageFrame.style.setProperty("--zoom-handle-edge-offset", `${-5 * inverse}px`);
  elements.imageFrame.style.setProperty("--zoom-handle-corner", `${12 * inverse}px`);
  elements.imageFrame.style.setProperty("--zoom-handle-corner-offset", `${-6 * inverse}px`);
  elements.imageFrame.style.setProperty("--zoom-label-top", `${-20 * inverse}px`);
  elements.imageFrame.style.setProperty("--zoom-label-left", `${-1 * inverse}px`);
  elements.imageFrame.style.setProperty("--zoom-label-padding-y", `${2 * inverse}px`);
  elements.imageFrame.style.setProperty("--zoom-label-padding-x", `${4 * inverse}px`);
  elements.imageFrame.style.setProperty("--zoom-label-font", `${11 * inverse}px`);
  elements.imageFrame.style.transform =
    `translate(${state.zoomX}px, ${state.zoomY}px) scale(${state.zoomScale})`;
}

function resetZoom() {
  state.zoomScale = 1;
  state.zoomX = 0;
  state.zoomY = 0;
  applyZoom();
}

async function openLabelImage(imageId, { remember = true, saveCurrent = true } = {}) {
  const image = state.labelImages.find((candidate) => candidate.id === imageId);
  if (!image) return;
  if (state.currentImage && state.currentImage.id !== image.id && hasIncompleteBoxes()) {
    showToast("每個 bbox 至少要選一個 W/E/R；不要的框請用 Delete 刪除。");
    return false;
  }
  if (saveCurrent && !(await saveCurrentAnnotations())) return false;
  if (remember && state.currentImage && state.currentImage.id !== image.id) {
    state.imageHistory.push(state.currentImage.id);
  }
  updateImageNavigation();
  state.currentImage = image;
  state.annotations = [];
  state.selectedBoxId = null;
  state.resize = null;
  state.cropRegion = null;
  state.cropDraft = null;
  state.cropMode = false;
  updateCropControls();
  resetZoom();
  elements.labelImage.onload = () => {
    elements.labelEmpty.hidden = true;
    elements.imageFrame.hidden = false;
    renderAnnotations();
  };
  elements.labelImage.src = image.image_url;
  try {
    const response = await fetch(`/api/v1/images/${image.id}/annotations`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error?.message || "無法載入標注");
    state.annotations = payload.annotations || [];
    state.cropRegion = payload.crop_region || null;
    updateCropControls();
    renderAnnotations();
    updateLabelProgress();
    return true;
  } catch (error) {
    showToast(error.message);
    return false;
  }
}

async function loadLabelImages(datasetId) {
  state.loadedLabelDatasetId = datasetId;
  const dataset = state.workspace.datasets.find((item) => item.id === datasetId);
  state.loadedLabelImageCount = Number(dataset?.image_count || 0);
  state.currentImage = null;
  state.annotations = [];
  state.imageHistory = [];
  updateImageNavigation();
  resetZoom();
  elements.imageFrame.hidden = true;
  elements.labelEmpty.hidden = false;
  try {
    const response = await fetch(`/api/v1/datasets/${datasetId}/images`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error?.message || "無法載入影格");
    state.labelImages = payload.images || [];
    updateImageNavigation();
    updateLabelProgress();
    if (state.labelImages.length) await openLabelImage(state.labelImages[0].id, { remember: false });
  } catch (error) {
    state.labelImages = [];
    updateImageNavigation();
    updateLabelProgress();
    showToast(error.message);
  }
}

async function randomNextImage({ saveCurrent = true } = {}) {
  if (!state.labelImages.length) return;
  if (hasIncompleteBoxes()) {
    showToast("每個 bbox 至少要選一個 W/E/R；不要的框請用 Delete 刪除。");
    return;
  }
  const candidates = state.labelImages.filter((image) => image.id !== state.currentImage?.id);
  if (!candidates.length) return;
  // Reviewed-but-empty frames are done, not pending; without this they kept coming back.
  const untouched = candidates.filter((image) => Number(image.annotation_count) === 0 && !image.reviewed);
  const pool = untouched.length ? untouched : candidates;
  const next = pool[Math.floor(Math.random() * pool.length)];
  await openLabelImage(next.id, { saveCurrent });
}

async function saveAndNextImage() {
  if (state.cropMode || state.cropDraft) {
    showToast("請先拖出區域，或再按一次「區域」取消。");
    return;
  }
  if (hasIncompleteBoxes()) {
    showToast("每個 bbox 至少要選一個 W/E/R；不要的框請用 Delete 刪除。");
    return;
  }
  if (!(await saveCurrentAnnotations())) return;
  await randomNextImage({ saveCurrent: false });
}

async function previousImage() {
  if (hasIncompleteBoxes()) {
    showToast("每個 bbox 至少要選一個 W/E/R；不要的框請用 Delete 刪除。");
    return;
  }
  const previousId = state.imageHistory.at(-1);
  if (!previousId) {
    showToast("目前沒有更早的圖片。");
    return;
  }
  const opened = await openLabelImage(previousId, { remember: false });
  if (opened) state.imageHistory.pop();
  updateImageNavigation();
}

function handleLabelKey(event) {
  if (!document.querySelector("#label-page").classList.contains("active")) return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
  if (event.code === "Space") {
    event.preventDefault();
    if (!event.repeat) saveAndNextImage();
    return;
  }
  const action = ACTIONS[event.key.toLowerCase()];
  if (action) {
    event.preventDefault();
    toggleAction(action.field);
    return;
  }
  if ((event.key === "Delete" || event.key === "Backspace") && state.selectedBoxId) {
    event.preventDefault();
    state.annotations = state.annotations.filter((box) => box.id !== state.selectedBoxId);
    state.selectedBoxId = null;
    renderAnnotations();
    queueSave();
  }
}

function handleLabelWheel(event) {
  if (!state.currentImage) return;
  event.preventDefault();
  const oldScale = state.zoomScale;
  const newScale = Math.min(8, Math.max(1, oldScale * Math.exp(-event.deltaY * 0.0015)));
  if (newScale === oldScale) return;
  if (newScale === 1) {
    resetZoom();
    return;
  }
  const stageRect = elements.labelStage.getBoundingClientRect();
  const cursorX = event.clientX - stageRect.left;
  const cursorY = event.clientY - stageRect.top;
  const frameX = elements.imageFrame.offsetLeft;
  const frameY = elements.imageFrame.offsetTop;
  const imageX = (cursorX - frameX - state.zoomX) / oldScale;
  const imageY = (cursorY - frameY - state.zoomY) / oldScale;
  state.zoomX = cursorX - frameX - imageX * newScale;
  state.zoomY = cursorY - frameY - imageY * newScale;
  state.zoomScale = newScale;
  applyZoom();
}

elements.navItems.forEach((item) => item.addEventListener("click", () => selectPage(item.dataset.page)));
elements.driveForm.addEventListener("submit", inspectSource);
elements.driveUrl.addEventListener("input", updateSourceHint);
elements.clearUrl.addEventListener("click", () => {
  elements.driveUrl.value = "";
  elements.driveUrl.focus();
  elements.resultsPanel.classList.add("hidden");
  elements.nextStepPanel.hidden = true;
  state.scan = null;
  updateSourceHint();
  setStatus("idle", "等待來源連結", "貼上 Drive 資料夾、影片或 Facebook 貼文連結。");
});
elements.datasetName.addEventListener("input", updateDatasetCreateState);
elements.sharerName.addEventListener("input", () => {
  rememberSharerName(elements.sharerName.value);
  updateDatasetCreateState();
});
elements.createDatasetButton.addEventListener("click", createDataset);
elements.connectDriveButton.addEventListener("click", () => {
  if (state.config?.private_drive_connected) return;
  elements.driveSetupNote.hidden = !elements.driveSetupNote.hidden;
});
elements.ingestDatasetSelect.addEventListener("change", updateIngestButton);
elements.startIngestButton.addEventListener("click", startIngest);
elements.trainHost.addEventListener("input", renderLatestTrainingRun);
elements.startTrainButton.addEventListener("click", startTraining);
elements.labelDatasetSelect.addEventListener("change", (event) => {
  state.loadedLabelDatasetId = null;
  loadLabelImages(event.target.value);
});
elements.cropRegionButton.addEventListener("click", toggleCropMode);
elements.clearRegionButton.addEventListener("click", clearCropRegion);
elements.previousImageButton.addEventListener("click", previousImage);
elements.randomImageButton.addEventListener("click", randomNextImage);
elements.annotationLayer.addEventListener("pointerdown", beginBox);
elements.annotationLayer.addEventListener("pointermove", moveBox);
elements.annotationLayer.addEventListener("pointerup", finishBox);
elements.annotationLayer.addEventListener("pointercancel", finishBox);
elements.labelStage.addEventListener("wheel", handleLabelWheel, { passive: false });
elements.actionButtons.forEach((button) => button.addEventListener("click", () => toggleAction(button.dataset.action)));
document.addEventListener("keydown", handleLabelKey);

// Prefilled once here, never from renderLatestTrainingRun: loadWorkspace polls every 2s
// and would overwrite the host while it is being typed.
elements.trainHost.value = rememberedTrainHost();
state.selectedSharers = rememberedSelectedSharers();

const initialPage = ["dataset", "label", "train"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "dataset";
selectPage(initialPage);
Promise.all([loadConfig(), loadWorkspace()]);
window.setInterval(loadWorkspace, 2000);
