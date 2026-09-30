"use strict";

const state = { opacity: 0.55, lastMap: null, lastImage: null };

async function api(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `请求失败：${response.status}`);
  }
  return payload;
}

function showError(message) {
  const box = document.getElementById("errorBox");
  box.textContent = message;
  box.hidden = false;
}

function clearError() {
  document.getElementById("errorBox").hidden = true;
}

async function refreshStatus() {
  const s = await api("/api/status");
  const bar = document.getElementById("statusBar");
  bar.textContent =
    `参考图 ${s.reference_count} 张 · 校准图 ${s.calibration_count} 张 · ` +
    `记忆库 ${s.ready ? s.memory_bank_size + " 项" : "未建立"} · ` +
    `95% 阈值 ${s.threshold.toFixed(4)} · 热图颜色上限 ${s.color_upper_bound.toFixed(4)}`;
}

async function refreshGroup(group) {
  const samples = await api(`/api/samples/${group}`);
  const container = document.getElementById(
    group === "reference" ? "referenceList" : "calibrationList"
  );
  container.innerHTML = "";
  for (const sample of samples) {
    const card = document.createElement("div");
    card.className = "sample-card";
    const image = document.createElement("img");
    image.src = `/api/samples/${group}/${sample.id}/image?t=${Date.now()}`;
    image.alt = sample.filename;
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = sample.filename;
    const button = document.createElement("button");
    button.textContent = "删除";
    button.onclick = async () => {
      clearError();
      try {
        await api(`/api/samples/${group}/${sample.id}`, { method: "DELETE" });
        await Promise.all([refreshStatus(), refreshGroup(group)]);
      } catch (error) {
        showError(error.message);
      }
    };
    card.append(image, name, button);
    container.appendChild(card);
  }
}

async function uploadFiles(group, files) {
  for (const file of files) {
    const form = new FormData();
    form.append("file", file);
    try {
      await api(`/api/samples/${group}`, { method: "POST", body: form });
    } catch (error) {
      showError(`「${file.name}」上传失败：${error.message}`);
    }
  }
  await Promise.all([
    refreshStatus(),
    refreshGroup("reference"),
    refreshGroup("calibration"),
  ]);
}

function turboColorMap(value) {
  // Approximate matplotlib turbo LUT with 7 smooth segments.
  const v = Math.min(1, Math.max(0, value));
  const stops = [
    [0.0, [48, 18, 59]],
    [0.15, [65, 88, 210]],
    [0.35, [28, 170, 222]],
    [0.55, [34, 221, 130]],
    [0.7, [178, 232, 50]],
    [0.85, [247, 182, 39]],
    [1.0, [122, 24, 32]],
  ];
  for (let i = 1; i < stops.length; i += 1) {
    if (v <= stops[i][0]) {
      const [x0, c0] = stops[i - 1];
      const [x1, c1] = stops[i];
      const t = (v - x0) / (x1 - x0);
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * t),
        Math.round(c0[1] + (c1[1] - c0[1]) * t),
        Math.round(c0[2] + (c1[2] - c0[2]) * t),
      ];
    }
  }
  return stops[stops.length - 1][1];
}

function renderResult() {
  if (!state.lastMap || !state.lastImage) return;
  const { width, height, data, upperBound } = state.lastMap;
  const canvas = document.getElementById("resultCanvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(state.lastImage, 0, 0, width, height);

  const overlay = document.createElement("canvas");
  overlay.width = width;
  overlay.height = height;
  const octx = overlay.getContext("2d");
  const imageData = octx.createImageData(width, height);
  for (let i = 0, j = 0; i < data.length; i += 1, j += 4) {
    const normalized = data[i] / upperBound; // fixed global bound, no autoscale
    const [r, g, b] = turboColorMap(normalized);
    imageData.data[j] = r;
    imageData.data[j + 1] = g;
    imageData.data[j + 2] = b;
    imageData.data[j + 3] = Math.round(state.opacity * 255);
  }
  octx.putImageData(imageData, 0, 0);
  ctx.drawImage(overlay, 0, 0);
}

function decodeMap(encoded, width, height) {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer, 0, width * height);
}

async function detectFile(file, imageUrl) {
  const form = new FormData();
  form.append("file", file);
  const result = await api("/api/detect", { method: "POST", body: form });
  const image = new Image();
  image.src = imageUrl || URL.createObjectURL(file);
  await image.decode();
  state.lastImage = image;
  state.lastMap = {
    width: result.distance_map.width,
    height: result.distance_map.height,
    data: decodeMap(result.distance_map.data, result.distance_map.width, result.distance_map.height),
    upperBound: result.color_upper_bound,
  };
  renderResult();
  const meta = document.getElementById("resultMeta");
  meta.className = `result-meta ${result.is_anomaly ? "anomaly" : "normal"}`;
  meta.textContent = result.is_anomaly
    ? `判定：异常（检出划痕/异物）｜分数 ${result.score.toFixed(4)} ＞ 阈值 ${result.threshold.toFixed(4)}`
    : `判定：正常｜分数 ${result.score.toFixed(4)} ≤ 阈值 ${result.threshold.toFixed(4)}`;
  document.getElementById("heatmapNote").textContent =
    `颜色上限固定为 ${result.color_upper_bound.toFixed(4)}（阈值 ${result.threshold.toFixed(4)} 的两倍；阈值为 0 时改用校准最高分），跨图可比；状态版本 v${result.state_version}。`;
}

async function fetchExampleBlob(kind, variant) {
  const response = await fetch(`/api/examples/${kind}/${variant}.png`);
  if (!response.ok) throw new Error("示例下载失败");
  const blob = await response.blob();
  return new File([blob], `${kind}_${variant}.png`, { type: "image/png" });
}

async function renderExamples() {
  const examples = await api("/api/examples");
  const container = document.getElementById("examples");
  const labels = {
    normal_ref: "正常 · 建库参考",
    normal_cal: "正常 · 独立校准",
    scratch: "异常 · 划痕",
    foreign_object: "异常 · 异物",
  };
  for (const [kind, spec] of Object.entries(examples)) {
    const block = document.createElement("div");
    block.className = "example-block";
    const selector = document.createElement("select");
    for (let v = 0; v < spec.variants; v += 1) {
      const option = document.createElement("option");
      option.value = String(v);
      option.textContent = `变体 ${v + 1}`;
      selector.appendChild(option);
    }
    const image = document.createElement("img");
    const updateImage = () => {
      image.src = `/api/examples/${kind}/${selector.value}.png?t=${Date.now()}`;
    };
    selector.onchange = updateImage;
    updateImage();
    const purpose = document.createElement("span");
    purpose.className = "hint";
    purpose.textContent = `${labels[kind]}：${spec.purpose}`;
    const row = document.createElement("div");
    row.className = "example-row";
    const action = document.createElement("button");
    if (kind === "normal_ref") {
      action.textContent = "送入参考组";
      action.onclick = async () => {
        clearError();
        try {
          await uploadFiles("reference", [await fetchExampleBlob(kind, Number(selector.value))]);
        } catch (error) { showError(error.message); }
      };
    } else if (kind === "normal_cal") {
      action.textContent = "送入校准组";
      action.onclick = async () => {
        clearError();
        try {
          await uploadFiles("calibration", [await fetchExampleBlob(kind, Number(selector.value))]);
        } catch (error) { showError(error.message); }
      };
    } else {
      action.textContent = "立即检测";
      action.onclick = async () => {
        clearError();
        try {
          const variant = Number(selector.value);
          const file = await fetchExampleBlob(kind, variant);
          await detectFile(file, `/api/examples/${kind}/${variant}.png`);
        } catch (error) { showError(error.message); }
      };
    }
    row.append(image, selector, action);
    block.append(row, purpose);
    container.appendChild(block);
  }
}

document.getElementById("referenceInput").addEventListener("change", async (event) => {
  clearError();
  try { await uploadFiles("reference", Array.from(event.target.files)); }
  catch (error) { showError(error.message); }
  event.target.value = "";
});
document.getElementById("calibrationInput").addEventListener("change", async (event) => {
  clearError();
  try { await uploadFiles("calibration", Array.from(event.target.files)); }
  catch (error) { showError(error.message); }
  event.target.value = "";
});
document.getElementById("detectInput").addEventListener("change", async (event) => {
  clearError();
  const file = event.target.files[0];
  if (file) {
    try { await detectFile(file); } catch (error) { showError(error.message); }
  }
  event.target.value = "";
});
document.getElementById("opacityRange").addEventListener("input", (event) => {
  state.opacity = Number(event.target.value) / 100;
  document.getElementById("opacityValue").textContent = `${event.target.value}%`;
  renderResult();
});

(async function init() {
  try {
    await Promise.all([
      refreshStatus(),
      refreshGroup("reference"),
      refreshGroup("calibration"),
      renderExamples(),
    ]);
  } catch (error) {
    showError(`初始化失败：${error.message}`);
  }
})();
