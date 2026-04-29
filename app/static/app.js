var cameraInput = document.querySelector("#cameraInput");
var galleryInput = document.querySelector("#galleryInput");
var previewImage = document.querySelector("#previewImage");
var previewPlaceholder = document.querySelector("#previewPlaceholder");
var resultStage = document.querySelector("#resultStage");
var messageBox = document.querySelector("#messageBox");
var resultMeta = document.querySelector("#resultMeta");
var resultList = document.querySelector("#resultList");
var confRange = document.querySelector("#confRange");
var iouRange = document.querySelector("#iouRange");
var confValue = document.querySelector("#confValue");
var iouValue = document.querySelector("#iouValue");
var objectUrlApi = window.URL || window.webkitURL;

bindRange(confRange, confValue);
bindRange(iouRange, iouValue);
cameraInput.addEventListener("change", function () {
  handleFileSelection(getSelectedFile(cameraInput));
});
galleryInput.addEventListener("change", function () {
  handleFileSelection(getSelectedFile(galleryInput));
});

function getSelectedFile(input) {
  return input.files && input.files.length ? input.files[0] : null;
}

function bindRange(range, output) {
  output.textContent = Number(range.value).toFixed(2);
  range.addEventListener("input", function () {
    output.textContent = Number(range.value).toFixed(2);
  });
}

function handleFileSelection(file) {
  if (!file) {
    return;
  }

  if (objectUrlApi && objectUrlApi.createObjectURL) {
    showPreview(objectUrlApi.createObjectURL(file), true);
  }
  setMessage("图片已选中，正在识别中。", "info");
  clearResults();

  var formData = new FormData();
  formData.append("image", file);
  formData.append("conf", confRange.value);
  formData.append("iou", iouRange.value);

  var request = new XMLHttpRequest();
  request.open("POST", "/api/detect/image");
  request.onreadystatechange = function () {
    if (request.readyState !== 4) {
      return;
    }

    var payload;
    try {
      payload = JSON.parse(request.responseText || "{}");
    } catch (error) {
      payload = { ok: false, error: "识别结果解析失败。" };
    }

    if (request.status < 200 || request.status >= 300 || !payload.ok) {
      renderDetections([]);
      setMessage(payload.error || "识别请求失败。", "error");
      return;
    }

    showPreview(payload.image_url, false);
    renderMeta(payload.meta);
    renderDetections(payload.detections);
    setMessage("识别完成，共检测到 " + payload.meta.count + " 个目标。", "info");
  };
  request.onerror = function () {
    renderDetections([]);
    setMessage("网络请求失败，请检查服务是否正常运行。", "error");
  };
  request.send(formData);
}

function showPreview(src, isObjectUrl) {
  var previousObjectUrl = previewImage.getAttribute("data-object-url");
  if (previousObjectUrl && objectUrlApi && objectUrlApi.revokeObjectURL) {
    objectUrlApi.revokeObjectURL(previousObjectUrl);
    previewImage.removeAttribute("data-object-url");
  }
  if (isObjectUrl) {
    previewImage.setAttribute("data-object-url", src);
  }
  previewImage.src = src;
  previewImage.hidden = false;
  previewImage.className = "is-visible";
  previewPlaceholder.hidden = true;
  previewPlaceholder.className = "result-placeholder is-hidden";
  resultStage.className = "result-stage has-preview";
}

function setMessage(text, type) {
  messageBox.textContent = text;
  messageBox.className = "message message--" + type + " is-visible";
}

function clearResults() {
  resultMeta.innerHTML = "";
  resultList.innerHTML = "";
}

function renderMeta(meta) {
  resultMeta.innerHTML = "";
  var chips = [
    "目标数 " + meta.count,
    "图片尺寸 " + meta.image_width + " x " + meta.image_height,
    "置信度 " + Number(meta.conf).toFixed(2),
    "IOU " + Number(meta.iou).toFixed(2),
  ];

  chips.forEach(function (text) {
    var chip = document.createElement("div");
    chip.className = "meta-chip";
    chip.textContent = text;
    resultMeta.appendChild(chip);
  });
}

function renderDetections(detections) {
  resultList.innerHTML = "";
  if (!detections.length) {
    var empty = document.createElement("div");
    empty.className = "result-item";
    empty.innerHTML = ""
      + '<div class="result-item__title">'
      + "<span>没有命中目标</span>"
      + "<span>0</span>"
      + "</div>"
      + '<div class="result-item__meta">可以降低置信度阈值后重新识别。</div>';
    resultList.appendChild(empty);
    return;
  }

  detections.forEach(function (item, index) {
    var card = document.createElement("div");
    card.className = "result-item";
    card.innerHTML = ""
      + '<div class="result-item__title">'
      + "<span>" + (index + 1) + ". " + escapeHtml(item.class_name) + "</span>"
      + "<span>" + (item.confidence * 100).toFixed(1) + "%</span>"
      + "</div>"
      + '<div class="result-item__meta">bbox: [' + item.bbox.join(", ") + "]</div>";
    resultList.appendChild(card);
  });
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
