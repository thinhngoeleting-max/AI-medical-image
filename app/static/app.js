"use strict";
const $ = (id) => document.getElementById(id);
const state = { file: null, example: null, result: null, view: "original", busy: false, chatBusy: false, version: 0, previewUrl: null, models: [], controller: null };
const percent = (value) => (100 * value).toFixed(1) + "%";

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = "Something went wrong. Check the app terminal and try again.";
    try { const data = await response.json(); if (typeof data.detail === "string") detail = data.detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}
function errorBox(id, message = "") { $(id).textContent = message; $(id).hidden = !message; }
function updateControls() {
  $("analyze-button").disabled = state.busy || (!state.file && !state.example);
  const canChat = !!state.result && state.models.length > 0 && !state.chatBusy && !state.busy;
  $("chat-input").disabled = !canChat;
  $("send-button").disabled = !canChat;
  document.querySelectorAll("[data-question]").forEach((button) => button.disabled = !canChat);
  document.querySelector('[data-view="gradcam"]').disabled = !state.result;
  $("llm-model").disabled = state.models.length === 0 || state.chatBusy;
}
function clearResult() {
  state.version += 1;
  if (state.controller) state.controller.abort();
  state.result = null; state.view = "original"; state.busy = false; state.chatBusy = false;
  $("result-empty").hidden = false; $("result-content").hidden = true;
  $("result-state").textContent = "AWAITING ANALYSIS";
  $("attribution-note").hidden = true;
  $("chat-messages").replaceChildren();
  const welcome = document.createElement("div"); welcome.className = "chat-welcome";
  welcome.textContent = "Analyse this image to explore the model’s result and limitations.";
  $("chat-messages").append(welcome); $("chat-input").value = "";
  $("analyze-button").classList.remove("loading"); $("analyze-button").firstElementChild.textContent = "Analyse image";
  errorBox("page-error"); errorBox("chat-error"); updateControls();
}
function showPreview(src) {
  $("image-preview").src = src; $("image-preview").hidden = false;
  $("drop-placeholder").hidden = true; $("drop-zone").classList.add("has-image");
  $("reset-button").hidden = false; setView("original");
}
function releasePreview() { if (state.previewUrl) URL.revokeObjectURL(state.previewUrl); state.previewUrl = null; }
function selectFile(file) {
  if (!file) return;
  if (!/\.(jpe?g|png)$/i.test(file.name) || file.size > 10 * 1024 * 1024 || file.size === 0) {
    errorBox("page-error", "Choose a nonempty JPEG or PNG image up to 10 MB."); return;
  }
  clearResult(); releasePreview(); state.file = file; state.example = null;
  state.previewUrl = URL.createObjectURL(file); showPreview(state.previewUrl);
  $("file-name").textContent = file.name; $("preview-caption").hidden = true;
  document.querySelectorAll(".example-button").forEach((b) => b.classList.remove("selected")); updateControls();
}
function selectExample(example, button) {
  clearResult(); releasePreview(); state.file = null; state.example = example;
  $("file-input").value = ""; showPreview(example.image_url);
  $("file-name").textContent = "Test dataset example " + (example.id + 1);
  $("preview-caption").textContent = "Dataset label: " + example.label; $("preview-caption").hidden = false;
  document.querySelectorAll(".example-button").forEach((b) => b.classList.toggle("selected", b === button)); updateControls();
}
function setView(view) {
  if (view === "gradcam" && !state.result) return;
  state.view = view;
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.classList.toggle("selected", button.dataset.view === view);
    button.setAttribute("aria-pressed", String(button.dataset.view === view));
  });
  if (state.result) $("image-preview").src = view === "gradcam" ? state.result.gradcam_image : state.result.original_image;
  $("image-preview").alt = view === "gradcam" ? "Grad-CAM attribution overlay for the model prediction" : "Selected X-ray";
  $("view-description").textContent = view === "gradcam" ? "Attribution, not disease localisation" : "Images are not saved by this app";
  $("attribution-note").hidden = view !== "gradcam";
}
function showResult(result) {
  state.result = result; $("result-empty").hidden = true; $("result-content").hidden = false;
  $("result-state").textContent = "ANALYSIS COMPLETE";
  $("prediction-label").textContent = result.prediction === "NORMAL" ? "Normal class" : "Pneumonia class";
  $("prediction-score").textContent = percent(result.model_probability);
  $("probability-bars").replaceChildren();
  Object.entries(result.probabilities).forEach(([label, probability]) => {
    const row = document.createElement("div"); row.className = "probability-row";
    const caption = document.createElement("div"); caption.className = "probability-caption";
    const name = document.createElement("span"); name.textContent = label;
    const value = document.createElement("span"); value.textContent = percent(probability); caption.append(name, value);
    const track = document.createElement("div"); track.className = "bar-track";
    const fill = document.createElement("div"); fill.className = "bar-fill"; fill.style.width = percent(probability);
    track.append(fill); row.append(caption, track); $("probability-bars").append(row);
  });
  $("dataset-label").hidden = !result.dataset_label;
  $("dataset-label").textContent = result.dataset_label ? "Dataset label: " + result.dataset_label + " · " + (result.dataset_label === result.prediction ? "Matches this label" : "Does not match this label") : "";
  $("result-caveat").textContent = result.prediction === "NORMAL" ? "A NORMAL score does not establish that someone is healthy or rule out pneumonia or other conditions." : "A PNEUMONIA score does not confirm pneumonia, its cause, or the treatment needed.";
  $("model-details").textContent = result.model.architecture.toUpperCase() + " · Epoch " + (result.model.epoch ?? "unknown") + " · " + result.model.image_size + " × " + result.model.image_size + " · Patient separation " + (result.model.patient_separation_verified ? "verified" : "unverified");
  $("chat-messages").replaceChildren();
  addMessage("assistant", "The image model selected " + result.prediction + " with a score of " + percent(result.model_probability) + ". You can ask your local AI to explain that score or the heatmap. This output is experimental, not a diagnosis.", "CLASSIFIER SUMMARY");
  setView("gradcam"); updateControls();
}
async function analyze() {
  if (state.busy || (!state.file && !state.example)) return;
  state.busy = true; const version = state.version; state.controller = new AbortController();
  errorBox("page-error"); updateControls();
  $("analyze-button").classList.add("loading"); $("analyze-button").firstElementChild.textContent = "Analysing image and generating Grad-CAM…";
  $("result-state").textContent = "ANALYSING…";
  try {
    let url, options = { method: "POST", signal: state.controller.signal };
    if (state.example) url = "/api/examples/" + state.example.id + "/analyze";
    else { url = "/api/analyze"; options.body = state.file; options.headers = { "Content-Type": state.file.type || "application/octet-stream" }; }
    const result = await request(url, options);
    if (version === state.version) showResult(result);
  } catch (error) {
    if (error.name !== "AbortError" && version === state.version) { errorBox("page-error", error.message); $("result-state").textContent = "TRY AGAIN"; }
  } finally {
    if (version === state.version) { state.busy = false; $("analyze-button").classList.remove("loading"); $("analyze-button").firstElementChild.textContent = "Analyse image again"; updateControls(); }
  }
}
function addMessage(role, text, label) {
  const message = document.createElement("div"); message.className = "chat-message " + role;
  const heading = document.createElement("span"); heading.className = "message-label"; heading.textContent = label || (role === "user" ? "YOU" : "LOCAL ASSISTANT");
  const body = document.createElement("span"); body.textContent = text;
  message.append(heading, body); $("chat-messages").append(message); $("chat-messages").scrollTop = $("chat-messages").scrollHeight;
  return message;
}
async function sendQuestion(question) {
  question = question.trim(); if (!question || !state.result || state.chatBusy || !state.models.length) return;
  const version = state.version; const model = $("llm-model").value;
  state.chatBusy = true; updateControls(); errorBox("chat-error"); $("chat-input").value = "";
  const questionBubble = addMessage("user", question);
  const waiting = addMessage("assistant", "Thinking locally… The first response may take a little longer.", model);
  try {
    const result = await request("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ analysis_id: state.result.analysis_id, question, model }) });
    if (version === state.version) { waiting.remove(); addMessage("assistant", result.answer, result.model); }
  } catch (error) {
    if (version === state.version) { waiting.remove(); questionBubble.remove(); errorBox("chat-error", error.message); $("chat-input").value = question; }
  } finally { if (version === state.version) { state.chatBusy = false; updateControls(); } }
}
async function refreshStatus() {
  $("refresh-models").disabled = true;
  try {
    const data = await request("/api/status"); $("image-model").textContent = data.architecture.toUpperCase() + (data.checkpoint_ready ? " · Ready" : " · No checkpoint");
    if (!data.checkpoint_ready) errorBox("page-error", "No trained checkpoint found. Run python -m medical_gradcam train, then refresh this page.");
    const selected = $("llm-model").value; state.models = data.ollama.models; $("llm-model").replaceChildren();
    if (state.models.length) {
      state.models.forEach((name) => { const option = document.createElement("option"); option.value = name; option.textContent = name; $("llm-model").append(option); });
      $("llm-model").value = state.models.includes(selected) ? selected : state.models.includes(data.configured_model) ? data.configured_model : state.models[0];
      $("llm-status").textContent = "Ollama connected · Choose an installed model above";
    } else {
      const option = document.createElement("option"); option.textContent = "No local model available"; $("llm-model").append(option);
      $("llm-status").textContent = data.ollama.online ? "Ollama is running, but no local model is installed. Run: ollama pull " + data.configured_model : "Ollama is offline. Image analysis still works. Start the Ollama app, then refresh.";
    }
    $("llm-status").classList.toggle("offline", !state.models.length); updateControls();
  } catch (error) { errorBox("page-error", "Could not connect to the local app. Check that python -m app is still running. " + error.message); }
  finally { $("refresh-models").disabled = false; }
}
async function loadExamples() {
  try {
    const examples = await request("/api/examples"); $("examples").replaceChildren();
    if (!examples.length) { $("examples").textContent = "No test examples found. You can upload an image instead."; return; }
    examples.forEach((example) => {
      const button = document.createElement("button"); button.type = "button"; button.className = "example-button";
      button.setAttribute("aria-label", "Use test example " + (example.id + 1) + ", dataset label " + example.label);
      const image = document.createElement("img"); image.src = example.image_url; image.alt = "Chest X-ray test example " + (example.id + 1); image.loading = "lazy";
      const label = document.createElement("span"); label.textContent = example.label;
      button.append(image, label); button.addEventListener("click", () => selectExample(example, button)); $("examples").append(button);
    });
  } catch (_) { $("examples").textContent = "Examples could not be loaded. Upload an image instead."; }
}
$("file-input").addEventListener("change", (event) => selectFile(event.target.files[0]));
$("drop-zone").addEventListener("click", () => { if (!state.file && !state.example) $("file-input").click(); });
$("drop-zone").addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); $("file-input").click(); } });
$("drop-zone").addEventListener("dragover", (event) => { event.preventDefault(); $("drop-zone").classList.add("drag-over"); });
$("drop-zone").addEventListener("dragleave", () => $("drop-zone").classList.remove("drag-over"));
$("drop-zone").addEventListener("drop", (event) => { event.preventDefault(); $("drop-zone").classList.remove("drag-over"); selectFile(event.dataTransfer.files[0]); });
$("reset-button").addEventListener("click", () => {
  clearResult(); releasePreview(); state.file = null; state.example = null; $("file-input").value = "";
  $("image-preview").hidden = true; $("image-preview").removeAttribute("src"); $("drop-placeholder").hidden = false;
  $("drop-zone").classList.remove("has-image"); $("reset-button").hidden = true; $("preview-caption").hidden = true;
  $("file-name").textContent = "No image selected"; setView("original");
  document.querySelectorAll(".example-button").forEach((button) => button.classList.remove("selected")); updateControls();
});
$("analyze-button").addEventListener("click", analyze);
document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
$("chat-form").addEventListener("submit", (event) => { event.preventDefault(); sendQuestion($("chat-input").value); });
document.querySelectorAll("[data-question]").forEach((button) => button.addEventListener("click", () => sendQuestion(button.dataset.question)));
$("refresh-models").addEventListener("click", refreshStatus);
Promise.all([refreshStatus(), loadExamples()]);
