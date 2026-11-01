# Clarity: local X-ray interface

Run from the project root with the existing virtual environment:

```powershell
python -m pip install -r requirements-ui.txt
python -m app
```

Open **http://127.0.0.1:8000**. Stop with Ctrl+C. To use another port:
`python -m app --port 8001`. This app runs on your computer and is not deployed
to a hosted Site: it needs your local PyTorch checkpoint and Ollama service.

## Use it

1. Drop in a JPEG/PNG chest X-ray up to 10 MB, browse for a file, or select a test example.
2. Click **Analyse image** to get NORMAL/PNEUMONIA scores and predicted-class Grad-CAM.
3. Switch between the resized original and overlay. Test examples also show their dataset label.
4. Select an installed local language model and ask about the measured result.

The classifier loads `run_dir/best.pt` using `config.json`, with the preprocessing
and architecture stored in the checkpoint. Train first if it does not exist:

```powershell
python -m medical_gradcam train
```

No retraining is required to use this interface with an existing checkpoint.
Finish training before launching the app. Inference loads the model once and
serializes classifier/Grad-CAM calls so hooks do not interfere across requests.
If the checkpoint changes, the next analysis reloads it.

## Optional local chat

The Google model configured in `chatbot_config.json` is `gemma4:e2b`. Download it
separately in Ollama (a multi-GB download):

```powershell
ollama pull gemma4:e2b
```

Start the Ollama app (or `ollama serve`), then click **Refresh** beside the model
selector. The selector lists downloaded local models; choose another without
changing the image classifier. The app makes no LLM call until you ask a question.
Classification, Grad-CAM and the fixed next-step guidance work without Ollama.

The LLM receives classifier probabilities and metadata, not image pixels. It
explains the result; it is not a second image interpreter. The existing text-report
explainer remains a separate terminal feature. No generated-report pipeline has
been added back.

## Limits and next-step guidance

The classifier always chooses one of two labels. It cannot verify that an upload
is an X-ray, determine disease severity/cause, rule out other conditions, or decide
treatment. Its probabilities are not validated clinical certainty. Grad-CAM
attribution is not a disease annotation. Use examples from the training data's
domain; results on unrelated images or populations may be misleading.

The interface recommends discussing actual imaging, a radiology report and symptoms
with a qualified clinician, rather than acting on this research output. Its generic
urgent-symptom note follows [NHS symptom guidance](https://www.nhs.uk/conditions/pneumonia/).
It does not provide prescriptions or a triage assessment. LLM instructions cannot
guarantee medical correctness; do not use this prototype for clinical decisions.

## Files and privacy

Uploads are read into bounded memory; original uploads and generated overlays are
not written to the dataset, training folder or an upload directory. The browser
receives resized PNG previews. Up to 20 server-side analysis/chat contexts are held
in memory and expire after 30 minutes; restarting clears them. The browser retains
its current result until cleared or closed. Host and same-origin checks limit
browser access to the local app. This is an unauthenticated single-computer prototype,
not a multi-user medical records service. Standard server logs record request paths,
not uploaded bytes or chat request bodies. Ollama's own logging/configuration is
outside this app.

## Implementation

- `static/`: HTML, CSS and JavaScript interface; no frontend build tool needed.
- `server.py`: local FastAPI endpoints and bounded conversation sessions.
- `medical_gradcam/inference.py`: in-memory prediction and Grad-CAM.
- `healthcare_chatbot/image_assistant.py`: measured-result chat context.
- `healthcare_chatbot/prompts/image_result.txt`: image-result explanation instructions.

API upload uses a raw JPEG/PNG request body, not multipart form data. This avoids
temporary upload files. Example image IDs map to server-selected test files and
cannot request arbitrary paths. Offline tests check inference, endpoints, and
LLM request plumbing; they do not establish clinical validity.
