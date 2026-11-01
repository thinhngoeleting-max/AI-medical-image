# My Report, Explained

Local conversation prototype for explaining existing radiology report wording.
It is a separate feature beside the image-classification research pipeline.
Its input is plain text, not an X-ray, a Grad-CAM map, or generated classifier findings.

## First run

From the project root, with `.venv` active, install and start Ollama separately.
The application client uses Python's standard library; no extra Python dependency
or paid API key is needed. Download the configured local model:

```powershell
ollama pull gemma4:e2b
python -m healthcare_chatbot --example uncertainty --chat
```

If Ollama is not running, start its desktop app or run `ollama serve` in another
terminal. Run after image-model training to leave GPU memory available.
Model choice, generation limits and connection settings are in `chatbot_config.json`.

To inspect the source prompt without any model/server:

```powershell
python -m healthcare_chatbot --example negation --dry-run
```

To ask one question or read a plain-text file:

```powershell
python -m healthcare_chatbot --example follow_up --question "What follow-up does this report actually specify?"
python -m healthcare_chatbot --report private_reports/example.txt --chat
```

The `private_reports/` directory is optional and ignored by Git. Use fictional
examples first; remove identifying information from any permitted real input.
The application does not persist prompts, reports or chat history. Reports go to
the configured loopback Ollama endpoint; use an actually local model/server.
Ollama's own configuration, logging and model behaviour are outside this app.
`--dry-run` intentionally prints the request, including report text, to the terminal.

## Relationship to the existing project

- `medical_gradcam/`: trains and evaluates the NORMAL/PNEUMONIA classifier.
- `Evaluation.ipynb`: displays classification, Grad-CAM and robustness results.
- `healthcare_chatbot/`: explains source report wording with an LLM.
- `config.json`: imaging settings; `chatbot_config.json`: conversation settings.

Neither part writes into the other's outputs. The LLM does not automatically
receive image-model predictions, turn them into radiology reports, or change the
classifier. No separate generated-report module is restored.

## Behaviour and limits

The prompt asks for exact source phrases, plain-language definitions, preserved
negation/uncertainty, missing-information limits, and useful clinician questions.
Follow-ups keep the original report and bounded conversation history in context.
Prompt instructions are not a guarantee of factual correctness, clinical safety,
or protection against malicious document instructions. A human must check outputs.
Character limits bound input size but are not token counts; large reports plus
history may require a larger context or a shorter session.

The text-report explainer has a terminal chat interface. The separate browser
interface (`python -m app`) handles X-ray upload, ResNet inference and an optional
`ImageResultAssistant` conversation. `ReportExplainer.ask()` remains available for
future report-specific interfaces. The report explainer does
not parse PDFs, perform OCR, diagnose conditions, prescribe treatment, or retrieve
external medical references. It is not ready for clinical use.

See `docs/chatbot_scenarios.md` for example conversations and evaluation criteria.
Offline tests use fake responses and verify plumbing, not an LLM's medical accuracy.

The client follows the official [Ollama chat API](https://docs.ollama.com/api/chat).
