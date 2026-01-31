## What’s inside


* **Notebooks**

  * `Action Detection Tutorial.ipynb`
  * `Action Detection Refined.ipynb`
* **Apps / Scripts**

  * `ui.py` — UI entrypoint (recommended)
  * `app.py` — app/runner script
  * `test.py` — quick test harness
  * `modeltens.py` — model utilities / helpers
* **Models / Assets**

  * `action.h5`, `nats-action.h5` — Keras/TensorFlow models
  * `gesture_recognizer.task` — gesture recognizer task file

> Note: the repo currently has **no GitHub “About” description** set. 

---

## Quickstart

### 1) Clone & install

```bash
git clone https://github.com/nathanieluriri/signs.git
cd signs

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Run the UI (recommended)

If `ui.py` is a Streamlit UI, run it like this (Streamlit’s standard CLI): ([Streamlit Docs][2])

```bash
streamlit run ui.py
```

### 3) Or run the app script

```bash
python app.py
```

---

## How it works (high level)

Typical real-time gesture/sign pipelines follow this loop:

1. Capture frames from your webcam
2. Detect/track hands and extract landmarks/features
3. Feed a short window of features into a model
4. Show the predicted label + confidence in the UI

This repo provides both:

* **Training/experimentation notebooks** (`Action Detection*.ipynb`)  
* **Inference-ready assets** (`.h5` models + `gesture_recognizer.task`)  

---

## Project structure

```text
.
├─ app.py
├─ ui.py
├─ test.py
├─ modeltens.py
├─ requirements.txt
├─ Action Detection Tutorial.ipynb
├─ Action Detection Refined.ipynb
├─ action.h5
├─ nats-action.h5
└─ gesture_recognizer.task
```

 

---

## Using different models

If the UI/app supports selecting models, these are your shipped options: 
* `action.h5`
* `nats-action.h5`

A common pattern is an argument or a variable in the script, e.g.:

```python
MODEL_PATH = "action.h5"  # or "nats-action.h5"
```

---

## Troubleshooting

* **Webcam not opening**

  * Close other apps using the camera (Zoom/Meet/etc.)
  * Try a different camera index (0, 1, 2…) if your script supports it
* **Installation issues**

  * Create a clean venv and reinstall requirements
  * If you’re on macOS/Linux and OpenCV fails, install OS deps first (varies by distro)
* **Model load errors (`.h5`)**

  * Ensure your TensorFlow/Keras version matches what the model expects (pin in `requirements.txt`)

---

## Roadmap (nice upgrades)

* Add a short **About** line + topics on GitHub (currently empty) 
* Add a `labels.json` / `classes.txt` so users know what gestures are supported
* Add a demo GIF in the README
* Add a `--model` flag and `--camera-index` flag for `app.py`
* Add a `requirements-lock.txt` (or `uv.lock`/`poetry.lock`) for reproducible installs

---

## Contributing

PRs welcome—especially:

* Better docs + examples
* Performance improvements (FPS, latency)
* Adding new gestures + retraining notebook improvements

---

## License

No license file is currently present in the repo view. Consider adding one (MIT/Apache-2.0/GPL) to clarify reuse. 

