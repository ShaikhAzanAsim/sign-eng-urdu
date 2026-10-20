# Bilingual Sign Language Alphabet Recognizer

Real-time webcam app with two modes: **English (ASL fingerspelling)** and **Urdu (Pakistan Sign Language alphabet)**.

Webcam frame → MediaPipe Tasks `HandLandmarker` (21 3D points) → shared normalize → language-specific static Random Forest + DTW for motion letters.

This project uses the existing venv at:

`C:\Users\QBS PC\PycharmProjects\ais-handler-template\.venv`

Wrappers `python.cmd` and `py.ps1` in this folder call that interpreter.

## Setup (once)

```powershell
cd C:\Users\QBS PC\PycharmProjects\sign-urdu-eng
.\py.ps1 -m pip install -r requirements.txt
.\py.ps1 -m src.download_model
.\py.ps1 -m src.urdu_text_render
```

## Run

Live landmarks only:

```powershell
.\py.ps1 run_overlay.py
```

Full app (mode toggle, sentence buffer; classifiers load if you have trained them):

```powershell
.\py.ps1 run_app.py
```

Keys in the OpenCV window: `m` English ⇄ Urdu, `space` word break, `backspace`, `c` clear, `q` quit.

## Record + train

Target **150–300 samples per static letter**. Urdu takes longer (more letters, no large public landmark set).

```powershell
.\py.ps1 -m src.data_collection --lang english
.\py.ps1 -m src.train_static --lang english
.\py.ps1 -m src.train_dynamic_templates --lang english

.\py.ps1 -m src.data_collection --lang urdu
.\py.ps1 -m src.train_static --lang urdu
.\py.ps1 -m src.train_dynamic_templates --lang urdu
```

Data collection: `r` starts/stops recording, `n`/`p` change letter.

Normalization self-test (no camera):

```powershell
.\py.ps1 -m src.normalization
.\py.ps1 -m src.alphabet
```

## Alphabets

- English: 26 ASL letters. **Dynamic: J, Z.**
- Urdu: UAlpha40 — **40 letters, 36 static + 4 dynamic**. Dynamic: **آ Alif mad, ج Jeem, ڑ Aray, ہ Hay (heh-goal)**. The 2021 37-letter photo chart is the visual reference for overlapping shapes; it does not include آ, ں, or ھ.

## Notes

- Landmark extraction and normalization are shared. Mode switch only swaps label maps + model files.
- `cv2.putText` cannot render Urdu. The HUD uses Pillow + `arabic-reshaper` + `python-bidi` and Noto Nastaliq Urdu.
- Static English clusters that usually confuse: M/N/S and U/V/R/K. That is expected, not automatically a pipeline bug.
