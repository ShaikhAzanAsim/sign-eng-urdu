# Bilingual Sign Language Alphabet Recognizer — Research & Project Plan
### English (ASL fingerspelling) + Urdu (Pakistan Sign Language) — dual-mode webcam app

---

## 1. Research summary (what's proven to work)

I looked at ~15 papers/writeups/repos on landmark-based sign alphabet recognition, plus the specific Urdu/PSL literature. The consistent, battle-tested pattern is:

**Webcam frame → hand landmark detector (21 keypoints/hand) → normalize landmarks → lightweight classifier → letter.**

Nobody doing this well runs a CNN directly on pixels anymore for alphabet-level signs. Reasons:
- Landmarks are invariant to background, skin tone, clothing, lighting — a raw-pixel CNN has to learn to ignore all that; a landmark model already ignores it.
- The feature vector is tiny (63 numbers for 21 points × x,y,z), so you can train a fast, accurate classifier on a few hundred samples per class instead of needing thousands of images.
- Reported accuracies: 86–99%+ depending on classifier (KNN/Random Forest/MLP/CNN-on-landmarks/LSTM), with MediaPipe + a simple dense NN or Random Forest regularly hitting 99%+ on static ASL letters, and MediaPipe + NN hitting 99.9% on static Urdu/PSL letters in one 2024 study.

**Important correction on MediaPipe itself:** the old `mediapipe.solutions.hands` API (what most tutorials/blog posts still show) is deprecated as of March 2023. Use the current **MediaPipe Tasks API** — `HandLandmarker` from `mediapipe.tasks.python.vision`, loading `hand_landmarker.task`. Same 21-point output, same model family, but it's the maintained API (GPU delegate support, cleaner running-mode handling for VIDEO vs LIVE_STREAM). I'll use this throughout.

**Alternatives to MediaPipe I checked, and why they lose:**

| Option | Verdict |
|---|---|
| **MediaPipe HandLandmarker (Tasks API)** | **Winner.** 21 3D landmarks/hand, runs real-time on CPU, actively maintained, Python/JS/C++/Android/iOS. |
| MediaPipe legacy `solutions.hands` | Deprecated, drawing utils removed in recent versions. Don't start new code on it. |
| MediaPipe **Gesture Recognizer** (Model Maker) | A built-in classifier on top of landmarks, trainable via transfer learning with their `mediapipe_model_maker`. Good shortcut, but it's a black box (fixed embedding), less flexible for two independent language modes and for the dynamic letters (J/Z, Urdu motion letters). I'll mention it as an optional fast-path, but the plan below uses your own classifier on raw landmarks for full control. |
| OpenPose | Heavier, GPU-hungry, no longer actively developed at the pace MediaPipe is. Was used in the *original* 2019 PSL Kaggle dataset (Saad Butt) — that's legacy, not a reason to use it now. |
| MMPose / YOLO-pose (body pose) | Overkill — these are full-body pose estimators. You only need hand keypoints; MediaPipe's dedicated hand model is faster and more accurate for this. |
| Raw CNN on cropped hand images | Works but needs way more data, is sensitive to background/lighting, and is slower to iterate on. Skip unless landmarks fail badly on some letters (see §5.3 fallback). |

**Conclusion: MediaPipe HandLandmarker (Tasks API) is genuinely the best tool here — the deprecation is a "use the new API" issue, not a "use a different library" issue.**

---

## 2. How the alphabets actually look (and why this matters for your architecture)

### 2.1 English — American Sign Language (ASL) fingerspelling
- 26 letters, one hand (right or left), no motion **except two letters**:
  - **J** — traced with a small hooking motion (pinky finger draws a "J" in the air).
  - **Z** — traced with the index finger drawing a "Z" shape in the air.
  - Every other letter (A, B, C, D, E, F, G, H, I, K–Y minus Z) is a **static hand shape** — one frame is enough to classify it.
- A few letter pairs are visually close and are the usual confusion points in every paper I read: **M vs N vs S** (fist variants with thumb position differing), **U vs V vs R vs K** (finger spacing/crossing), **G vs H** (orientation). Expect your confusion matrix to cluster here — this is normal, not a bug in your pipeline.

### 2.2 Urdu — Pakistan Sign Language (PSL) alphabet
This needed more digging since it's far less documented than ASL.
- The Urdu alphabet has **37–39 letters** depending on how compound/variant letters are counted (sources differ: 37 in the earliest PSL dataset paper, 39 in a 2024 web-app paper, 40 in the newest and most rigorous 2025 dataset **UAlpha40**).
- **UAlpha40 (2025, most authoritative)** explicitly splits it as: **36 static signs + 4 dynamic signs = 40 total**. This is the most useful reference because it explicitly separates static vs motion letters — mirroring the ASL J/Z situation, just with more dynamic letters. I'd anchor your Urdu label set on UAlpha40's list rather than an older, less complete dataset.
- Practical implication: your Urdu mode needs the **same two-tier design as English** (static classifier + small motion classifier for a handful of letters) — just with ~4 dynamic classes instead of 2.
- No single, universally-agreed, richly-sampled public PSL dataset exists the way ASL has (ASL alphabet datasets on Kaggle have 3000+ images/class from many contributors). PSL datasets that do exist are much smaller and less standardized:
  - *Pakistan Sign Language Dataset (OpenPose)* — Saad Butt, 2019, Kaggle — ~2000 images/37 letters + 700 word images, OpenPose skeleton JSON (not MediaPipe format).
  - *Dataset of Pakistan Sign Language...* (Mendeley/PMC, 2021) — 37 letters × 40 images each, small, single-orientation-focused.
  - **UAlpha40** (Mendeley, 2025) — 36 static (JPG) + 4 dynamic (MP4) classes, ~2,900 originally captured + heavy augmentation to 22,280 — best available, still modest for real-world webcam variety (9 contributors).
- **My recommendation:** don't build your Urdu classifier by downloading these and hoping the landmarks transfer cleanly (formats differ — one is OpenPose skeletons, not MediaPipe). Instead, **record your own MediaPipe-landmark dataset for Urdu directly** (same pipeline you'll use at inference time), using UAlpha40's letter list as your label reference and picture guide for correct hand shapes. This sidesteps format mismatches entirely and is exactly what most of these papers ended up doing anyway because "the dataset of PSL is not available publicly" (a near-verbatim line in two separate papers).
- Practically, for correct hand-shape references per Urdu letter while you record your own data, use the **UAlpha40 Mendeley page** and the **PSL Dictionary** as your visual reference (I'll link both below) rather than me trying to describe 37+ hand shapes in text — a photo reference is far more reliable for you to sign against correctly than a text description would be, and getting a letter's shape subtly wrong is exactly what causes bad training data.

### 2.3 One thing to sanity-check early
A couple of the PSL papers mention some Urdu letters use hand **orientation/rotation** more than English does (e.g., palm facing left vs right distinguishing letters), and MediaPipe gives you z-depth and full 3D landmark position, so your normalization step (§4) needs to preserve *rotation* information for Urdu even though for ASL you can often get away with rotation-normalizing more aggressively. Keep this in mind when designing features — don't blindly copy an ASL feature-normalization recipe onto Urdu.

---

## 3. Two-mode architecture

```
                     ┌──────────────────────┐
   Webcam frame  →   │  MediaPipe HandLandmarker │  →  21 (x,y,z) landmarks/hand
                     └──────────────────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Landmark normalization │
                     └──────────────────────┘
                                │
                     ┌──────────┴───────────┐
                     ▼                       ▼
            [Mode = English]         [Mode = Urdu]
                     │                       │
          ┌──────────┴─────────┐   ┌─────────┴──────────┐
          │ Static classifier   │   │ Static classifier   │
          │ (24 static letters) │   │ (~36 static letters)│
          ├──────────────────────┤   ├──────────────────────┤
          │ Motion classifier    │   │ Motion classifier    │
          │ (J, Z — 2 classes)   │   │ (~4 dynamic letters) │
          └──────────┬─────────┘   └─────────┬──────────┘
                      ▼                       ▼
                 Predicted letter (mapped through that mode's label file)
                      │
                      ▼
          Temporal smoothing → confirmed letter → word/sentence buffer → on-screen text (+ optional TTS)
```

Mode switch = just swaps which label map + which pair of trained model files (static + motion) are loaded. The landmark extraction and normalization code is 100% shared between modes — you're not duplicating the hard part.

---

## 4. Feature engineering (the part everyone gets subtly wrong)

Raw landmark (x, y, z) coordinates from MediaPipe are in normalized image coordinates — meaning they shift if the hand moves around the frame and scale if the hand moves closer/farther from the camera. Feed raw coordinates into a classifier and it'll overfit to "where in the frame was my hand" instead of "what shape is my hand," and it'll fall apart the moment you or someone else signs at a different distance/position.

**Standard fix, used in essentially every source I found:**
1. **Translate**: subtract the wrist landmark (landmark 0) from all 21 points, so the wrist becomes the origin. This removes position-in-frame dependence.
2. **Scale-normalize**: divide all coordinates by a reference distance (commonly the distance from wrist to middle-finger MCP joint, or the max distance among all landmarks from the wrist). This removes distance-from-camera dependence.
3. **Flatten** to a 63-length vector (21 × 3) — this is your feature vector, per hand, per frame.
4. **(Optional, more robust)**: also compute the angles between adjacent finger-joint vectors (e.g., angle at each knuckle). This is what the "Sign Language Recognition using MediaPipe" (DTW) project does, and it buys you rotation-invariance the raw-coordinate approach doesn't have. I'd add this as extra features (not a replacement) for the Urdu mode specifically, given the orientation-sensitivity note in §2.3 — but concatenate it, don't rely on it alone, since some Urdu letters *need* orientation to be distinguishable.
5. **Handedness**: MediaPipe tells you left/right hand. Mirror left-hand landmarks to right-hand convention (flip the x-axis) so you only ever need to train on one handedness and it still works for lefties.

For the **dynamic letters** (J, Z, and Urdu's ~4 motion letters), a single frame's feature vector isn't enough — you need a short sequence. Buffer the last N frames (e.g., 20–30 frames, ~1 second) of the fingertip trajectory (just the relevant fingertip's normalized x,y over time is often enough — you don't need full 63-dim sequences) and feed that sequence to a small temporal model.

---

## 5. Classifier choices

### 5.1 Static letters (majority of both alphabets)
Pick one — all are used successfully in the literature, in increasing order of complexity:

| Model | Pros | Cons | When to use |
|---|---|---|---|
| **Random Forest / KNN (scikit-learn)** | Fastest to build, trains in seconds, surprisingly high accuracy (paper reported 99%+ with RF), fully interpretable | Slightly less smooth probability calibration | **Recommended starting point** — get the whole pipeline working end-to-end with this first |
| **Small MLP (63/126 → 128 → 64 → num_classes, PyTorch or TF)** | Marginally higher ceiling, easy to export to ONNX/TFLite for mobile later | Needs a bit more data per class, more hyperparameter fiddling | Swap in once pipeline works and you want to squeeze accuracy |
| **CNN on landmark "image"** (some papers reshape landmarks into a small grid and CNN over it) | Can pick up subtler spatial relationships | Unnecessary complexity for 63 numbers | Skip — not worth it here |

Given you're comfortable with CV/ML already, I'd say: build the Random Forest version first (fast feedback loop on whether your data collection and normalization are correct), then swap the classifier for an MLP once the pipeline is validated — the feature extraction code doesn't change either way.

### 5.2 Dynamic letters (J/Z in English, ~4 in Urdu)
Two workable approaches:
- **Simple/robust: rule-based trajectory matching.** Track the relevant fingertip's normalized (x,y) path over your buffer window, resample to a fixed length, and compare against a stored reference trajectory per dynamic letter using **DTW (Dynamic Time Warping)** — cheap, no training needed, works well for just a handful of classes. This is what the Theodo/MediaPipe-DTW project does.
- **ML approach: small LSTM/GRU** over the sequence of (normalized landmark vectors or just fingertip path) per frame, trained like a mini seq2class model. More setup, marginally more robust to speed variation, but DTW is genuinely fine for 2–6 classes and is much less work.

**My recommendation given the class count you have (2 for English, ~4 for Urdu): DTW, not LSTM.** You don't have enough dynamic classes to justify training a neural sequence model, and DTW needs almost no data (a handful of reference recordings per dynamic letter).

### 5.3 Fallback if landmarks alone underperform on a specific letter
If, after building it, a handful of letters keep confusing each other (expect this around the M/N/S and U/V/R/K clusters in English), don't panic and don't switch architectures — first try: (a) adding the joint-angle features from §4.4, (b) collecting more varied training samples for just those confused classes (different hand rotations/distances), (c) as a last resort, a small secondary CNN on the cropped hand-bounding-box image *only* as a tiebreaker between the 2-3 confused classes, not as the primary classifier.

### 5.4 Optional shortcut: MediaPipe Model Maker's Gesture Recognizer
Google provides `mediapipe_model_maker.gesture_recognizer` — you give it images (or a landmark dataset), it fine-tunes an embedding + classifier head for you, and exports a `.task` file usable directly in the Tasks API. It's a legitimate faster path if you want something working quickly, but it's less transparent/tunable than rolling your own classifier on landmarks, and doesn't natively solve the dynamic-letter problem — you'd still need §5.2 for J/Z and the Urdu motion letters on top of it. I'd treat it as an experiment to run in parallel, not your primary path, given you want real control over two full alphabets.

---

## 6. Data collection plan

This is the part that determines your final accuracy more than any classifier choice.

**For English/ASL:** you have two options and I'd actually do both:
1. Use an existing public ASL alphabet image dataset (e.g., Kaggle's "ASL Alphabet" — ~3000 images/letter from a few contributors) — run it through MediaPipe once, offline, to extract landmark vectors, and use that as a base training set. This gets you volume and some contributor diversity fast.
2. **Also record your own webcam samples** (a few hundred frames per static letter, signing them yourself, varying hand distance/rotation/lighting a bit) — this matters because your specific webcam, lighting, and signing style will differ from the dataset's, and mixing in self-recorded data closes that domain gap significantly. This is standard practice in every one of these papers.

**For Urdu/PSL:** given the dataset scarcity discussed in §2.2, self-recording is not optional here — plan on it as the primary data source, using UAlpha40 and the PSL Dictionary purely as your *visual reference* for correct hand shapes, not as directly-ingestible training data (skeleton format mismatch). Budget meaningfully more data-collection time for Urdu than English.

**Per-class volume**: papers report good results from as few as 40 images/class (with augmentation) up to several thousand. With landmarks (not raw pixels) as your feature, **150–300 samples per static letter, collected across a few sessions with varied hand angle/distance/lighting**, is a reasonable practical target — far less than raw-pixel CNNs need, because you're not asking a model to learn "what a hand looks like," only "what this landmark configuration means."

**Augmentation on landmarks** (cheap, do this regardless of raw sample count): small random rotation/scale/translation jitter, and small Gaussian noise on each coordinate — this is how UAlpha40 inflated ~2,900 real samples to 22,280.

**Left/right hand**: record with your dominant hand, then generate a mirrored copy of every sample (flip x-coordinates) so the model handles both hands without doubling your actual recording effort.

---

## 7. Project structure

```
sign-lang-recognizer/
├── README.md
├── requirements.txt
├── models/
│   ├── hand_landmarker.task              # MediaPipe's pretrained hand landmark model (downloaded once)
│   ├── english/
│   │   ├── static_classifier.pkl         # or .onnx/.pt — trained RF/MLP for 24 static letters
│   │   ├── static_labels.json
│   │   └── dynamic_templates/            # reference DTW trajectories for J, Z
│   └── urdu/
│       ├── static_classifier.pkl
│       ├── static_labels.json            # Urdu letter name ↔ class index, incl. Urdu script + transliteration
│       └── dynamic_templates/            # reference DTW trajectories for the ~4 dynamic letters
│
├── data/
│   ├── raw_videos_or_images/
│   │   ├── english/<letter>/*.jpg
│   │   └── urdu/<letter>/*.jpg
│   └── landmarks/
│       ├── english/<letter>.csv          # extracted+normalized 63-dim vectors, one row per sample
│       └── urdu/<letter>.csv
│
├── src/
│   ├── landmark_extraction.py            # wraps MediaPipe HandLandmarker (Tasks API), shared by both modes
│   ├── normalization.py                  # translate/scale/angle features, handedness mirroring — shared
│   ├── data_collection.py                # webcam capture tool: press a key/label, save frames+landmarks
│   ├── dataset_build.py                  # turns raw captures (+ any public dataset) into landmarks/*.csv
│   ├── train_static.py                   # trains RF or MLP per language, saves to models/<lang>/
│   ├── train_dynamic_templates.py        # records/stores DTW reference trajectories per language
│   ├── dtw_match.py                      # DTW matching against dynamic templates, shared logic
│   ├── inference.py                      # real-time loop: webcam → landmarks → mode-specific classifier → letter
│   ├── smoothing.py                      # temporal majority-vote smoothing + hold-to-confirm buffer
│   ├── text_buffer.py                    # letters → word/sentence assembly, backspace/space/clear gestures
│   └── ui.py                             # app window: video feed, mode toggle (EN/UR), predicted letter, sentence bar
│
├── labels/
│   ├── english_alphabet.json             # A–Z, static/dynamic flag
│   └── urdu_alphabet.json                # Urdu letters (script + name), static/dynamic flag, sourced from UAlpha40 list
│
└── notebooks/
    ├── explore_landmarks.ipynb           # sanity-check extraction, visualize normalized landmarks
    └── confusion_analysis.ipynb          # per-class accuracy, confusion matrix, error inspection
```

---

## 8. Real-time inference details worth planning for now

- **Temporal smoothing**: a single frame's prediction will flicker between letters as your hand micro-moves. Keep a rolling buffer of the last ~10–15 predictions and only "confirm" a letter once it's the majority for, say, 70% of that window — this alone fixes most of the jitter you'll see in your first working version.
- **Hold-to-confirm for spelling words**: require a letter to be stably predicted for ~0.5–1 second before appending it to the sentence buffer, plus a distinct gesture (or a brief "no hand detected" gap) to signal "next letter" — otherwise your buffer fills with the same letter repeated 30 times.
- **No-hand / low-confidence state**: MediaPipe returns nothing if it doesn't detect a hand — handle that explicitly (don't crash, don't predict garbage) and use it as a natural "space between letters" signal if you want.
- **Urdu text rendering**: this will bite you if you don't plan for it — **OpenCV's `cv2.putText` cannot render Urdu/Arabic script** (no RTL shaping, no joined letterforms). You'll need `PIL`/`Pillow` with an Urdu-capable font (e.g., Noto Nastaliq Urdu or Jameel Noori Nastaleeq) plus `arabic_reshaper` + `python-bidi` to correctly shape and reorder the text before drawing it onto the OpenCV frame. Budget for this explicitly — it's a common last-minute surprise in these projects.
- **Two-hand support**: `HandLandmarker` supports `num_hands=2` out of the box. Neither alphabet's *letters* should need two hands (both are one-handed fingerspelling systems), but you may still want 2-hand detection enabled so the app doesn't break if the user's other hand is visible in frame — just ignore the second hand's landmarks for classification, or use its presence/absence as a UI gesture (e.g., "raise both hands" = clear sentence).

---

## 9. Suggested build order

1. **Landmark extraction + visualization** — get `HandLandmarker` running on your webcam, draw the 21 points live, confirm it's stable. (Half a day.)
2. **Normalization module** — implement translate/scale, verify by eye that the same letter signed at different distances/positions gives near-identical feature vectors. (Half a day.)
3. **Data collection tool** — simple script: press a key = start/stop recording landmark vectors for the currently-selected letter, saves to CSV. Use it to record English static letters first (fastest to validate the whole loop since you already know the shapes well). (1 day.)
4. **Train + validate static English classifier** (Random Forest first) — get a confusion matrix, sanity-check accuracy, fix normalization bugs if certain letters are hopeless. (1 day.)
5. **Real-time inference loop for English static letters**, with smoothing — get the full pipeline feeling responsive before adding complexity. (1 day.)
6. **Add J/Z via DTW** — record a handful of reference trajectories, implement matching, integrate into the mode's prediction logic. (1 day.)
7. **Repeat 3–6 for Urdu**, using UAlpha40/PSL Dictionary as your visual reference while recording (expect this to take longer — more letters, less familiarity, dataset scarcity means more self-recording). (3–5 days.)
8. **Mode toggle + UI polish** — label maps, sentence buffer, Urdu text rendering via PIL. (1–2 days.)
9. **Confusion analysis + targeted data collection** on whichever letter clusters are underperforming per language. (ongoing)

---

## 10. Key references found during research

- MediaPipe Hands landmark model overview (21-point ML pipeline) — Google/GitHub docs
- "Leveraging Google Mediapipe for easy hand gesture classification" — Medium (German sign alphabet, landmark→NN pattern)
- "Enhancing Sign Language Detection through Mediapipe and CNN" — arXiv 2406.03729 (99.12% ASL accuracy, landmarks+CNN)
- "Sign Language Recognition using MediaPipe" — Theodo (DTW-based dynamic sign matching, angle-based features)
- "Sign Language Recognition using MediaPipe and Random Forest" — Medium/Intel oneAPI (RF classifier approach)
- "A Vision Based Approach for Pakistan Sign Language alphabets Recognition" — ResearchGate (MediaPipe + landmark-trajectory-to-image approach for PSL)
- "Dataset of Pakistan Sign Language and Automatic Recognition of Hand Configuration of Urdu Alphabet through ML" — PMC/ScienceDirect (37-letter PSL dataset, webcam-collected, Android app)
- **UAlpha40: A comprehensive dataset of Urdu alphabet for Pakistan sign language** — PMC/ScienceDirect/Mendeley, 2025 (36 static + 4 dynamic = 40, most complete and recent — use as your label/reference source)
- "A neural-network based web application on real-time recognition of Pakistani sign language" — ScienceDirect, 2024 (99.9% static accuracy, 39 letters, MediaPipe + NN)
- MediaPipe Tasks `HandLandmarker` docs and Model Maker `gesture_recognizer` docs — Google AI Edge (current API surface, custom gesture training path)

---

If you want, next step I can help with is scaffolding the actual repo (folders + starter code for `landmark_extraction.py` and `normalization.py`) so you can start recording data today.
