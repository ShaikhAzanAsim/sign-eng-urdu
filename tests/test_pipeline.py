from src.alphabet import letters_of_kind, load_alphabet
from src.normalization import normalize_landmarks
from src.smoothing import PredictionSmoother


def test_scale_and_translation_invariance():
    import numpy as np

    rng = np.random.default_rng(1)
    canonical = rng.normal(size=(21, 3)).astype(np.float32)
    near = canonical * 0.4 + np.array([0.1, 0.2, 0.0], dtype=np.float32)
    far = canonical * 1.1 + np.array([0.7, 0.05, 0.0], dtype=np.float32)
    d = np.max(np.abs(normalize_landmarks(near).vector_63 - normalize_landmarks(far).vector_63))
    assert d < 1e-5


def test_alphabet_inventories():
    english = load_alphabet("english")
    urdu = load_alphabet("urdu")
    assert len(english["letters"]) == 26
    assert {letter["id"] for letter in letters_of_kind(english, "dynamic")} == {"J", "Z"}
    assert len(urdu["letters"]) == 40
    assert len(letters_of_kind(urdu, "static")) == 36
    assert {letter["id"] for letter in letters_of_kind(urdu, "dynamic")} == {
        "alif_mad",
        "jeem",
        "aray",
        "hay_goal",
    }


def test_smoother_emits_after_hold():
    s = PredictionSmoother(window=5, majority_frac=0.6, hold_seconds=0.2, cooldown_seconds=0.0)
    emitted = None
    for t in [0.0, 0.05, 0.10, 0.15, 0.25]:
        emitted = s.observe("A", t)
    assert emitted == "A"
    assert s.observe("A", 0.30) is None
    s.observe(None, 0.40)
    again = None
    for t in [0.45, 0.50, 0.55, 0.60, 0.70]:
        again = s.observe("B", t)
    assert again == "B"
