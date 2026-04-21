"""
Klang validation — quantitative metrics for audio reconstructions.

Compares each candidate .wav against a reference (the original clip)
and emits JSON + prints a summary. Optional metrics (PESQ, STOI) are
used when the corresponding packages are importable; otherwise they
report "skipped".

Usage:
    python scripts/validate_klang.py                 # scan klang/ for .wav files
    python scripts/validate_klang.py --ref klang/original.wav
    python scripts/validate_klang.py --candidates klang/klang_1_2/decode_istft.wav

Metrics:
    - spectral_mse     Lower is better. MSE of magnitude STFT, normalized.
    - spectral_log_mae L1 of log-mag (perceptual proxy; lower better).
    - mcd13            Mel-cepstral distortion, 13 coeffs (lower better).
    - pesq             (optional) ITU-T P.862 wideband score [-0.5, 4.5].
    - stoi             (optional) short-time objective intelligibility [0, 1].

Gate thresholds (editable at top):
    gate A: candidate spectral_mse < BASELINE_MSE_CEIL  — not catastrophic
    gate B: candidate spectral_log_mae improves vs variant_b reference
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Gate thresholds. Tune empirically.
GATE_A_MSE_CEIL = 0.05       # normalized magnitude MSE
GATE_B_LOGMAE_IMPROVEMENT = 0.0  # must beat reference by >0


def _load_wav(path: str, sr: int) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path, sr=sr)
    return y


def _match_length(y_hat: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(y_hat), len(y))
    return y_hat[:n], y[:n]


def spectral_mse(y_hat: np.ndarray, y: np.ndarray, n_fft: int = 1024,
                 hop: int = 256) -> float:
    import librosa
    y_hat, y = _match_length(y_hat, y)
    mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    mag_hat = np.abs(librosa.stft(y_hat, n_fft=n_fft, hop_length=hop))
    # Normalize by reference max so different clips are comparable.
    scale = max(mag.max(), 1e-8)
    return float(np.mean(((mag - mag_hat) / scale) ** 2))


def spectral_log_mae(y_hat: np.ndarray, y: np.ndarray, n_fft: int = 1024,
                     hop: int = 256, eps: float = 1e-7) -> float:
    import librosa
    y_hat, y = _match_length(y_hat, y)
    mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    mag_hat = np.abs(librosa.stft(y_hat, n_fft=n_fft, hop_length=hop))
    return float(np.mean(np.abs(np.log(mag + eps) - np.log(mag_hat + eps))))


def mcd13(y_hat: np.ndarray, y: np.ndarray, sr: int) -> float:
    """Mel-cepstral distortion on 13 MFCCs. Not MCD-DTW; aligned."""
    import librosa
    y_hat, y = _match_length(y_hat, y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_hat = librosa.feature.mfcc(y=y_hat, sr=sr, n_mfcc=13)
    n = min(mfcc.shape[1], mfcc_hat.shape[1])
    diff = mfcc[:, :n] - mfcc_hat[:, :n]
    # Classic MCD constant 10 / ln(10) * sqrt(2) ≈ 6.142
    return float(6.142 * np.sqrt(np.sum(diff ** 2, axis=0)).mean())


def pesq_score(y_hat: np.ndarray, y: np.ndarray, sr: int) -> Optional[float]:
    try:
        from pesq import pesq
    except ImportError:
        return None
    y_hat, y = _match_length(y_hat, y)
    try:
        # PESQ expects 8k or 16k. Resample if needed.
        if sr not in (8000, 16000):
            import librosa
            y_r = librosa.resample(y, orig_sr=sr, target_sr=16000)
            yh_r = librosa.resample(y_hat, orig_sr=sr, target_sr=16000)
            return float(pesq(16000, y_r, yh_r, "wb"))
        return float(pesq(sr, y, y_hat, "wb" if sr == 16000 else "nb"))
    except Exception:
        return None


def stoi_score(y_hat: np.ndarray, y: np.ndarray, sr: int) -> Optional[float]:
    try:
        from pystoi import stoi
    except ImportError:
        return None
    y_hat, y = _match_length(y_hat, y)
    try:
        return float(stoi(y, y_hat, sr, extended=False))
    except Exception:
        return None


def evaluate(candidate_path: str, ref_path: str, sr: int) -> dict:
    y = _load_wav(ref_path, sr)
    y_hat = _load_wav(candidate_path, sr)
    out = {
        "candidate": candidate_path,
        "spectral_mse": spectral_mse(y_hat, y),
        "spectral_log_mae": spectral_log_mae(y_hat, y),
        "mcd13": mcd13(y_hat, y, sr),
    }
    pesq_v = pesq_score(y_hat, y, sr)
    stoi_v = stoi_score(y_hat, y, sr)
    out["pesq"] = pesq_v
    out["stoi"] = stoi_v
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Klang validation")
    p.add_argument("--ref", default="klang/original.wav",
                   help="Reference (ground truth) .wav")
    p.add_argument("--candidates", nargs="+", default=None,
                   help="Explicit candidate .wav paths. If omitted, scans "
                        "klang/ for known decode outputs.")
    p.add_argument("--reference-for-gates", default=None,
                   help="A second .wav to use as the 'baseline' that Klang "
                        "1.2 must beat (e.g. klang/variant_b_20L/audio.wav).")
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--output", default="results/klang_validation.json")
    return p.parse_args()


def _discover_candidates() -> list[str]:
    root = Path("klang")
    patterns = [
        "klang_1_2/decode_istft.wav",
        "klang_1_2/decode_griffinlim.wav",
        "klang_1_2/decode_hifigan.wav",
        "variant_b_10L/audio.wav",
        "variant_b_20L/audio.wav",
        "variant_b_40L/audio.wav",
    ]
    found = []
    for pat in patterns:
        p = root / pat
        if p.exists():
            found.append(str(p))
    # Top-level decodes
    for p in sorted(root.glob("*.wav")):
        if p.name in ("original.wav", "test_clip.wav"):
            continue
        found.append(str(p))
    return found


def main():
    args = parse_args()
    if not Path(args.ref).exists():
        print(f"Reference not found: {args.ref}")
        sys.exit(2)

    candidates = args.candidates or _discover_candidates()
    if not candidates:
        print("No candidate .wav files found.")
        sys.exit(2)

    print(f"Reference: {args.ref}")
    print(f"Evaluating {len(candidates)} candidate(s)...\n")

    results = []
    for c in candidates:
        if not Path(c).exists():
            print(f"  SKIP (missing): {c}")
            continue
        r = evaluate(c, args.ref, args.sr)
        results.append(r)
        pesq_s = f"{r['pesq']:.2f}" if r["pesq"] is not None else "—"
        stoi_s = f"{r['stoi']:.3f}" if r["stoi"] is not None else "—"
        print(
            f"  {Path(c).name:40s}  "
            f"mse={r['spectral_mse']:.5f}  "
            f"logmae={r['spectral_log_mae']:.3f}  "
            f"mcd={r['mcd13']:.2f}  "
            f"pesq={pesq_s}  stoi={stoi_s}"
        )

    # Gates (only applied if a reference-for-gates was named)
    gate_report = {}
    if args.reference_for_gates:
        ref_hit = next((r for r in results
                        if Path(r["candidate"]).resolve()
                        == Path(args.reference_for_gates).resolve()), None)
        klang12 = next((r for r in results
                         if "klang_1_2" in r["candidate"]), None)
        if ref_hit and klang12:
            gate_a = klang12["spectral_mse"] < GATE_A_MSE_CEIL
            gate_b = (ref_hit["spectral_log_mae"] - klang12["spectral_log_mae"]
                      > GATE_B_LOGMAE_IMPROVEMENT)
            gate_report = {
                "gate_A_mse_ceil": {
                    "candidate_mse": klang12["spectral_mse"],
                    "ceiling": GATE_A_MSE_CEIL,
                    "passed": gate_a,
                },
                "gate_B_logmae_improvement": {
                    "baseline": args.reference_for_gates,
                    "baseline_logmae": ref_hit["spectral_log_mae"],
                    "candidate_logmae": klang12["spectral_log_mae"],
                    "delta": ref_hit["spectral_log_mae"] - klang12["spectral_log_mae"],
                    "passed": gate_b,
                },
            }
            print("\nGates:")
            print(f"  [{ 'PASS' if gate_a else 'FAIL' }] A: spectral MSE ceiling")
            print(f"  [{ 'PASS' if gate_b else 'FAIL' }] B: log-MAE improvement vs {args.reference_for_gates}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "reference": args.ref,
            "results": results,
            "gates": gate_report,
        }, f, indent=2)
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
