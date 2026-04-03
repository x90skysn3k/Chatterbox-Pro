#!/usr/bin/env python3
"""
Quick A/B model comparison script — runs directly on P40.
Generates the same test sentences with each available model,
saves WAVs + an HTML report for side-by-side listening.

Usage:
  python3 test-models.py                    # all models, all tests
  python3 test-models.py --quick            # 3 tests only
  python3 test-models.py --models standard,turbo
"""

import argparse
import json
import os
import time
import datetime
import torch
import torchaudio
import gc

VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
DEFAULT_VOICE = os.path.join(VOICES_DIR, os.environ.get("DEFAULT_VOICE", "executed-edge.wav"))

# Test sentences
TESTS = [
    {
        "id": "simple-narration",
        "text": "Most traders lose money not because they lack skill, but because they lack discipline.",
        "difficulty": "easy",
    },
    {
        "id": "emotional-range",
        "text": "He watched his account drop from fifty thousand to twelve hundred in six weeks. His wife didn't know. His kids didn't know. He sat in his car in the driveway for forty minutes before walking inside.",
        "difficulty": "hard",
    },
    {
        "id": "short-punchy",
        "text": "Stop. Think about that for a second. Your broker made money. The exchange made money. You lost money.",
        "difficulty": "medium",
    },
    {
        "id": "numbers-and-stats",
        "text": "Exposed after three hundred days of trading. Ninety seven percent of retail traders lose money within their first year.",
        "difficulty": "medium",
    },
    {
        "id": "proper-names",
        "text": "Jesse Livermore once said that the market is designed to fool most of the people, most of the time. Paul Tudor Jones agreed.",
        "difficulty": "hard",
    },
    {
        "id": "technical-terms",
        "text": "The Vix spiked above thirty five while the S and P futures were limit down. Every algos risk model screamed to liquidate.",
        "difficulty": "hard",
    },
    {
        "id": "long-complex",
        "text": "What separates the five percent who survive from the ninety five percent who blow up isn't intelligence, it isn't access to better tools, and it certainly isn't some secret indicator that nobody else knows about, it's the ability to sit on your hands when every fiber of your being screams at you to click that button.",
        "difficulty": "hard",
    },
]

QUICK_IDS = {"simple-narration", "emotional-range", "short-punchy"}

# Model configs
MODELS = {
    "standard": {
        "label": "Original (500M)",
        "temperature": 0.75,
        "exaggeration": 0.65,
        "cfg_weight": 0.4,
    },
    "turbo": {
        "label": "Turbo (350M)",
        "temperature": 0.8,
        "exaggeration": 0.0,
        "cfg_weight": 0.0,
    },
    "multilingual": {
        "label": "Multilingual (500M)",
        "temperature": 0.75,
        "exaggeration": 0.65,
        "cfg_weight": 0.4,
    },
}


def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def _patch_turbo_dtype():
    """Patch ChatterboxTurboTTS.prepare_conditionals to cast audio to float32.
    Upstream bug: librosa returns float64, but s3tokenizer's mel filter bank is float32,
    causing 'expected scalar type Double but found Float' in matmul.
    """
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        if getattr(ChatterboxTurboTTS, '_dtype_patched', False):
            return
        import numpy as np
        original_prepare = ChatterboxTurboTTS.prepare_conditionals
        def patched_prepare(self, wav_fpath, exaggeration=0.5, norm_loudness=True):
            # Temporarily monkey-patch librosa.load to return float32
            import librosa
            orig_load = librosa.load
            def f32_load(*args, **kwargs):
                y, sr = orig_load(*args, **kwargs)
                return y.astype(np.float32), sr
            librosa.load = f32_load
            try:
                # Also patch librosa.resample to return float32
                orig_resample = librosa.resample
                def f32_resample(*args, **kwargs):
                    return orig_resample(*args, **kwargs).astype(np.float32)
                librosa.resample = f32_resample
                result = original_prepare(self, wav_fpath, exaggeration=exaggeration, norm_loudness=norm_loudness)
            finally:
                librosa.load = orig_load
                librosa.resample = orig_resample
            return result
        ChatterboxTurboTTS.prepare_conditionals = patched_prepare
        ChatterboxTurboTTS._dtype_patched = True
        print("  [patch] turbo prepare_conditionals: float64→float32 fix applied")
    except ImportError:
        pass
    except Exception as e:
        print(f"  [patch] turbo dtype patch failed: {e}")


def load_model(model_type, device):
    """Load a Chatterbox model variant."""
    free_vram()
    _patch_turbo_dtype()
    print(f"  Loading {model_type} model... (VRAM: {get_vram_mb():.0f}MB)")

    if model_type == "standard":
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device)
    elif model_type == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_pretrained(device)
    elif model_type == "multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        model = ChatterboxMultilingualTTS.from_pretrained(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"  Loaded! (VRAM: {get_vram_mb():.0f}MB)")
    return model


def generate_audio(model, model_type, text, voice_path, config, device):
    """Generate audio from a model, return (wav_tensor, sample_rate, duration_seconds)."""
    kwargs = {
        "text": text,
        "audio_prompt_path": voice_path,
        "temperature": config["temperature"],
    }

    # Standard and multilingual support exaggeration + cfg_weight
    if model_type in ("standard", "multilingual"):
        kwargs["exaggeration"] = config["exaggeration"]
        kwargs["cfg_weight"] = config["cfg_weight"]

    # Multilingual requires language_id
    if model_type == "multilingual":
        kwargs["language_id"] = "en"

    with torch.inference_mode():
        wav = model.generate(**kwargs)

    if isinstance(wav, torch.Tensor):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        sr = model.sr
        duration = wav.shape[-1] / sr
        return wav, sr, duration

    return wav, getattr(model, "sr", 24000), 0


def unload_model(model):
    """Unload model and free VRAM."""
    del model
    free_vram()
    print(f"  Model unloaded (VRAM: {get_vram_mb():.0f}MB)")


def generate_html_report(results, run_dir, models_tested):
    """Generate HTML comparison report."""
    # Group by test
    tests_in_order = []
    seen = set()
    for r in results:
        if r["test_id"] not in seen:
            tests_in_order.append(r["test_id"])
            seen.add(r["test_id"])

    test_lookup = {t["id"]: t for t in TESTS}

    rows_html = []
    for test_id in tests_in_order:
        test = test_lookup.get(test_id, {"text": "", "difficulty": "?"})
        cells = []
        for model_id in models_tested:
            r = next((x for x in results if x["test_id"] == test_id and x["model_id"] == model_id), None)
            if r and r.get("audio_path") and os.path.exists(r["audio_path"]):
                rel = os.path.relpath(r["audio_path"], run_dir)
                dur = r.get("duration", 0)
                gen_t = r.get("gen_time", 0)
                cells.append(f"""<td class="cell">
                    <audio controls preload="none" src="{rel}"></audio>
                    <div class="metrics">Duration: {dur:.1f}s | Gen: {gen_t:.1f}s</div>
                    <input class="notes" placeholder="Notes..." oninput="localStorage.setItem('n-{test_id}-{model_id}',this.value)"
                      value="">
                </td>""")
            else:
                err = r.get("error", "not generated") if r else "not generated"
                cells.append(f'<td class="cell"><em>{err}</em></td>')

        text_preview = test.get("text", "")[:100]
        rows_html.append(f"""<tr>
            <td><strong>{test_id.replace('-',' ')}</strong>
            <br><small>{test.get('difficulty','?')}</small>
            <br><small class="text-preview">{text_preview}...</small></td>
            {"".join(cells)}
        </tr>""")

    # Model averages
    summary_cards = []
    for model_id in models_tested:
        mr = [r for r in results if r["model_id"] == model_id and r.get("duration", 0) > 0]
        if mr:
            avg_dur = sum(r["duration"] for r in mr) / len(mr)
            avg_gen = sum(r["gen_time"] for r in mr) / len(mr)
            total_gen = sum(r["gen_time"] for r in mr)
            label = MODELS.get(model_id, {}).get("label", model_id)
            summary_cards.append(f"""<div class="card">
                <div class="label">{label}</div>
                <div class="value">{avg_dur:.1f}s avg</div>
                <div class="detail">{avg_gen:.0f}s avg gen | {total_gen:.0f}s total | {len(mr)} clips</div>
            </div>""")

    header_cols = "".join(f'<th>{MODELS.get(m,{}).get("label",m)}</th>' for m in models_tested)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>TTS Model Comparison</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,system-ui,sans-serif;background:#0a0a0f;color:#e0e0e0;padding:20px}}
h1{{color:#00e5ff;margin-bottom:5px}}
.sub{{color:#888;font-size:.9em;margin-bottom:20px}}
.summary{{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap}}
.card{{background:#151520;border:1px solid #333;border-radius:8px;padding:12px 16px;min-width:200px}}
.card .label{{color:#888;font-size:.8em;text-transform:uppercase}}
.card .value{{color:#00e5ff;font-size:1.3em;font-weight:bold;margin-top:4px}}
.card .detail{{color:#aaa;font-size:.75em;margin-top:2px}}
table{{width:100%;border-collapse:collapse;margin-top:15px}}
th{{background:#151520;color:#00e5ff;padding:10px 8px;text-align:left;border-bottom:2px solid #333;font-size:.85em}}
td{{padding:10px 8px;border-bottom:1px solid #222;vertical-align:top}}
tr:hover td{{background:#12121a}}
.cell{{min-width:240px}}
audio{{width:100%;max-width:240px;height:32px}}
.metrics{{font-size:.75em;color:#aaa;margin-top:4px}}
.notes{{width:100%;background:#1a1a2a;border:1px solid #333;color:#e0e0e0;border-radius:4px;padding:4px 6px;margin-top:4px;font-size:.75em}}
.text-preview{{color:#555;font-style:italic}}
</style></head>
<body>
<h1>TTS Model Comparison</h1>
<p class="sub">Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | Voice: {os.path.basename(DEFAULT_VOICE)}</p>
<div class="summary">{"".join(summary_cards)}</div>
<table>
<thead><tr><th style="width:250px">Test</th>{header_cols}</tr></thead>
<tbody>{"".join(rows_html)}</tbody>
</table>
<script>
document.querySelectorAll('.notes').forEach(el=>{{
  const k=el.getAttribute('oninput').match(/'([^']+)'/)[1];
  const v=localStorage.getItem(k);if(v)el.value=v;
}});
</script>
</body></html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Quick TTS model A/B comparison")
    parser.add_argument("--quick", action="store_true", help="Only 3 test sentences")
    parser.add_argument("--models", default="standard,turbo", help="Comma-separated model list")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Voice reference WAV")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    models_to_test = [m.strip() for m in args.models.split(",")]
    tests = TESTS if not args.quick else [t for t in TESTS if t["id"] in QUICK_IDS]

    print(f"\n{'='*50}")
    print(f"TTS Model Comparison")
    print(f"{'='*50}")
    print(f"Models: {', '.join(models_to_test)}")
    print(f"Tests:  {len(tests)}")
    print(f"Device: {args.device}")
    print(f"Voice:  {args.voice}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB total")
    print(f"{'='*50}\n")

    if not os.path.exists(args.voice):
        print(f"ERROR: Voice file not found: {args.voice}")
        return

    # Output directory
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qa-compare", ts)
    os.makedirs(run_dir, exist_ok=True)

    results = []
    total_start = time.time()

    for model_id in models_to_test:
        config = MODELS.get(model_id)
        if not config:
            print(f"WARNING: Unknown model '{model_id}', skipping")
            continue

        print(f"\n--- {config['label']} ({model_id}) ---")

        # Load model
        try:
            model = load_model(model_id, args.device)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            for test in tests:
                results.append({"test_id": test["id"], "model_id": model_id, "error": str(e)})
            continue

        # Create model output dir
        model_dir = os.path.join(run_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)

        for test in tests:
            wav_path = os.path.join(model_dir, f"{test['id']}.wav")
            print(f"  [{test['id']}] ", end="", flush=True)

            try:
                gen_start = time.time()
                wav, sr, duration = generate_audio(
                    model, model_id, test["text"], args.voice, config, args.device
                )
                gen_time = time.time() - gen_start

                # Save WAV
                if isinstance(wav, torch.Tensor):
                    torchaudio.save(wav_path, wav.cpu(), sr)
                    duration = wav.shape[-1] / sr

                print(f"OK  dur={duration:.1f}s  gen={gen_time:.1f}s  VRAM={get_vram_mb():.0f}MB")

                results.append({
                    "test_id": test["id"],
                    "model_id": model_id,
                    "audio_path": wav_path,
                    "duration": duration,
                    "gen_time": gen_time,
                })

            except Exception as e:
                print(f"FAILED: {e}")
                results.append({
                    "test_id": test["id"],
                    "model_id": model_id,
                    "error": str(e),
                })

        # Unload model before loading next
        unload_model(model)

    total_time = time.time() - total_start

    # Save results JSON
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate HTML report
    html = generate_html_report(results, run_dir, models_to_test)
    report_path = os.path.join(run_dir, "report.html")
    with open(report_path, "w") as f:
        f.write(html)

    # Summary
    print(f"\n{'='*50}")
    print(f"DONE in {total_time:.0f}s")
    print(f"{'='*50}")
    for model_id in models_to_test:
        mr = [r for r in results if r["model_id"] == model_id and r.get("duration", 0) > 0]
        failed = [r for r in results if r["model_id"] == model_id and r.get("error")]
        if mr:
            avg_dur = sum(r["duration"] for r in mr) / len(mr)
            avg_gen = sum(r["gen_time"] for r in mr) / len(mr)
            label = MODELS.get(model_id, {}).get("label", model_id)
            print(f"  {label}: {len(mr)} ok, {len(failed)} failed | avg dur {avg_dur:.1f}s | avg gen {avg_gen:.0f}s")

    print(f"\nResults: {run_dir}")
    print(f"Report:  {report_path}")
    print(f"\nTo view: scp root@your-server:{run_dir}/ ./qa-results/ && open ./qa-results/report.html")


if __name__ == "__main__":
    main()
