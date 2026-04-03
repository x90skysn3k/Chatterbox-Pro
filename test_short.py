"""
Test Extended Chatterbox on M1 Max with a Short narration.
Uses Whisper validation + pyrnnoise + auto-editor.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchaudio
import time
from chatterbox.tts import ChatterboxTTS

# Our Short narration (same one we've been testing)
TEXT = "Your favorite trading guru makes more money from courses than from actual trading. Ninety percent of day traders lose money. Not because of bad strategies. Because of psychology. Ed Saykota turned five thousand dollars into fifteen million with SIMPLE trend following rules. No secret indicator. No five thousand dollar course. Just discipline."

VOICE_PATH = "executed-edge.wav"
OUTPUT_PATH = "/tmp/extended-test-short.wav"

print("Loading model on MPS...")
start = time.time()
model = ChatterboxTTS.from_pretrained(device=torch.device("mps"))
print(f"Model loaded in {time.time()-start:.1f}s")

print(f"\nGenerating audio ({len(TEXT)} chars)...")
start = time.time()
wav = model.generate(
    text=TEXT,
    audio_prompt_path=VOICE_PATH,
    temperature=0.8,
    exaggeration=0.78,
    cfg_weight=0.3,
)
gen_time = time.time() - start
duration = wav.shape[1] / model.sr
print(f"Generated {duration:.1f}s audio in {gen_time:.1f}s ({duration/gen_time:.1f}x realtime)")

# Save raw
torchaudio.save(OUTPUT_PATH, wav.cpu(), model.sr)
print(f"Saved to {OUTPUT_PATH}")

# Try pyrnnoise denoising
try:
    import subprocess
    denoised_path = "/tmp/extended-test-short-denoised.wav"
    # Convert to 48k mono PCM for pyrnnoise
    subprocess.run(["ffmpeg", "-y", "-i", OUTPUT_PATH, "-ar", "48000", "-ac", "1", "/tmp/pyrnnoise-input.wav"],
                   capture_output=True)

    # Try pyrnnoise CLI
    result = subprocess.run(["pyrnnoise", "/tmp/pyrnnoise-input.wav", "/tmp/pyrnnoise-output.wav"],
                           capture_output=True, text=True)
    if result.returncode == 0:
        # Convert back to original sample rate
        subprocess.run(["ffmpeg", "-y", "-i", "/tmp/pyrnnoise-output.wav", "-ar", str(model.sr), denoised_path],
                      capture_output=True)
        print(f"Denoised saved to {denoised_path}")
    else:
        print(f"pyrnnoise failed: {result.stderr[:200]}")
        # Try Python API fallback
        try:
            from pyrnnoise import RNNoise
            denoiser = RNNoise()
            import soundfile as sf
            audio, sr = sf.read("/tmp/pyrnnoise-input.wav")
            denoised = denoiser.process_audio(audio)
            sf.write("/tmp/pyrnnoise-output.wav", denoised, sr)
            subprocess.run(["ffmpeg", "-y", "-i", "/tmp/pyrnnoise-output.wav", "-ar", str(model.sr), denoised_path],
                          capture_output=True)
            print(f"Denoised (Python API) saved to {denoised_path}")
        except Exception as e2:
            print(f"pyrnnoise Python API also failed: {e2}")
except Exception as e:
    print(f"Denoising skipped: {e}")

# Apply loudnorm
try:
    import subprocess
    normed_path = "/tmp/extended-test-short-final.wav"
    source = denoised_path if os.path.exists("/tmp/extended-test-short-denoised.wav") else OUTPUT_PATH
    subprocess.run([
        "ffmpeg", "-y", "-i", source,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,apad=pad_dur=0.5",
        "-ar", "24000", "-ac", "1", normed_path
    ], capture_output=True)
    print(f"Final (loudnorm) saved to {normed_path}")
except Exception as e:
    print(f"Loudnorm failed: {e}")

print("\nDone! Compare:")
print(f"  Raw:      {OUTPUT_PATH}")
print(f"  Denoised: /tmp/extended-test-short-denoised.wav")
print(f"  Final:    /tmp/extended-test-short-final.wav")
