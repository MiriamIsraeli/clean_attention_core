"""
Test: noise + speech simulation — verify speech is fully preserved.
3 scenarios:
  1. Pure stationary noise → should be cancelled
  2. Speech-like signal alone → should NOT be affected
  3. Noise + speech mixed → noise cancelled, speech preserved
"""
import numpy as np
import sys
sys.path.insert(0, ".")
from algorithms import NoiseDetector

sr = 48000
duration = 3.0
t = np.arange(0, duration, 1/sr)

# === Stationary noise: 130Hz hum + broadband (like AC / engine) ===
rng = np.random.default_rng(42)
from scipy.signal import butter, lfilter
b, a = butter(4, 400/(sr/2))
broadband = lfilter(b, a, rng.normal(0, 0.15, len(t)))
hum = 0.1 * np.sin(2 * np.pi * 130 * t)
noise = broadband + hum

# === Speech-like signal: formants at ~500, 1500, 2500 Hz ===
# Simulates vowel "ah" with varying amplitude (natural speech envelope)
speech_env = np.zeros(len(t))
# Speech appears from 1.0s to 2.5s (after calibration)
speech_start = int(1.0 * sr)
speech_end = int(2.5 * sr)
speech_env[speech_start:speech_end] = np.hanning(speech_end - speech_start)

formants = (
    0.3 * np.sin(2 * np.pi * 500 * t) +
    0.2 * np.sin(2 * np.pi * 1500 * t) +
    0.1 * np.sin(2 * np.pi * 2500 * t)
)
speech = formants * speech_env

# === 3 test signals ===
signals = {
    "Pure noise": noise,
    "Pure speech": speech,
    "Noise + Speech": noise + speech,
}

chunk_size = 2304  # Same as server

for name, sig in signals.items():
    detector = NoiseDetector(fft_size=1024, threshold=0.3)
    cleaned = np.zeros(len(sig))
    for i in range(0, len(sig), chunk_size):
        end = min(i + chunk_size, len(sig))
        cleaned[i:end] = detector.update_and_extract(sig[i:end])
    
    # Analyze different time regions
    cal_end = int(0.3 * sr)
    noise_only = slice(int(0.5 * sr), int(0.9 * sr))  # Before speech
    speech_region = slice(int(1.2 * sr), int(2.3 * sr))  # During speech
    
    rms_in_noise = np.sqrt(np.mean(sig[noise_only]**2))
    rms_out_noise = np.sqrt(np.mean(cleaned[noise_only]**2))
    
    rms_in_speech = np.sqrt(np.mean(sig[speech_region]**2))
    rms_out_speech = np.sqrt(np.mean(cleaned[speech_region]**2))
    
    noise_red = 20 * np.log10(rms_out_noise / rms_in_noise) if rms_in_noise > 1e-10 else -99
    speech_red = 20 * np.log10(rms_out_speech / rms_in_speech) if rms_in_speech > 1e-10 else -99
    
    print(f"\n=== {name} ===")
    print(f"  Noise region (0.5-0.9s): in={rms_in_noise:.4f} out={rms_out_noise:.4f} change={noise_red:+.1f}dB")
    print(f"  Speech region (1.2-2.3s): in={rms_in_speech:.4f} out={rms_out_speech:.4f} change={speech_red:+.1f}dB")

# === Detailed: how much speech is preserved in the mixed signal? ===
print("\n=== SPEECH PRESERVATION ANALYSIS ===")
mixed = noise + speech
detector = NoiseDetector(fft_size=1024, threshold=0.3)
cleaned_mix = np.zeros(len(mixed))
for i in range(0, len(mixed), chunk_size):
    end = min(i + chunk_size, len(mixed))
    cleaned_mix[i:end] = detector.update_and_extract(mixed[i:end])

# Compare cleaned mix against pure speech in the speech region
speech_reg = slice(int(1.2 * sr), int(2.3 * sr))
pure_speech_rms = np.sqrt(np.mean(speech[speech_reg]**2))
cleaned_speech_component = cleaned_mix[speech_reg]

# Correlation between cleaned output and pure speech
corr = np.corrcoef(speech[speech_reg], cleaned_speech_component)[0, 1]
# How much of the speech energy remains?
speech_energy = np.sum(speech[speech_reg]**2)
# Project cleaned onto speech direction
projection = np.sum(cleaned_speech_component * speech[speech_reg]) / (speech_energy + 1e-10)

print(f"  Pure speech RMS: {pure_speech_rms:.4f}")
print(f"  Cleaned RMS (speech region): {np.sqrt(np.mean(cleaned_speech_component**2)):.4f}")
print(f"  Correlation (clean vs pure speech): {corr:.4f} (1.0 = perfect preservation)")
print(f"  Speech projection coefficient: {projection:.4f} (1.0 = speech fully kept)")
print(f"  Noise in cleaned (noise-only region): {np.sqrt(np.mean(cleaned_mix[int(0.5*sr):int(0.9*sr)]**2)):.4f}")

if corr > 0.9 and np.sqrt(np.mean(cleaned_mix[int(0.5*sr):int(0.9*sr)]**2)) < 0.02:
    print("\n  PASS: Speech preserved + noise cancelled!")
else:
    print(f"\n  NEEDS WORK: corr={corr:.3f} (want >0.9), noise_rms={np.sqrt(np.mean(cleaned_mix[int(0.5*sr):int(0.9*sr)]**2)):.4f} (want <0.02)")
