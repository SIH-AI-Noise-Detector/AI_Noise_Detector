# AI/ML Enabled Adaptive Noise Cancellation for Defence Communications

## 1. Project Overview

This project develops an AI/ML-based adaptive noise cancellation system designed to suppress stationary, non-stationary, and impulsive environmental noise while preserving speech intelligibility.

The prototype focuses on defence communication scenarios involving noisy acoustic environments such as helicopter/engine noise, sirens, aircraft, trains, chainsaws, fireworks, and vehicle horns.

---

## 2. Objective

The main objective is to develop a speech enhancement pipeline that:

- Reduces environmental acoustic noise.
- Preserves important speech information.
- Uses deep learning for speech enhancement.
- Supports deployment on embedded AI hardware.
- Uses INT8 quantization for efficient edge inference.

---

## 3. Dataset

The prototype uses:

- Mini LibriSpeech train-clean-5 for clean speech.
- ESC-50 for environmental noise.

Selected noise categories:

- Helicopter
- Engine
- Siren
- Airplane
- Train
- Chainsaw
- Fireworks
- Car horn

Total generated clean/noisy pairs:

**1,013**

Dataset split:

| Split | Samples |
|---|---:|
| Training | 810 |
| Validation | 101 |
| Testing | 102 |

Audio sampling rate:

**16 kHz**

Audio segment duration:

**2 seconds**

---

## 4. Preprocessing

The audio signal is converted into a Short-Time Fourier Transform (STFT) representation.

Configuration:

- FFT size: 512
- Hop length: 128
- Window: Hann
- Input representation: Real + Imaginary STFT components

Model input shape:

```text
(1, 2, 257, 251)
