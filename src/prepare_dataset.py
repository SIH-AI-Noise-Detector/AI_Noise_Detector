import librosa

audio_path = "data/raw/LibriSpeech/train-clean-5/1088/134315/1088-134315-0000.flac"

audio, sample_rate = librosa.load(audio_path, sr=None)

duration = len(audio) / sample_rate

print("Sample rate:", sample_rate, "Hz")
print("Number of samples:", len(audio))
print("Duration:", round(duration, 2), "seconds")
print("Audio shape:", audio.shape)