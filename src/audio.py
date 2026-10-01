import torch


def audio_to_stft(audio, n_fft=512, hop_length=128):
    """
    Convert audio waveform into a complex STFT.

    audio:
        Tensor containing audio samples.

    Returns:
        Complex spectrogram.
    """

    window = torch.hann_window(n_fft)

    stft = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True
    )

    return stft