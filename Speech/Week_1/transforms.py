from functools import partial

import librosa
import numpy as np
import scipy


class Sequential:
    def __init__(self, *args):
        self.transforms = args

    def __call__(self, inp: np.ndarray):
        res = inp
        for transform in self.transforms:
            res = transform(res)
        return res


class Windowing:
    def __init__(self, window_size=1024, hop_length=None):
        self.window_size = window_size
        self.hop_length = hop_length if hop_length else self.window_size // 2
    
    def __call__(self, waveform):
        waveform = np.asarray(waveform, dtype=float)
        win_size = self.window_size
        hop = self.hop_length

        pad = win_size // 2
        padded = np.pad(waveform, (pad, pad), mode='constant', constant_values=0)

        n_windows = (len(waveform)-win_size%2)//hop+1

        windows = np.zeros((n_windows, win_size), dtype=float)
        for i in range(n_windows):
            start = i*hop
            windows[i]=padded[start:start+win_size]

        return windows
    

class Hann:
    def __init__(self, window_size=1024):
        self.window_size = window_size
        # so just make edges near zero 
        n = np.arange(window_size)
        self.hann_window = 0.5 * (1 - np.cos(2 * np.pi * n / window_size)).astype(np.float32)

    def __call__(self, windows):
        return (windows.astype(np.float32) * self.hann_window)



class DFT:
    def __init__(self, n_freqs=None):
        self.n_freqs = n_freqs

    def __call__(self, windows):
        windows = np.asarray(windows, dtype=float)
        n_windows, N = windows.shape

        K = N // 2 + 1

        if self.n_freqs is not None:
            K = min(K, self.n_freqs)

        k = np.arange(N)
        n = np.arange(K).reshape(-1, 1)

        W = np.exp(-2j * np.pi * n * k / N)

        spec = windows @ W.T

        return np.abs(spec)             


class Square:
    def __call__(self, array):
        return np.square(array)


class Mel:
    def __init__(self, n_fft, n_mels=80, sample_rate=22050):
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate

        self.n_freqs = n_fft // 2 + 1

        # fmin=1, fmax=8192
        self.mel_fb = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=1,
            fmax=8192
        ).astype(np.float64)

        self.mel_fb_pinv = np.linalg.pinv(self.mel_fb)
        
    def __call__(self, spec):
        spec = np.asarray(spec, dtype=np.float64)
        mel = spec @ self.mel_fb.T
        return mel
        
    def restore(self, mel):
        mel = np.asarray(mel, dtype=np.float64)
        spec = mel @ self.mel_fb_pinv.T
        return spec




class GriffinLim:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.griffin_lim = partial(
            librosa.griffinlim,
            n_iter=32,
            hop_length=hop_length,
            win_length=window_size,
            n_fft=window_size,
            window='hann'
        )

    def __call__(self, spec):
        return self.griffin_lim(spec.T)


class Wav2Spectrogram:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.windowing = Windowing(window_size=window_size, hop_length=hop_length)
        self.hann = Hann(window_size=window_size)
        self.fft = DFT(n_freqs=n_freqs)
        # self.square = Square()
        self.griffin_lim = GriffinLim(window_size=window_size, hop_length=hop_length, n_freqs=n_freqs)

    def __call__(self, waveform):
        return self.fft(self.hann(self.windowing(waveform)))

    def restore(self, spec):
        return self.griffin_lim(spec)


class Wav2Mel:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None, n_mels=80, sample_rate=22050):
        self.wav_to_spec = Wav2Spectrogram(
            window_size=window_size,
            hop_length=hop_length,
            n_freqs=n_freqs)
        self.spec_to_mel = Mel(
            n_fft=window_size,
            n_mels=n_mels,
            sample_rate=sample_rate)

    def __call__(self, waveform):
        return self.spec_to_mel(self.wav_to_spec(waveform))

    def restore(self, mel):
        return self.wav_to_spec.restore(self.spec_to_mel.restore(mel))


class TimeReverse:
    def __call__(self, mel):
        mel = np.asarray(mel)
        return mel[::-1]
        # like just do -1 like in python array



class Loudness:
    def __init__(self, loudness_factor):
        self.factor = float(loudness_factor)
        # >1 louder, <1 quiter

    def __call__(self, mel):
        mel = np.asarray(mel)
        return mel * self.factor
        # just multiply




class PitchUp:
    def __init__(self, num_mels_up):
        self.shift = int(num_mels_up)

    def __call__(self, mel):
        mel = np.asarray(mel)
        T, M = mel.shape
        out = np.zeros_like(mel)

        if self.shift >= M:
            return out

        out[:, self.shift:] = mel[:, :M - self.shift]
        return out
        # moving all notes up the ladder.



class PitchDown:
    def __init__(self, num_mels_down):
        self.shift = int(num_mels_down)

    def __call__(self, mel):
        mel = np.asarray(mel)
        T, M = mel.shape
        out = np.zeros_like(mel)

        if self.shift >= M:
            return out

        out[:, :M - self.shift] = mel[:, self.shift:] # same but there you do M - shift
        return out


'''
The number of mel frames equals to int(speed_up_factor * number_of_mel_frames). And the frames are copied. Each frame with index idx is copied to round(idx * speed_up_factor)
'''
class SpeedUpDown:
    def __init__(self, speed_up_factor=1.0):
        self.factor = float(speed_up_factor)

    def __call__(self, mel):
        mel = np.asarray(mel)
        T, M = mel.shape
        print(T, M)

        new_T = int(self.factor * T) #  -> int(speed_up_factor * number_of_mel_frames)
        if new_T <= 0:
            new_T = 1

        out = np.zeros((new_T, M), dtype=mel.dtype)

        for idx in range(T):
            dst = round(idx * self.factor) # -> round(idx * speed_up_factor)
            # print(idx, dst)
            if dst < new_T:
                out[dst] = mel[idx]

        # T = mel.shape[0]
        # for idx in range(T-10, T):
        #     print(idx, int(np.round(idx*self.factor)))

        return out
        # factor > 1 → faster (shorter)
        # factor < 1 → slower (longer) 



class FrequenciesSwap:
    def __call__(self, mel):
        mel = np.asarray(mel)
        return mel[:, ::-1]
        # bass becomes treble, treble becomes bass
        


class WeakFrequenciesRemoval:
    def __init__(self, quantile=0.05):
        self.q = float(quantile)

    def __call__(self, mel):
        mel = np.asarray(mel)
        thresh = np.quantile(mel, self.q)
        out = mel.copy()
        out[out < thresh] = 0.0
        return out
        # like removing the noise



class Cringe1:
    def __init__(self, drop_prob=0.2):
        self.drop_prob = drop_prob

    def __call__(self, mel):
        mel = np.asarray(mel)
        T, M = mel.shape
        mask = (np.random.rand(M) > self.drop_prob).astype(mel.dtype) # like randomly removing some bins (ubrupted audio)
        return mel * mask


class Cringe2:
    def __init__(self, max_width=10):
        self.max_width = max_width

    def __call__(self, mel):
        mel = np.asarray(mel)
        T, M = mel.shape
        out = mel.copy()

        width = np.random.randint(1, min(self.max_width, T))
        start = np.random.randint(0, T - width + 1)

        out[start:start+width] = 0 # same as above but randomly remove chunks like 1 sec etc. which makes it stop mid sentence.
        return out

