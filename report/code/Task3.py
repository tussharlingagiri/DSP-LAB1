# Task 3 - aural exciter on the 5cm recording
# Based on the top path of the Aphex diagram: highpass the voice, push it through tanh
# to make harmonics, keep only the harmonics we want, then add a bit back to the dry voice.
# Every filter here is just FFT bins multiplied by 0 or 1, then IFFT.

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

FILE = 'Assignment_1_audio_5cm.wav'

# Each version is one step of the design. Notes are what we heard / saw.
VERSIONS = {
    'v1': dict(hp=1000, drive=10, harm_band=(1000, 22050), mix=0.5),   # fuzzy, sounds like static
    'v2': dict(hp=3000, drive=5,  harm_band=(4000, 10000), mix=0.2),   # clean but hard to tell from original
    'v3': dict(hp=3000, drive=5,  harm_band=(4000, 10000), mix=0.5),   # s a bit louder, otherwise same
    'v4': dict(hp=2000, drive=5,  harm_band=(4000, 10000), mix=0.5),   # no change from v3
    'v5': dict(hp=3000, drive=20, harm_band=(4000, 10000), mix=0.15),  # s in "Scotland" clearer, no fizz
}
SHOW = ('v1', 'v3', 'v5')    # only these go on the comparison plots, 5 lines was unreadable
FINAL = 'v5'

# read wav, take one channel if stereo, scale so the peak is 1
def load(path):
    fs, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float64)
    return fs, x / np.max(np.abs(x))

# one-sided magnitude spectrum in dB, DC bin dropped so the log axis works
def spectrum_dB(x, fs):
    N = len(x)
    X = np.fft.fft(x)[:N // 2] / N
    f = np.arange(N // 2) * fs / N
    return f[1:], 20 * np.log10(np.abs(X[1:]) + 1e-12)

# 1 for bins inside f_lo..f_hi, 0 elsewhere. We can't use fftfreq so the bin
# frequencies are worked out by hand. The top half of the FFT is the negative
# frequencies, so bin N-k mirrors bin k. Masking both keeps the IFFT output real.
def band_mask(N, fs, f_lo, f_hi):
    k = np.arange(N)
    f = np.minimum(k, N - k) * fs / N
    return ((f >= f_lo) & (f <= f_hi)).astype(float)

# bandpass by zeroing FFT bins outside the band
def fft_filter(x, fs, f_lo, f_hi):
    X = np.fft.fft(x)
    mask = band_mask(len(x), fs, f_lo, f_hi)
    X_filtered = X * mask
    x_filtered = np.fft.ifft(X_filtered)
    return np.real(x_filtered)    # imag part is only rounding error

# at 5cm breath hits the mic and gives low thumps. In Task 1 there was nothing
# useful below 80 Hz (voice f0 is around 90-120 Hz) so we just cut it
def remove_pops(x, fs):
    X = fft_filter(x, fs, 80, fs/2)
    return X

def exciter(x, fs, hp, drive, harm_band, mix):
    side = x
    # highpass first so the loud vowels don't go into tanh and mix into mud
    side = fft_filter(side, fs, hp, fs / 2)
    # tanh bends the waveform -> new harmonics. Higher drive = more bending
    side = np.tanh(drive * side)
    # tanh also makes stuff we don't want (low intermod, hiss up high), keep only harm_band
    side = fft_filter(side, fs, *harm_band)
    y = x + mix * side
    # no normalising here, otherwise y - x in the plots would include the whole voice
    return y

# fixed colour per version so the same version looks the same in every plot
COLORS = {'original': 'C0', 'v1': 'C1', 'v2': 'C7', 'v3': 'C3', 'v4': 'C4', 'v5': 'C2'}

# full spectrum of original and the chosen versions
def plot_spectra(x, out, fs):
    plt.figure(figsize=(10, 6))
    for name in ('original',) + SHOW:
        plt.semilogx(*spectrum_dB(out.get(name, x), fs), color=COLORS[name], alpha=0.6, label=name)
    plt.axvline(6000, color='k', ls='--', label='6 kHz (ear most sensitive)')
    plt.axvspan(*VERSIONS['v1']['harm_band'], color=COLORS['v1'], alpha=0.1, label='v1 harmonic band')
    plt.axvspan(*VERSIONS[FINAL]['harm_band'], color='grey', alpha=0.2, label='v3/v5 harmonic band')
    plt.annotate('identical below 1 kHz:\nvowels untouched', xy=(300, -60), xytext=(300, -120),
                 arrowprops=dict(arrowstyle='->'))
    plt.annotate('v1 lifts whole top end\n(fizz)', xy=(15000, -85), xytext=(1200, -145),
                 arrowprops=dict(arrowstyle='->'))
    plt.xlim(20, fs / 2); plt.ylim(-160, -30)
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Magnitude [dB]')
    plt.title('Spectrum: original vs exciter versions'); plt.legend(fontsize=8, loc='lower left')
    plt.savefig('task3_spectra.pdf')

# spectrum of only what the exciter added (output minus input)
def plot_added(x, out, fs):
    plt.figure(figsize=(10, 6))
    for name in SHOW:
        plt.semilogx(*spectrum_dB(out[name] - x, fs), color=COLORS[name], alpha=0.7, label=f'{name}: added by exciter')
    plt.axvline(6000, color='k', ls='--', label='6 kHz')
    plt.annotate('v1: broadband fizz 1-22 kHz', xy=(1500, -55), xytext=(100, -45),
                 arrowprops=dict(arrowstyle='->'))
    plt.annotate('v3/v5: only 4-10 kHz around 6 kHz', xy=(7000, -95), xytext=(200, -130),
                 arrowprops=dict(arrowstyle='->'))
    plt.xlim(20, fs / 2); plt.ylim(-160, -30)
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Magnitude [dB]')
    plt.title('What each exciter version adds (output - input)'); plt.legend(fontsize=8, loc='lower left')
    plt.savefig('task3_added.pdf')

# shows how much tanh actually distorts. tanh(d*s)/d would equal s if tanh was
# linear, so the difference is only the new harmonics. Used to compare drive 5 vs 20
def plot_harmonics(x, fs, names):
    plt.figure(figsize=(10, 6))
    for name in names:
        p = VERSIONS[name]
        s = fft_filter(x, fs, p['hp'], fs / 2)
        new = np.tanh(p['drive'] * s) / p['drive'] - s
        plt.semilogx(*spectrum_dB(new, fs), color=COLORS[name], alpha=0.7,
                     label=f"{name}: drive {p['drive']}")
    plt.axvspan(*VERSIONS[FINAL]['harm_band'], color='grey', alpha=0.2, label='harmonic band kept')
    plt.annotate('drive 5: tanh almost linear,\nfew new harmonics', xy=(5000, -125), xytext=(100, -150),
                 arrowprops=dict(arrowstyle='->'))
    plt.annotate('drive 20: real harmonics\n(incl. below highpass)', xy=(1000, -95), xytext=(60, -60),
                 arrowprops=dict(arrowstyle='->'))
    plt.xlim(20, fs / 2); plt.ylim(-160, -30)
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Magnitude [dB]')
    plt.title('Harmonics created by the tanh (non-linear part only)'); plt.legend(fontsize=8, loc='lower right')
    plt.savefig('task3_harmonics.pdf')

# original drawn on top so the coloured bits are where the exciter added something
def plot_time(x, y, fs, name):
    t = np.arange(len(x)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(t, y, color=COLORS[name], label=name)
    plt.plot(t, x, color=COLORS['original'], label='original')
    for t0 in (0.17, 0.92):
        plt.annotate('consonant boosted', xy=(t0, 0.25), xytext=(t0 - 0.1, 0.75),
                     arrowprops=dict(arrowstyle='->'))
    plt.xlabel('Time [s]'); plt.ylabel('Normalised amplitude')
    plt.title(f'Time domain: original vs {name}'); plt.legend()
    plt.savefig(f'task3_time_{name}.pdf')

fs, x = load(FILE)
x = remove_pops(x, fs)

# original at the same peak level as the outputs, so A/B listening is fair
wavfile.write('original_norm_5cm.wav', fs, (x / np.max(np.abs(x)) * 32767).astype(np.int16))
out = {name: exciter(x, fs, **p) for name, p in VERSIONS.items()}
for name, y in out.items():
    wavfile.write(f'exciter_{name}_5cm.wav', fs, (y / np.max(np.abs(y)) * 32767).astype(np.int16))
    # just the added part on its own, easier to hear the difference between versions
    wavfile.write(f'added_{name}_5cm.wav', fs, ((y - x) / np.max(np.abs(y - x)) * 32767).astype(np.int16))

plot_spectra(x, out, fs)
plot_added(x, out, fs)
plot_harmonics(x, fs, ('v3', 'v5'))
plot_time(x, out[FINAL], fs, FINAL)
plt.show()
