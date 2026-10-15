import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Task 1 - time and frequency plots of both recordings, with vowels,
# consonants and noise marked by hand.
# All the numbers below were read off the plots by eye, nothing is detected automatically.
#   vowels / cons / noise: time segments (s) of a vowel, a consonant and silence
#   f0: vowel fundamentals picked from the spectrum (Hz)
#   cons_range: band where the consonants show up above the noise (Hz)
#   noise_low / noise_high: below / above these it's only noise (Hz)
REC = {
    '5cm': dict(file='Assignment_1_audio_5cm.wav',
                vowels=[(0.43, 0.67), (1.00, 1.13)],
                cons=[(0.12, 0.22), (0.88, 0.96)],
                noise=(1.22, 1.50),
                f0=[120, 90],                  
                cons_range=(3000, 15000),
                noise_low=80, noise_high=18000),
    '1m':  dict(file='Assignment_1_audio_1m.wav',
                vowels=[(0.55, 0.90), (1.30, 1.50)],
                cons=[(0.95, 1.03), (1.20, 1.30)],
                noise=(0.00, 0.45),
                f0=[120, 95],                  
                cons_range=(2000, 12000),
                noise_low=80, noise_high=15000),
}

# read wav, take one channel if stereo, scale so the peak is 1
def load(path):
    fs, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float64)
    return fs, x / np.max(np.abs(x))

# one-sided spectrum in dB. fftfreq isn't allowed so f is built from k * fs / N.
# DC bin dropped so the log x axis works
def spectrum_dB(x, fs):
    N = len(x)
    X = np.fft.fft(x)[:N // 2] / N            
    f = np.arange(N // 2) * fs / N           
    return f[1:], 20 * np.log10(np.abs(X[1:]) + 1e-12)

# cut out t0..t1 seconds
def seg(x, fs, t0, t1):
    return x[int(t0 * fs):int(t1 * fs)]

# ---- REQUIRED ---------------------------------------------------------------
def plot_time(x, fs, key):
    plt.figure()
    plt.plot(np.arange(len(x)) / fs, x)
    plt.xlabel('Time [s]'); plt.ylabel('Normalised amplitude')
    plt.title(f'Time domain ({key})')
    plt.savefig(f'task1_time_{key}.pdf')
  
# the main plot for the report: full spectrum with f0, consonant and noise regions marked
def plot_annotated(x, fs, key, c):
    f, m = spectrum_dB(x, fs)
    plt.figure(figsize=(10, 6))
    plt.semilogx(f, m)
    for i, f0 in enumerate(c['f0']):
        plt.axvline(f0, color='r', ls='--',
                    label=f"Vowel fundamentals ({', '.join(map(str, c['f0']))} Hz)" if i == 0 else None)
    plt.axvspan(*c['cons_range'], color='orange', alpha=0.2,
                label=f"Consonants ({c['cons_range'][0]}-{c['cons_range'][1]} Hz)")
    plt.axvspan(f[0], c['noise_low'], color='grey', alpha=0.3,
                label=f"Noise (<{c['noise_low']} Hz, >{c['noise_high']} Hz)")
    plt.axvspan(c['noise_high'], fs / 2, color='grey', alpha=0.3)
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Magnitude [dB]')
    plt.title(f'Annotated spectrum ({key})')
    plt.legend(loc='lower left', fontsize=8)
    plt.savefig(f'task1_spectrum_{key}.pdf')
 
# ---- SUPPORTING -------------------------------------------------------------
# short FFTs over time, used to find where the vowels / consonants / silence are.
# Hann window to reduce leakage between bins, 50% overlap
def plot_spectrogram(x, fs, key, win=1024, hop=512):
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)
    frames = np.array([x[i:i + win] * w for i in range(0, len(x) - win, hop)])
    S = 20 * np.log10(np.abs(np.fft.fft(frames, axis=1))[:, :win // 2] + 1e-12)
    plt.figure()
    plt.imshow(S.T, origin='lower', aspect='auto', extent=[0, len(x) / fs, 0, fs / 2])
    plt.colorbar(label='Intensity [dB]')
    plt.xlabel('Time [s]'); plt.ylabel('Frequency [Hz]')
    plt.title(f'Spectrogram, full range ({key})')
    plt.savefig(f'task1_spectrogram_{key}.pdf')
   
# spectrum of one vowel only, zoomed to 50-2000 Hz so the f0 and its harmonics
# are easier to see than in the full recording
def plot_vowel(x, fs, t0, t1, key):
    f, m = spectrum_dB(seg(x, fs, t0, t1), fs)
    plt.figure()
    plt.semilogx(f, m)
    plt.xlim(50, 2000)
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Magnitude [dB]')
    plt.title(f'Vowel {t0:.2f}-{t1:.2f} s ({key})')
    plt.savefig(f'task1_vowel_{key}_{t0:.2f}s.pdf')

# consonant vs silence on one plot. Where the two lines meet is where the
# consonants stop and only noise is left, that's how cons_range / noise_high were picked
def plot_overlay(x, fs, speech, noise, key):
    plt.figure()
    for name, (t0, t1) in (('Consonant', speech), ('Noise', noise)):
        f, m = spectrum_dB(seg(x, fs, t0, t1), fs)
        plt.semilogx(f, m, alpha=0.6, label=f'{name} {t0:.2f}-{t1:.2f} s')
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Magnitude [dB]')
    plt.title(f'Consonant vs noise ({key})'); plt.legend()
    plt.savefig(f'task1_cons_vs_noise_{key}.pdf')
  
# plots are also saved as pdf (vector) for the report
data = {k: load(c['file']) for k, c in REC.items()}

for key, c in REC.items():
    fs, x = data[key]
    # quick check: fs should be >= 44.1 kHz and no samples stuck at max (clipping)
    print(f"{key}: fs={fs}, duration={len(x) / fs:.2f}s, "
          f"samples at max={np.sum(np.abs(x) >= 0.999)}")
    plot_time(x, fs, key)                                  
    plot_annotated(x, fs, key, c)                           
    plot_spectrogram(x, fs, key)                            
    plot_vowel(x, fs, *c['vowels'][0], key)                
    plot_vowel(x, fs, *c['vowels'][1], key)                 
    plot_overlay(x, fs, c['cons'][0], c['noise'], key)     

plt.show()