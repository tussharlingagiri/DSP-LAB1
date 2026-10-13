# Tasks:
# Enhance both voice recordings, make it clearer and like a radio show and remove noise.

# To do this, manipulate the amplitudes of the frequency coefficients.

# For both recordings, explain the kind of noise 
# that needs to be removed and what frequency range it has.

# Explain how you would like to improve the speech by looking into loss 
# of bass frequencies with distance and the risk of popping sounds when close.

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

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

def load(path):
    fs, x = wavfile.read(path)
    print(x.dtype)
    x = x.astype(np.float64)
    # return fs, x / np.max(np.abs(x))
    return fs, x / 2147483647.0


def spectrum_dB(x, fs):
    N = len(x)
    X = np.fft.fft(x)[:N // 2] / N            
    f = np.arange(N // 2) * fs / N
    # for i in range(len(X)):
        # print(X[i])
        # X[i+initial_bin] = 0 # actual filtering part
    return f[1:], 20 * np.log10(np.abs(X[1:]) + 1e-12)

def filter_bandwidth(initial_frequency,final_frequency,x,fs):
    N = len(x) #Number of samples
    print("N:", N)
    X = np.fft.fft(x) #fft of audio signal
    print("X:", len(X))
    f_res = fs / N
    
    k_initial = int(initial_frequency / f_res)
    k_final = int(final_frequency / f_res)
    X[k_initial:k_final] = 0 # temp amplification of audio signal
    
    # X_2 = np.fft.fft(x)[:N // 2] / N
    # X_2[k_initial:k_final] = 0 # temp amplification of audio signal
    # f = np.arange(N // 2) * f_res
    # X_mag = 20 * np.log10(np.abs(X_2[1:]) + 1e-12)
    # plot_annotated(x, fs, key, c, f[1:], X_mag)

    X_filtered = np.fft.ifft(X).real # inverse fft of adjusted signal
    print("filtered X:", len(X_filtered))
    return X_filtered # outputting a complex output array

def filter_magnitude(x, fs, modifier):
    N = len(x) #Number of samples
    print("N:", N)
    X = np.fft.fft(x) #fft of audio signal
    PSD = X * np.conj(X)
    PSD_norm = PSD / np.max(PSD)
    frequency = (fs/N) * np.arange(N)
    L = np.arange(1,np.floor(N/2),dtype='int')
    print("X:", len(X))

    plt.plot(frequency[L], PSD_norm[L])
    # f_res = fs / N

    X_filter = PSD_norm > modifier
    PSD_norm[int(-len(PSD_norm)/2):] = PSD_norm[:int(len(PSD_norm)/2)][::-1]

    print(len(X_filter))
    print(len(X))
    X = X * X_filter

    PSD_norm = X * np.conj(X) / np.max(PSD)

    plt.plot(frequency[L], PSD_norm[L])
    
    # k_initial = int(initial_frequency / f_res)
    # k_final = int(final_frequency / f_res)
    # X[k_initial:k_final] = 0 # temp amplification of audio signal
    
    # X_2 = np.fft.fft(x)[:N // 2] / N
    # X_2[k_initial:k_final] = 0 # temp amplification of audio signal
    # f = np.arange(N // 2) * f_res
    # X_mag = 20 * np.log10(np.abs(X_2[1:]) + 1e-12)
    # plot_annotated(x, fs, key, c, f[1:], X_mag)
    # plt.plot(frequency[L], PSD[L])
    X_filtered = np.fft.ifft(X).real # inverse fft of adjusted signal
    print("filtered X:", len(X_filtered))
    return X_filtered # outputting a complex output array

def boost_amplitude(initial_frequency, final_frequency, x, fs, modifier):
    N = len(x) #Number of samples
    print("N:", N)
    X = np.fft.fft(x) #fft of audio signal
    print("X:", len(X))
    f_res = fs / N
    
    k_initial = int(initial_frequency / f_res)
    k_final = int(final_frequency / f_res)
    X[k_initial:k_final] *= modifier # temp amplification of audio signal
    
    X_filtered = np.fft.ifft(X).real # inverse fft of adjusted signal
    print("filtered X:", len(X_filtered))
    return X_filtered # outputting a complex output array
    
    
def plot_time(x, fs, key):
    plt.figure()
    plt.plot(np.arange(len(x)) / fs, x)
    plt.xlabel('Time [s]'); plt.ylabel('Normalised amplitude')
    plt.title(f'Time domain ({key})')

def plot_annotated(x, fs, key, c, f, m):
    # f, m = spectrum_dB(x, fs)
    # f, m = filter_bandwidth(2000,22050,x, fs)
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

def plot_boosted(x, fs, key, c):
    # f, m = spectrum_dB(x, fs)
    # f, m = filter_bandwidth(2000,22000,x, fs)#
    print("length of fs", fs)
    f, m = boost_amplitude(2000,22050,x, fs)
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
    plt.title(f'boosted spectrum  ({key})')
    plt.legend(loc='lower left', fontsize=8)

data = {k: load(c['file']) for k, c in REC.items()}

for key, c in REC.items():
    fs, x = data[key]
    # plot_time(x, fs, key)
    X = filter_magnitude(x,fs)
    X = filter_bandwidth(8000,22050,X,fs,key)
    X = filter_bandwidth(0,250,X,fs,key)
    X = boost_amplitude(300,800,X,fs)
    X = boost_amplitude(5000,8000,X,fs)

    # plot_time(X, fs, key)         
    # print("fs:",fs)
    # print("x:",x)
    # print(f"{key}: fs={fs}, duration={len(x) / fs:.2f}s, "
    #       f"samples at max={np.sum(np.abs(x) >= 0.999)}")
    # f , m = filter_bandwidth(2000,22050,x, fs)
    audio_int16 = (X * 32767).astype(np.int16)
    wavfile.write("filtered_audio_test.wav", 44100, audio_int16)
    # plot_time(x, fs, key)                                  
    # plot_annotated(x, fs, key, c, f, m)
    # plot_boosted(x, fs, key, c)
    # filter_bandwidth(1,3000,x,fs,key)
    break
plt.show()