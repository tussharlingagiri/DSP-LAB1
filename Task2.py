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
    x = x.astype(np.float64)
    return fs, x / 2147483647.0

def plot_time(x, fs, key, title):
    plt.figure()
    plt.plot(np.arange(len(x)) / fs, x)
    plt.xlabel('Time [s]'); plt.ylabel('Normalised amplitude')
    plt.title(f' {title} ({key})')

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

def remove_hvac_noise(x, fs, N, N_initial_noise, frame_size=2048, alpha=2.0, beta=0.02):
    hop_size = frame_size // 2  # 50% overlap for smooth blending
    n = np.arange(frame_size)
    window = window = 0.5 - 0.5 * np.cos(2 * np.pi * n / (frame_size - 1)) # Hanning filtering window
    
    # 1. Capture the noise profile from a known quiet block at the start
    noise_block = x[N_initial_noise:N] * window
    X_noise = np.fft.fft(noise_block)
    noise_mag = np.abs(X_noise)
    
    output_signal = np.zeros(N)
    
    # 2. Loop through the entire audio file frame-by-frame
    for i in range(0, N - frame_size, hop_size):
        # Extract the current frame from the full audio and apply window
        frame = x[i:i + frame_size] * window
        
        # Convert frame to frequency domain
        X = np.fft.fft(frame)
        mag = np.abs(X)
        
        # Isolate the phase angles so we don't distort timing/speech quality
        phase = X / (mag + 1e-12)
        # phase = X / (mag)
        
        # 3. Subtract the noise profile magnitude from this frame's magnitude
        cleaned_mag = mag - (alpha * noise_mag)
        # Apply a spectral floor (beta) so bins aren't forced to absolute zero (prevents harsh artifacts)
        cleaned_mag = np.maximum(cleaned_mag, beta * mag)
        # 4. Reconstruct the frame using the cleaned magnitude and original phase
        X_clean = cleaned_mag * phase
        # Inverse FFT back to time domain, window again, and add to output
        frame_back = np.fft.ifft(X_clean).real * window
        output_signal[i:i + frame_size] += frame_back
        
    return output_signal

def filter_hp(initial_frequency,final_frequency,x,fs,N, key):
    X = np.fft.fft(x) #fft of audio signal
    f_res = fs / N
    
    k_final = int(final_frequency / f_res)
    X[0:k_final] = 1e-12 # temp amplification of audio signal
    X[-k_final:] = 1e-12
    F_smoothing = np.arange(0,1,0.005)
    X[k_final:k_final+200] *= F_smoothing
    X[-(k_final+200):-k_final] *= F_smoothing[::-1] 
    # X[0:k_final] = 0 # temp amplification of audio signal
    # X[-k_final:] = 0


    X_filtered = np.fft.ifft(X).real # inverse fft of adjusted signal
    return X_filtered # outputting a complex output array

def filter_lp(initial_frequency,final_frequency,x,fs, N, key):
    X = np.fft.fft(x) #fft of audio signal
    f_res = fs / N
    k_initial = int(initial_frequency / f_res)
    k_final = int(final_frequency / f_res)
    X[k_initial:k_final] = 1e-12 # temp amplification of audio signal
    X[-k_final:-k_initial] = 1e-12
    
    F_smoothing = np.arange(0,1,0.005)
    X[k_initial-200:k_initial] *= F_smoothing[::-1]
    X[-k_initial:-(k_initial-200)] *= F_smoothing
    X_filtered = np.fft.ifft(X).real # inverse fft of adjusted signal
    return X_filtered # outputting a complex output array

def boost_amplitude(initial_frequency, final_frequency, x, fs,N, modifier):
    X = np.fft.fft(x) #fft of audio signal
    f_res = fs / N
    k_initial = int(initial_frequency / f_res)
    k_final = int(final_frequency / f_res)
    X[k_initial:k_final] *= modifier # Amplification of chosen Fourier Coefficients
    X[-k_final:-k_initial] *= modifier

    Fc_smoothing = np.linspace(1,modifier,200) # Fourier Coefficient smoothing from normal power (1) towards the amplifying value
    if k_initial-200 < 0: #If we want to boost lower than 200hz, start from 0
        X[0:k_initial] *= Fc_smoothing[k_initial-1::-1]
        X[-k_initial:] *= Fc_smoothing[:k_initial]
    else:
        X[k_initial-200:k_initial] *= Fc_smoothing[::-1]
        X[-k_initial:-(k_initial-200)] *= Fc_smoothing

    X[k_final:k_final+200] *= Fc_smoothing # No If statement required as higher frequencies will be cut 
    X[-(k_final+200):-k_final] *= Fc_smoothing[::-1]

    X_filtered = np.fft.ifft(X).real # inverse fft of adjusted signal
    return X_filtered # outputting a complex output array

data = {k: load(c['file']) for k, c in REC.items()}

for key, c in REC.items():

    fs, x = data[key]
    N = len(x) # Total samples

    if key == '5cm':
        N_initial_noise = 54047 # Initial noise bin for 5cm
    else: N_initial_noise = 68467 # Initial noise bin for 1m

    noise_N = N - N_initial_noise # Sample length of Noise
    f_res = 44100/N # Frequency Resolution Calculation
    f_res_noise = 44100/(N-N_initial_noise) # Frequency Resoluation of Noise Block
    X_noise = x[N_initial_noise:N]
    noise_frame = N - N_initial_noise

    plot_time(x, fs, key, "Unfiltered Audio Time Domain")

    Xn_fft = np.fft.fft(X_noise)[:noise_N // 2] / noise_N

    plt.figure()
    f = np.arange(noise_N//2) * f_res_noise
    noise_mag = 20 * np.log10(np.abs(Xn_fft[1:]) + 1e-12)
    # noise_mag = 20 * np.log10(np.abs(Xn_fft[1:]))
    plt.plot(f[1:],noise_mag)

    x = x*2

    X_filtered = remove_hvac_noise(x, fs, N, N_initial_noise, noise_frame, alpha=5.0, beta=0.01)

    plot_time(X_filtered,fs,key, "Filtered Audio Time Domain")

    X_filtered = filter_hp(0,70, X_filtered, fs, N, key)

    X_filtered = filter_lp(6000,22050, X_filtered, fs, N, key)

    plot_time(X_filtered,fs,key, "Bandpass filtered in the Time Domain")

    X_filtered = boost_amplitude(80,2000,X_filtered,fs,N, 10.0) 

    plt.figure()   
    X_2 = np.fft.fft(X_filtered)[:N // 2] / N
    f = np.arange(N // 2) * f_res
    X_mag = 20 * np.log10(np.abs(X_2[1:]) + 1e-12)
    # X_mag = 20 * np.log10(np.abs(X_2[1:]))
    plt.plot(f[1:],X_mag)

    # X_filtered = np.tanh(X_filtered) # adds a weird distortion

    # X_filtered = boost_amplitude(2000,3000,X_filtered,fs,N, 30.0)

    # plt.figure()
    # X_2 = np.fft.fft(X_filtered)[:N // 2] / N
    # f = np.arange(N // 2) * f_res
    # X_mag = 20 * np.log10(np.abs(X_2[1:]) + 1e-12)
    # plt.plot(f[1:],X_mag)

    audio_int16 = (X_filtered * 32767).astype(np.int16)
    if key == '5cm' :
        wavfile.write("filtered_audio_test_5cm.wav", 44100, audio_int16)
    else: wavfile.write("filtered_audio_test_1m.wav", 44100, audio_int16)

plt.show()