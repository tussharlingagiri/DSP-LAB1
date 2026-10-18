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

def clear_noise(x, fs, N, N_initial_noise, frame_size, alpha, beta):
    hop_size = frame_size // 2  # 50% overlap into the other frames so that it blends well
    n = np.arange(frame_size)
    window = window = 0.5 - 0.5 * np.cos(2 * np.pi * n / (frame_size - 1)) # Hanning filtering window, smoothens the edges of the frames
    noise_block = x[N_initial_noise:N] * window #Holds found block of noise at the end of the audio
    X_noise = np.fft.fft(noise_block) # Finds the fourier coefficients with their respective frequencies of the background noise
    noise_mag = np.abs(X_noise) # Finds their respective magnitudes
    X_filtered = np.zeros(N) # Stores filtered values

    for i in range(0, N - frame_size, hop_size):
        current_frame = x[i:i + frame_size] * window # Applies hanning window to current frame, middle of the window is peak amplitude, edges are lower but accumulated
        X = np.fft.fft(current_frame) # Converts the frame to the frequency domain
        X_mag = np.abs(X)
        X_phase = X / (X_mag + 1e-12) #Preserve phase angles, adds a small number to avoid 1/0
        
        cleaned_mag = X_mag - (alpha * noise_mag) # 3. Subtract the noise profile magnitude from this frame's magnitude 
        cleaned_mag = np.maximum(cleaned_mag, beta * X_mag) # If cleaned magnitude < 0, replace so that it is just above 0 since magnitudes cannot be negative. 
        X_clean = cleaned_mag * X_phase # Reconstructs X but after noise subtraction
        frame_back = np.fft.ifft(X_clean).real * window 
        X_filtered[i:i + frame_size] += frame_back # This is where the frames overlap and the edges accumulate smoothly
        
    return X_filtered 

def filter_hp(initial_frequency,final_frequency,x,fs,N, key):
    X = np.fft.fft(x) #fft of audio signal
    f_res = fs / N
    
    k_final = int(final_frequency / f_res)
    X[0:k_final] = 1e-12 # temp amplification of audio signal
    X[-k_final:] = 1e-12
    F_smoothing = np.arange(0,1,0.005)
    X[k_final:k_final+200] *= F_smoothing
    X[-(k_final+200):-k_final] *= F_smoothing[::-1] 
        
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

    X[k_final:k_final+200] *= Fc_smoothing[::-1] # No If statement required as higher frequencies will be cut 
    X[-(k_final+200):-k_final] *= Fc_smoothing

    X_filtered = np.fft.ifft(X).real # inverse fft of adjusted signal
    return X_filtered # outputting a complex output array


def add_bass_saturation(x, drive=2.0):
  """Applies soft-clipping saturation to generate harmonics,

  making weak bass feel thicker and more present.
  """
  # 'drive' amplifies the signal before clipping, pushing it into the non-linear zone
  driven_signal = x * drive

  # np.tanh (hyperbolic tangent) is the classic formula for warm, "analog-style" soft clipping
  saturated_signal = np.tanh(driven_signal)

  # Scale it back down slightly so it doesn't clip your overall output volume
  output = saturated_signal / np.tanh(drive)

  return output

data = {k: load(c['file']) for k, c in REC.items()}

for key, c in REC.items():

    fs, x = data[key]
    N = len(x) # Total samples

    if key == '5cm':
        N_initial_noise = 54047 # Initial noise bin for 5cm
    else: N_initial_noise = 68467 # Initial noise bin for 1m

    noise_N = N - N_initial_noise # Sample length of Noise
    f_res = 44100/N # Frequency Resolution Calculation
    f_res_noise = 44100/(N-N_initial_noise) # Frequency Resolution of Noise Block
    X_noise = x[N_initial_noise:N]
    noise_frame = N - N_initial_noise

    plot_time(x, fs, key, "Unfiltered Audio Time Domain")

    Xn_fft = np.fft.fft(X_noise)[:noise_N // 2] / noise_N

    plt.figure()
    f = np.arange(noise_N//2) * f_res_noise
    noise_mag = 20 * np.log10(np.abs(Xn_fft[1:]) + 1e-12)
    plt.plot(f[1:],noise_mag)

    X_filtered = clear_noise(x, fs, N, N_initial_noise, noise_frame, 2 , 0.1)

    # plot_time(X_filtered,fs,key, "Filtered Audio Time Domain")

    # X_filtered = filter_lp(8000,22050, X_filtered, fs, N, key)

    # plot_time(X_filtered,fs,key, "Bandpass filtered in the Time Domain")

    X_filtered = boost_amplitude(100, 500,X_filtered,fs,N, 3.0) 

    X_filtered = boost_amplitude(500, 2000,X_filtered,fs,N, 2) # starting from 900 so the full band from 125 to 2000 is boosted either directly or by smoothing
    X_filtered = boost_amplitude(2000,4000,X_filtered,fs,N, 0.3) # hissing and high frequency consonants are attenuated
    X_filtered = boost_amplitude(4000,8000,X_filtered,fs,N, 0.2) # hissing and high frequency consonants are attenuated

    if key == "1m": # Extra boost of bass if 1m away
        X_filtered = add_bass_saturation(X_filtered,3)
        X_filtered = boost_amplitude(50, 300,X_filtered,fs,N, 5.0) 

    audio_int16 = (X_filtered * 32767).astype(np.int16)
    wavfile.write(f"enhanced_speech_{key}.wav", 44100, audio_int16)

plt.show()