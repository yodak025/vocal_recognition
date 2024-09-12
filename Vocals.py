import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from IPython.display import Audio
from tds_utils_22 import *

import scipy.io.wavfile as wav
import sys
sys.path.append('../')      # allows to import a module in a diff folder creo que no hace falta

import scipy.signal as sig
import playsound

filename = 'vocals/A_Bruno_nr.wav'
fs, y = wav.read(filename)

# playsound.playsound(filename)

print(y)
print(y.shape)
print(fs)

Audio(y, rate=fs)

# plot wav signal

# time vector
t = np.arange(0,len(y)/fs, 1/fs)

plt.figure(figsize=(7,5))
plt.plot(t, y)

# plot figure


plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')
plt.title('Vocal')
plt.show()


N = int(0.02*fs) #length in samples

r = sig.boxcar(N)     #rectangular window
h = sig.hamming(N)      #hamming window

t = np.arange(0, len(r))/fs #time in sec

plt.figure(figsize=(7, 5))

#plot rectangular window

plt.plot(t, r)
plt.xlabel('Time [sec]')
plt.title('Ventana Rectangular')
plt.show()


plt.figure(figsize=(7, 5))

#plot hamming window

plt.plot(t, h)
plt.xlabel('Time [sec]')
plt.title('Ventana Hamming')
plt.show()

r_psd_20, h_psd_20, f_20 = espectro_ventanas(r, h)

#plot
plt.figure(figsize=(8,6))

plt.subplot(211)
#plot rectangular window spectrum
plt.plot(f_20, r_psd_20)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Rect window PSD (dB)')

plt.subplot(212)
#plot hamming window spectrum

plt.plot(f_20, h_psd_20)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Rect window PSD (dB)')
plt.show()

