
import sys
import scipy.signal.windows as sig
sys.path.append('../')
from tds_utils_22 import *
import scipy.io.wavfile as wf
import librosa

filename = 'vocals/a_g.wav'

fs, y = wf.read(filename)

#obtain frame s2
s2 = y[0:240]
#set LPC order
p = 12

N = int(fs*0.03) #length in samples
h = sig.hamming(N) #hamming window

x = s2 * h

lpc = librosa.lpc(x,order= p)
print(lpc)

