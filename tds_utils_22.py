#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:48:01 2018

Utils: functions to support jupyter notebooks for TDS

@author: Óscar Barquero Pérez y Rebeca Goya Esteban
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal


def xcorr(x, y, normed=False,det=False):
    """
    Taken from Axes.xcorr
        
    Plot the cross correlation between *x* and *y*.

       Call signature::

    xcorr(self, x, y, normed=True, detrend=mlab.detrend_none,
              usevlines=True, maxlags=10, **kwargs)

    If *normed* = *True*, normalize the data by the cross
        correlation at 0-th lag.  *x* and y are detrended by the
        *detrend* callable (default no normalization).  *x* and *y*
        must be equal length.

        Data are plotted as ``plot(lags, c, **kwargs)``

        Return value is a tuple (*lags*, *c*, *line*) where:

          - *lags* are a length ``2*maxlags+1`` lag vector

          - *c* is the ``2*maxlags+1`` auto correlation vector

          - *line* is a :class:`~matplotlib.lines.Line2D` instance
             returned by :func:`~matplotlib.pyplot.plot`.

        The default *linestyle* is *None* and the default *marker* is
        'o', though these can be overridden with keyword args.  The
        cross correlation is performed with :func:`numpy.correlate`
        with *mode* = 2.
Ƒ
        If *usevlines* is *True*:

           :func:`~matplotlib.pyplot.vlines`
           rather than :func:`~matplotlib.pyplot.plot` is used to draw
           vertical lines from the origin to the xcorr.  Otherwise the
           plotstyle is determined by the kwargs, which are
           :class:`~matplotlib.lines.Line2D` properties.

           The return value is a tuple (*lags*, *c*, *linecol*, *b*)
           where *linecol* is the
           :class:`matplotlib.collections.LineCollection` instance and
           *b* is the *x*-axis.

        *maxlags* is a positive integer detailing the number of lags to show.
        The default value of *None* will return all ``(2*len(x)-1)`` lags.

        **Example:**

        :func:`~matplotlib.pyplot.xcorr` is top graph, and
        :func:`~matplotlib.pyplot.acorr` is bottom graph.

        .. plot:: mpl_examples/pylab_examples/xcorr_demo.py
        """

    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')
        
    if det == True:
        
        x = scipy.signal.detrend(np.asarray(x))
        y = scipy.signal.detrend(np.asarray(y))

    c = np.correlate(x, y, mode=2)

    if normed:
        c /= np.sqrt(np.dot(x, x) * np.dot(y, y))

    maxlags = Nx - 1
    
    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags:Nx + maxlags]
    return lags, c

def NextPowerOfTwo(number):
    # Returns next power of two following 'number'
    return np.ceil(np.log2(number))

#TO_DO: redefine this function to be periodogram, allowing for a differente windows
def my_spectra(x,fs):
    """
    Function that computes the PSD from a given signal using hamming window a NFFT
    the next smaller power of 2 the length of x
    """
    
    x = x.flatten()
    w = np.hamming(len(x)) #hamming window
    
    #TO_DO why 2 times nextpower of two?
    NFFT = int(2*(2**NextPowerOfTwo(len(x))))
    
    x_w = w*x
    x#_w = x[:]
    
   # fft_x = np.fft.fft(x_w,NFFT)/ NFFT
    fft_x = np.fft.fft(x_w,NFFT)/len(x) #TO_DO tengo alguna duda con L o NFFT
    
    f = np.fft.fftfreq(len(fft_x),d=1/fs)
    
    pds = np.fft.fftshift(2*np.abs(fft_x)) #TO_DO multipliying by two in order to get all
    #the power in positive frequencies.
    
    f = np.fft.fftshift(f)
    
    return pds, f

def espectro_ventanas(r,h):
    """
    Function that compute and plot psd from the rectangular and hamming windows
    passed as parameters
    
    TO_DO: quiza es mejor que esto solo devuelva las psd (en db) y f. Y que los
    alumnos pinten en el notebook
    """
    
    #get NFFT points
    NFFT = int(8*2**NextPowerOfTwo(len(r)))
    Rect_Frec = np.fft.fft(r,NFFT);
    Hamm_Frec = np.fft.fft(h,NFFT);

    f = np.fft.fftfreq(len(Rect_Frec))
    f = np.fft.fftshift(f)
    
    #psd in db
    r_psd = 20*np.log10(np.abs(np.fft.fftshift(Rect_Frec)))
    h_psd = 20*np.log10(np.abs(np.fft.fftshift(Hamm_Frec)))
    
    #Get only positivie frequencies
    idx = f >= 0
    
    #TO_DO plot only from 0 to 0.1 Hz ?
    idx = np.logical_and(f >= 0,f <=0.1)
    
    f = f[idx]
    r_psd = r_psd[idx]
    h_psd = h_psd[idx]
    
    #normalize
    r_psd = r_psd/np.max(np.abs(r_psd))
    h_psd = h_psd/np.max(np.abs(h_psd))
    
    return r_psd, h_psd, f    

def energia(s,w):
    """
    Function that computes localized energy from a signal s, given a window w.
    w should be os length smaller than s
    """
    
    Energy = []
    #for over the signal in steps of len(w)
    #for n in range(0,len(s)-len(w),len(w)):
    for n in range(0,len(s)-len(w)):
       
        #print(n,':',n+len(w))
        #print(len(s))
        trama = s[n:n+len(w)] * w #actual windowed segment
        
        Energy.append(np.sum(trama**2))
        
    return np.array(Energy)
    

def zcr(s,w):
    """
    Function that computes l zero-crossing rate from a signal, given a window w.
    w should be os length smaller than sig
    """
    zcr_a = []
    
    for n in range(0,len(s)-len(w),len(w)):
        trama = s[n:n+len(w)]
        
        zcr_a.append(np.sum((0.5/len(trama))*(np.abs(np.sign(trama[1:])-np.sign(trama[:-1])))))
        
    return zcr_a

def my_spectrogram(x, N , fs, plot_flag = True):
    """
    Function that computes and plot the spectrogram of x, given a lenght window N (samples), and 
    a sampling frequency fs (Hz).
    """
    
    f, t, Sxx = scipy.signal.spectrogram(x,fs,window = 'hamming',nperseg = N,noverlap = 0,nfft = N)
    
    if plot_flag:
        
        fig, ax = plt.subplots(figsize = (8,6))
        pm = plt.pcolormesh(t, f, 10*np.log10(Sxx),cmap = 'jet')
        cbar = fig.colorbar(pm)
        cbar.ax.set_ylabel('PSD dB')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
     
