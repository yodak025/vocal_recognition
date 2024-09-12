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
import spectrum
import librosa 

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
     
        
def predlin(s,p,w):
    """
    Function that computes de linear prediction using lpc for a given signal
    """
    x = s*w
    
    lpc = librosa.lpc(x,order= p)
    ar,sigma,lt = spectrum.aryule(x,p)
    
    
    # plt.subplot(3,1,1)
    # plt.plot(s)
    # plt.title("Speech signal")
    # xx = np.arange(0,len(w))
    # plt.plot(xx,w*np.max(s))
    #
    # plt.subplot(312)
    # plt.plot(x)
    # plt.title("Windowed signal")
    #
    # plt.subplot(313)
    # S_x = 20*np.log10(np.abs(np.fft.fft(x,2*len(w))))
    # plt.plot(S_x[0:len(w)])
    # ff,h = scipy.signal.freqz(1,ar,len(w),whole=False)
    # h2 = spectrum.arma2psd(ar,NFFT=len(w)*2)
    # plt.plot(20*np.log10(np.abs(h2[0:len(w)])))
    return lpc      # Lo ha añadido David
   

#Complex cepstrum from python-acoustics
#https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/cepstrum.py
def complex_cepstrum(x, n=None):
    """Compute the complex cepstrum of a real sequence.
    Parameters
    ----------
    x : ndarray
        Real sequence to compute complex cepstrum of.
    n : {None, int}, optional
        Length of the Fourier transform.
    Returns
    -------
    ceps : ndarray
        The complex cepstrum of the real data sequence `x` computed using the
        Fourier transform.
    ndelay : int
        The amount of samples of circular delay added to `x`.
    The complex cepstrum is given by
    .. math:: c[n] = F^{-1}\\left{\\log_{10}{\\left(F{x[n]}\\right)}\\right}
    where :math:`x_[n]` is the input signal and :math:`F` and :math:`F_{-1}
    are respectively the forward and backward Fourier transform.
    See Also
    --------
    real_cepstrum: Compute the real cepstrum.
    inverse_complex_cepstrum: Compute the inverse complex cepstrum of a real sequence.
    Examples
    --------
    In the following example we use the cepstrum to determine the fundamental
    frequency of a set of harmonics. There is a distinct peak at the quefrency
    corresponding to the fundamental frequency. To be more precise, the peak
    corresponds to the spacing between the harmonics.
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import complex_cepstrum
    >>> duration = 5.0
    >>> fs = 8000.0
    >>> samples = int(fs*duration)
    >>> t = np.arange(samples) / fs
    >>> fundamental = 100.0
    >>> harmonics = np.arange(1, 30) * fundamental
    >>> signal = np.sin(2.0*np.pi*harmonics[:,None]*t).sum(axis=0)
    >>> ceps, _ = complex_cepstrum(signal)
    >>> fig = plt.figure()
    >>> ax0 = fig.add_subplot(211)
    >>> ax0.plot(t, signal)
    >>> ax0.set_xlabel('time in seconds')
    >>> ax0.set_xlim(0.0, 0.05)
    >>> ax1 = fig.add_subplot(212)
    >>> ax1.plot(t, ceps)
    >>> ax1.set_xlabel('quefrency in seconds')
    >>> ax1.set_xlim(0.005, 0.015)
    >>> ax1.set_ylim(-5., +10.)
    References
    ----------
    .. [1] Wikipedia, "Cepstrum".
           http://en.wikipedia.org/wiki/Cepstrum
    .. [2] M.P. Norton and D.G. Karczub, D.G.,
           "Fundamentals of Noise and Vibration Analysis for Engineers", 2003.
    .. [3] B. P. Bogert, M. J. R. Healy, and J. W. Tukey:
           "The Quefrency Analysis of Time Series for Echoes: Cepstrum, Pseudo
           Autocovariance, Cross-Cepstrum and Saphe Cracking".
           Proceedings of the Symposium on Time Series Analysis
           Chapter 15, 209-243. New York: Wiley, 1963.
    """
    def _unwrap(phase):
        samples = phase.shape[-1]
        unwrapped = np.unwrap(phase)
        center = (samples+1)//2
        if samples == 1:
            center = 0
        ndelay = np.array(np.round(unwrapped[...,center]/np.pi))
        unwrapped -= np.pi * ndelay[...,None] * np.arange(samples) / center
        return unwrapped, ndelay

    spectrum = np.fft.fft(x, n=n)
    unwrapped_phase, ndelay = _unwrap(np.angle(spectrum))
    log_spectrum = np.log(np.abs(spectrum)) + 1j*unwrapped_phase
    ceps = np.fft.ifft(log_spectrum).real

    return ceps, ndelay


        

