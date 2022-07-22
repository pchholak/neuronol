import mne
import pickle
import numpy as np
import scipy as sp
import os.path as op
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams['figure.figsize'] = [8, 5]
plt.rcParams['font.size'] = 14

def featureFunc_centroidFrequencyBand(psds, freqs, band, channels=None, plot_psd=None):
    '''
    Calculate the centroid frequency in the power spectrum of the given frequency band. The power
    spectrum can be averaged over all channels as done by Saletu-Zyhlarz 2004 or a different
    selection of channels can be used.

    Parameters
    ----------
    psds : 2D array
        The power spectral densities of all channels [n_channels, n_freqs].
    freqs : 1D array
        Frequencies in Hertz.
    band : tuple
        The lowest and highest frequencies to consider the power spectral densities over.
    channels : list of int | None
        Channel indices for channels to include (None, default, includes all).
    plot_psd: bool | None
        Whether to plot the power spectrum along with the centroid frequency marked.
        None, default, does not.

    Returns
    -------
    cf : scalar
        The centroid frequency in the power spectrum of the given frequency band.
    '''
    # Average power spectrum over desired set of channels
    if channels is None:
        mean_psd = np.mean(psds, axis=0)
    elif len(channels) > 1:
        mean_psd = np.mean(psds[channels, :], axis=0)
    else:
        mean_psd = psds[channels, :].reshape((-1,))

    # Isolate power spectrum of the given frequency band
    inds_range = [(np.abs(freqs - band[0])).argmin(), (np.abs(freqs - band[1])).argmin()]
    inds = np.arange(inds_range[0], inds_range[1]+1)
    psd_band, f_band = mean_psd[inds], freqs[inds]

    # Calculate the centroid frequency
    cf = np.trapz(f_band * psd_band, f_band) / np.trapz(psd_band, f_band)

    # Calculate the standard deviation of the centroid
    sigma = np.sqrt(np.sum(np.square(f_band - cf) * psd_band))

    # Plot FFT
    if plot_psd and plot_psd is not None:
        plt.figure()
        plt.plot(freqs, mean_psd, color='k', lw=2)
        plt.plot(f_band, psd_band, color='g', lw=3)
        plt.axvline(x=cf, color='r', lw=2)
        plt.xlim([freqs.min(), freqs.max()])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power Spectral Density [μV2/Hz]')
        plt.tight_layout()
        plt.show()

    return cf, sigma

def featureFunc_dominantFrequency(psds, freqs, band, channels=None, plot_psd=None):
    '''
    Calculate the dominant frequency in the power spectrum in the given frequency band. The power
    spectrum can be averaged over all channels as done by Saletu-Zyhlarz 2004 or a different
    selection of channels can be used.

    Parameters
    ----------
    psds : 2D array
        The power spectral densities of all channels [n_channels, n_freqs].
    freqs : 1D array
        Frequencies in Hertz.
    band : tuple
        The lowest and highest frequencies to consider the power spectral densities over.
    channels : list of int | None
        Channel indices for channels to include (None, default, includes all).
    plot_psd: bool | None
        Whether to plot the power spectrum along with the dominant frequency marked.
        None, default, does not.

    Returns
    -------
    df : scalar
        The dominant frequency in the power spectrum in the given frequency band.
    '''
    # Average power spectrum over desired set of channels
    if channels is None:
        mean_psd = np.mean(psds, axis=0)
    elif len(channels) > 1:
        mean_psd = np.mean(psds[channels, :], axis=0)
    else:
        mean_psd = psds[channels, :].reshape((-1,))

    # Isolate power spectrum of the given frequency band
    inds_range = [(np.abs(freqs - band[0])).argmin(), (np.abs(freqs - band[1])).argmin()]
    inds = np.arange(inds_range[0], inds_range[1]+1)
    psd_band = mean_psd[inds]; f_band = freqs[inds]

    # Calculate dominant frequency
    i_df = psd_band.argmax()
    df = f_band[i_df]
    psd_df = psd_band[i_df]

    # Calculate absolute power of dominant frequency
    delta_freq = f_band[1] - f_band[0]
    ap_df = psd_df * delta_freq

    # Calculate relative power of dominant frequency
    tp = featureFunc_bandAbsolutePower(psds, freqs, band, channels=channels,
                                       norm_and_log_density=None, compute_magnitude=None,
                                       plot_psd=None)
    rp_df = ap_df / tp

    # Plot FFT
    if plot_psd and plot_psd is not None:
        plt.figure()
        plt.plot(freqs, mean_psd, color='k', lw=2)
        plt.plot(f_band, psd_band, color='g', lw=3)
        plt.scatter(df, psd_df, color='r', marker='o')
        plt.xlim([freqs.min(), freqs.max()])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power Spectral Density [μV2/Hz]')
        plt.tight_layout()
        plt.show()

    return df, ap_df, rp_df

def featureFunc_bandAbsolutePower(psds, freqs, band, channels=None, norm_and_log_density=None,
                                  compute_magnitude=None, plot_psd=None):
    '''
    Calculate absolute power in a given frequency band for a given set of channels as used
    by Bauer 2001. Optionally, normalize to bandwidth and log-transform absolute power density
    as done by Rangaswamy et al. 2002.

    The power spectral densities are averaged over channels. Estimates of power within a
    frequency band are estimated by integrating the averaged power spectral density over the
    corresponding frequency band intervals.

    Parameters
    ----------
    psds : 2D array
        The power spectral densities of all channels [n_channels, n_freqs].
    freqs : 1D array
        Frequencies in Hertz.
    band : tuple
        The lowest and highest frequencies to integrate the power spectral densities over.
    channels : list of int | None
        Channel indices for channels to include (None, default, includes all).
    norm_and_log_density : bool | None
        Whether or not to normalize to bandwidth and log-transform absolute band power as done by
        Rangaswamy et al. 2002. None, default, does not.
    compute_magnitude: bool | None
        Whether to report magnitude (square root of absolute power) instead of absolute power in
        the given frequency band. None, default, does not.
    plot_psd: bool | None
        Whether to plot the power spectral densities used for calculating the band absolute power.
        None, default, does not.

    Returns
    -------
    res : scalar
        The integrated absolute power in the given frequency band.
        [Optionally, if norm_and_log_dens == True] :-
        The integrated absolute power in the given frequency band after normalization to
        bandwidth and log-transformation.
        [Optionally, if compute_magnitude == True] :-
        The square root of the integrated absolute power in the given frequency band, either with
        or without normalization to bandwidth and log-transformation.
    '''
    if channels is None:
        mean_psd = np.mean(psds, axis=0)
    elif len(channels) > 1:
        mean_psd = np.mean(psds[channels, :], axis=0)
    else:
        mean_psd = psds[channels, :].reshape((-1,))

    # Plot FFT
    if plot_psd and plot_psd is not None:
        plt.figure()
        plt.plot(freqs, mean_psd, color='k', lw=2)
        plt.xlim([0, 50])
        # plt.xlim([freqs.min(), freqs.max()])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power Spectral Density [μV2/Hz]')
        plt.tight_layout()
        plt.show()

    # Integrate PSD in the desired frequency band
    res = _integrate_psd(mean_psd, freqs, band)

    # Scale and log
    if norm_and_log_density and norm_and_log_density is not None:
        res = np.log(res / (band[1] - band[0]))

    # Compute frequency magnitude (in μV)
    if compute_magnitude and compute_magnitude is not None:
        res = np.sqrt(res)

    return res

def featureFunc_bandRelativePower(psds_mean_epochs, freqs, band_target, band_range,
                                  channels=None, norm_and_log_density=None, plot_psd=None):
    '''
    Get absolute power (AP) measures using the desired AP implementation in the target frequency
    band and the total frequency range to be studied. Subsequently, calculate relative power (RP)
    of the target band.

    Parameters
    ----------
    psds_mean_epochs : 2D array
        Power spectral densities for all channels averaged over epochs
        [n_channels, n_freqs].
    freqs : 1D array
        Frequencies in Hertz.
    band_target : tuple
        The lowest and highest frequencies of the target band.
    band_range : tuple
        The lowest and highest frequencies of the studied frequency range.
    channels : list of int | None
        Channel indices for channels to include (None, default, includes all).
    norm_and_log_density : bool | None
        Whether or not to normalize to bandwidth and log-transform absolute band power as done by
        Rangaswamy et al. 2002. None, default, does not.
    plot_psd: bool | None
        Whether to plot the power spectral densities used for calculating the band absolute power.
        None, default, does not.

    Returns
    -------
    rp_target : scalar
        The relative power in the given target frequency band.
        [Optionally, if norm_and_log_dens == True] :-
        The relative power in the given target frequency band after normalization to bandwidth
        and log-transformation.
    '''

    ap_target = featureFunc_bandAbsolutePower(psds_mean_epochs, freqs, band_target,
                                             channels=channels,
                                             norm_and_log_density=norm_and_log_density,
                                             plot_psd=plot_psd)
    ap_range = featureFunc_bandAbsolutePower(psds_mean_epochs, freqs, band_range,
                                            channels=channels,
                                            norm_and_log_density=norm_and_log_density,
                                            plot_psd=None)
    return ap_target / ap_range

def psd_M1(params, raw, plot_psd_epochs=None):
    '''
    Bauer2001
    All EEG epochs are detrended and cosine tapered. EEG epochs with zero-
    overlap are submitted to a Fast Fourier Transform which computes the power
    spectral density (in μV2/Hz) of the EEG. The PSDs are averaged over epochs.

    Parameters
    ----------
    params : dict
        Chosen parameters to be used in calculations.
    raw : MNE-Python Raw object
        Raw file instance under study
    plot_psd_epochs: bool | None
        Whether to plot the power spectral densities for each epoch or not.
        None, default, does not.

    Returns
    -------
    psds_mean_epochs : 2D array
        Power spectral densities for all channels averaged over epochs
        [n_channels, n_freqs].
    freqs : 1D array
        Frequencies in Hertz.
    '''
    # Unpack parameters
    T_max = params['T_max']
    print('A total of %d seconds of raw EEG data has been studied.' % T_max)
    Fs = params['sfreq']
    print('Fs =', Fs)
    T_ep = params['size_epoch']
    print('Each epoch has %0.1f seconds of EEG data.' % T_ep)
    times = params['epoch_times']
    print('Shape of times =', np.shape(times))

    # Get raw data
    raw = raw.copy()
    # raw.del_proj(0) # delete Average EEG reference
    raw_data = raw.get_data(picks='eeg')
    raw_data = raw_data[:, :int(Fs * T_max)]
    print('Shape of raw_data =', np.shape(raw_data))

    # Data segmentation and PSD calculation
    n_epochs = int(np.floor(T_max / T_ep))
    print('Total number of epochs studied =', n_epochs)
    epochs = []; psds = []
    for i_ep in range(n_epochs):
        # Segment data into epochs
        start, stop = int(i_ep * np.floor(T_ep * Fs)), int((i_ep + 1) * np.floor(T_ep * Fs))
        ep_data = raw_data[:, start:stop]
        epochs.append(ep_data)
        # Compute PSDs
        psds_ep, freqs = _computeFFTPSD(ep_data, Fs, detrend=True, window='cosine',
                                       plot_psd_mean=plot_psd_epochs)
        psds.append(psds_ep * 1e12) # in μV^2
    print('Shape of epochs =', np.shape(epochs))
    print('Shape of psds =', np.shape(psds))
    psds_mean_epochs = np.mean(psds, axis=0) # Average psds over epochs

    return psds_mean_epochs, freqs

def psd_M2(params, raw, plot_psd_epochs=None):
    '''
    Rangaswamy2002
    The referencing is changed from monopolar to bipolar. Fourier transform is
    applied to overlapping epochs (overlap = 50%) after Hamming windowing. The
    resulting power spectral densities are averaged across epochs.
    '''
    # Unpack parameters
    T_max = params['T_max']
    print('A total of %d seconds of raw EEG data has been studied.' % T_max)
    Fs = params['sfreq']
    print('Fs =', Fs)
    T_ep = params['size_epoch']
    print('Each epoch has %0.1f seconds of EEG data.' % T_ep)
    times = params['epoch_times']
    print('Shape of times =', np.shape(times))

    # Make a copy of raw data and delete projector for Average EEG reference
    raw = raw.copy()
    # raw.del_proj(0) # delete Average EEG reference

    # Get anodes and cathodes of bipolar montage
    bipolar_cnx = params['bipolar_connections']
    anode = []; cathode = []
    for a, c in bipolar_cnx.values():
        anode.append(a); cathode.append(c)

    # Set referencing as bipolar
    raw.load_data()
    raw_bip_ref = mne.set_bipolar_reference(raw, anode=anode, cathode=cathode)
    # raw_bip_ref.drop_channels('M2')
    # print(raw_bip_ref.ch_names)

    # Get raw data
    raw_data = raw_bip_ref.get_data(picks='eeg')
    raw_data = raw_data[:, :int(Fs * T_max)]
    print('Shape of raw_data =', np.shape(raw_data))

    # Data segmentation and PSD calculation
    n_win = int(np.floor(T_max / T_ep))
    n_epochs = int(1 + 2 * (n_win - 1))
    print('Total number of epochs studied =', n_epochs)
    epochs = []; psds = []
    for i_ep in range(n_epochs):
        # Segment data into epochs
        start, stop = int(i_ep / 2 * np.floor(T_ep*Fs)), int((i_ep / 2 + 1) * np.floor(T_ep*Fs))
        ep_data = raw_data[:, start:stop]
        epochs.append(ep_data)
        # Compute PSDs
        psds_ep, freqs = _computeFFTPSD(ep_data, Fs, detrend=False, window='hamming',
                                           plot_psd_mean=plot_psd_epochs)
        psds.append(psds_ep * 1e12) # in μV^2
    print('Shape of epochs =', np.shape(epochs))
    print('Shape of psds =', np.shape(psds))
    psds_mean_epochs = np.mean(psds, axis=0) # Average psds over epochs

    return psds_mean_epochs, freqs

def psd_M3(params, raw, plot_psd_epochs=None):
    '''
    Winterer1998
    EEG epochs are submitted to a Fast Fourier Transform which computes the power spectral
    densities (PSDs) of the EEG. The PSDs are then averaged over epochs.
    '''
    # Unpack parameters
    T_max = params['T_max']
    print('A total of %d seconds of raw EEG data has been studied.' % T_max)
    Fs = params['sfreq']
    print('Fs =', Fs)
    T_ep = params['size_epoch']
    print('Each epoch has %0.1f seconds of EEG data.' % T_ep)
    times = params['epoch_times']
    print('Shape of times =', np.shape(times))

    # Get raw data
    raw = raw.copy()
    # raw.del_proj(0) # Delete Average EEG reference
    raw_data = raw.get_data(picks='eeg')
    raw_data = raw_data[:, :int(Fs * T_max)]
    print('Shape of raw_data =', np.shape(raw_data))

    # Data segmentation and PSD calculation
    n_epochs = int(np.floor(T_max / T_ep))
    print('Total number of epochs studied =', n_epochs)
    epochs = []; psds = []
    for i_ep in range(n_epochs):
        # Segment data into epochs
        start, stop = int(i_ep * np.floor(T_ep * Fs)), int((i_ep + 1) * np.floor(T_ep * Fs))
        ep_data = raw_data[:, start:stop]
        epochs.append(ep_data)
        # Compute PSDs
        psds_ep, freqs = _computeFFTPSD(ep_data, Fs, detrend=False, window=False,
                                         plot_psd_mean=plot_psd_epochs)
        psds.append(psds_ep * 1e12) # in μV^2
    print('Shape of epochs =', np.shape(epochs))
    print('Shape of psds =', np.shape(psds))
    psds_mean_epochs = np.mean(psds, axis=0) # Average psds over epochs

    return psds_mean_epochs, freqs

def psd_M4(params, raw, plot_psd_epochs=None):
    '''
    Saletu-Zyhlarz2004
    EEG epochs are submitted to a Fast Fourier Transform which computes the power spectral
    densities (PSDs) of the EEG. The PSDs are then averaged over epochs.
    '''
    # Unpack parameters
    T_max = params['T_max']
    print('A total of %d seconds of raw EEG data has been studied.' % T_max)
    Fs = params['sfreq']
    print('Fs =', Fs)
    T_ep = params['size_epoch']
    print('Each epoch has %0.1f seconds of EEG data.' % T_ep)
    times = params['epoch_times']
    print('Shape of times =', np.shape(times))

    # Get raw data
    raw = raw.copy()
    # raw.del_proj(0) # Delete Average EEG reference
    raw_data = raw.get_data(picks='eeg')
    raw_data = raw_data[:, :int(Fs * T_max)]
    print('Shape of raw_data =', np.shape(raw_data))

    # Data segmentation and PSD calculation
    n_epochs = int(np.floor(T_max / T_ep))
    print('Total number of epochs studied =', n_epochs)
    epochs = []; psds = []
    for i_ep in range(n_epochs):
        # Segment data into epochs
        start, stop = int(i_ep * np.floor(T_ep * Fs)), int((i_ep + 1) * np.floor(T_ep * Fs))
        ep_data = raw_data[:, start:stop]
        epochs.append(ep_data)
        # Compute PSDs
        psds_ep, freqs = _computeFFTPSD(ep_data, Fs, detrend=False, window=False,
                                         plot_psd_mean=plot_psd_epochs)
        psds.append(psds_ep * 1e12) # in μV^2
    print('Shape of epochs =', np.shape(epochs))
    print('Shape of psds =', np.shape(psds))
    psds_mean_epochs = np.mean(psds, axis=0) # Average psds over epochs

    return psds_mean_epochs, freqs

def psd_M5(params, raw, plot_psd_epochs=None):
    '''
    Default2022
    All EEG epochs are detrended and cosine tapered. EEG epochs with zero-
    overlap are submitted to a Fast Fourier Transform which computes the power
    spectral density (in μV^2/Hz) of the EEG. The power spectral densities are
    then averaged over epochs.
    '''
    # Unpack parameters
    T_max = params['T_max']
    print('A total of %d seconds of raw EEG data has been studied.' % T_max)
    Fs = params['sfreq']
    print('Fs =', Fs)
    T_ep = params['size_epoch']
    print('Each epoch has %0.1f seconds of EEG data.' % T_ep)
    times = params['epoch_times']
    print('Shape of times =', np.shape(times))

    # Get raw data
    raw = raw.copy()
    # raw.del_proj(0) # delete Average EEG reference
    raw_data = raw.get_data(picks='eeg')
    raw_data = raw_data[:, :int(Fs * T_max)]
    print('Shape of raw_data =', np.shape(raw_data))

    # Data segmentation and PSD calculation
    n_epochs = int(np.floor(T_max / T_ep))
    print('Total number of epochs studied =', n_epochs)
    epochs = []; psds = []
    for i_ep in range(n_epochs):
        # Segment data into epochs
        start, stop = int(i_ep * np.floor(T_ep * Fs)), int((i_ep + 1) * np.floor(T_ep * Fs))
        ep_data = raw_data[:, start:stop]
        epochs.append(ep_data)
        # Compute PSDs
        psds_ep, freqs = _computeFFTPSD(ep_data, Fs, detrend=True, window='cosine',
                                       plot_psd_mean=plot_psd_epochs)
        psds.append(psds_ep * 1e12) # in μV^2
    print('Shape of epochs =', np.shape(epochs))
    print('Shape of psds =', np.shape(psds))
    psds_mean_epochs = np.mean(psds, axis=0) # Average psds over epochs

    return psds_mean_epochs, freqs

def save_bipolar_connections(cnx, fname_bipolar_connections='fname.pickle'):
    with open(fname_bipolar_connections, 'wb') as f:
        pickle.dump(cnx, f, pickle.HIGHEST_PROTOCOL)
    return None

def load_bipolar_connections(fname_bipolar_connections):
    with open(fname_bipolar_connections, 'rb') as f:
        return pickle.load(f)

def _computeFFTPSD(X, Fs, times=None, detrend=None, baseline=None, window=None,
                  plot_psd_mean=None):
    '''
    Compute the Fourier power spectral density of the N-channel block `X`
    [nChannels, nSamples] sampled at frequency `Fs`.
    '''

    # Detrend data
    if detrend and detrend is not None:
        X = __detrend(X)

    # Baseline correction
    if baseline is not None:
        # Baseline correction
        X = __baseline_correction(X, times, baseline=baseline)
        # Discard baseline data
        qmin = np.where(times <= 0)[0]
        qmin = int(qmin[-1]) + 1
        X = X[:, qmin:]

    # Window size
    n = np.size(X, 1)

    # Windowing
    if window and window is not None:
        if window == 'hamming':
            w = np.hamming(n)
        elif window == 'hanning':
            w = np.hanning(n)
        elif window == 'cosine':
            w = signal.tukey(n)
        X = w * X

    # FFT using Numpy
    Xhat = np.fft.rfft(X)
    psd = np.real(Xhat * np.conj(Xhat)) / n # FFT PSD
    freq = Fs / n * np.arange(n)
    k = np.arange(1, np.floor(n / 2), dtype='int') # Only use the first half of frequencies
    freq = freq[k]; psd = psd[:, k];


    # Plot power spectrum
    if plot_psd_mean and plot_psd_mean is not None:
        psd_mean = np.mean(psd, axis=0)
        plt.figure()
        plt.plot(freq, psd_mean)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power Spectral Density [V2/Hz]')
        plt.show()

    return psd, freq

def _integrate_psd(psd, f, band):
    '''
    Integrate power spectral densities over desired frequency band.
    '''
    inds_range = [(np.abs(f - band[0])).argmin(), (np.abs(f - band[1])).argmin()]
    inds = np.arange(inds_range[0], inds_range[1]+1)
    return np.trapz(psd[inds], f[inds])
    # return sp.integrate.simps(psd[inds], f[inds])

def __detrend(data, detrend_type='linear', bp=0):
    """
    Remove linear trend along second axis from 2D array-like object `data`.
                            [Adapted from MNE-Python]
    """
    axis = -1
    if detrend_type not in ['linear', 'l', 'constant', 'c']:
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    data = np.asarray(data)
    if detrend_type in ['constant', 'c']:
        data_detrended = data - np.mean(data, axis, keepdims=True)
        return data_detrended
    else:
        dshape = data.shape
        N = dshape[axis]
        bp = np.sort(np.unique(np.r_[0, bp, N]))
        if np.any(bp > N):
            raise ValueError("Breakpoints must be less than length "
                             "of data along given axis.")
        Nreg = len(bp) - 1
        # Find leastsq fit and remove it for each piece
        newdata = data.T
        for m in range(Nreg):
            Npts = bp[m + 1] - bp[m]
            A = np.ones((Npts, 2))
            A[:, 0] = np.arange(1, Npts+1) / Npts
            sl = slice(bp[m], bp[m+1])
            coef, _, _, _ = np.linalg.lstsq(A, newdata[sl], rcond=None)
            newdata[sl] = newdata[sl] - np.dot(A, coef)
        data_detrended = newdata.T

        return data_detrended

def __baseline_correction(data, times, baseline=(None, 0)):
    '''
    Apply baseline correction to `data`.
                            [Adapted from MNE-Python]
    '''
    bmin, bmax = baseline
    if bmin is None:
        imin = 0
    else:
        imin = np.where(times >= bmin)[0]
        imin = int(imin[0])
    if bmax is None:
        imax = len(times)
    else:
        imax = np.where(times <= bmax)[0]
        imax = int(imax[-1]) + 1

    baseline = np.mean(data[..., imin:imax], axis=-1, keepdims=True)
    corrected_data = data - baseline

    return corrected_data
