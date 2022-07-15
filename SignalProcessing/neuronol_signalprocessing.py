import cv2
import numpy as np
import scipy as sp
from scipy import signal
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import pandas as pd
from PyEMD import EMD

def perform_EMD(x, plot_emd=None):
    '''
    Perform empirical mode decomposition on signal block 'x'.
    '''
    # EMD
    emd_decomp = EMD()
    imfs = emd_decomp(x)

    # Visualize EMD
    if plot_emd and plot_emd is not None:
        plt.figure(figsize=(12, 12))
        for i in range(len(imfs)-1):
            plt.subplot(len(imfs)+1, 1, i+1)
            plt.plot(t, x, color='0.8')
            plt.plot(t, imfs[i], 'k')
            plt.xlim([np.min(t), np.max(t)])
            plt.ylabel('IMF ' + str(i + 1))
        plt.subplot(len(imfs)+1, 1, i+2)
        plt.plot(t, x, color='0.8')
        plt.plot(t, imfs[-1], 'k')
        plt.xlim([np.min(t), np.max(t)])
        plt.ylabel('Residual')
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()
    return imfs

def _downsample_rows(arr, k):
    '''
    Downsample a measurement matrix along its rows.
    '''
    res = np.cumsum(arr, 0)[k-1::k]
    res[1:] = res[1:] - res[:-1]
    return res / k

def calculate_hilbert_spectrum(imfs, t, fs, n=5, k_dwnsamp=3, k_gauss=15,
                               smoothing_downsample_freq=None, smoothing_gauss_filt=None,
                               plot_marginal_hilbert_spec=None, plot_hilbert_spec=None,
                               plot_inst_freq=None):
    '''
    Calculate hilbert amplitude spectrum from a given set of intrinsic mode functions.
    '''

    ## Create Hilbert spectrum
    T = t[-1] - t[0]; delta_t = 1 / fs
    fmin = fres = 1 / T; fmax = 1 / (n * delta_t)
    N = int(T / (n * delta_t))
    bin_centres = np.arange(N) * fres + fmin
    bin_edges = np.arange(N + 1) * fres + (fmin - fres / 2)

    f_hht = bin_centres
    hhts = np.zeros((len(imfs), N, (len(t) - 2)))

    for j, imf in enumerate(imfs):
        Z = hilbert(imf)
        A = np.abs(Z)
        theta_inst = np.unwrap(np.angle(Z))
        f_inst = 0.5 * (np.angle(-Z[2:] * np.conj(Z[:-2])) + np.pi) / (2 * np.pi) * fs
        t_hht = t[1:-1]; A_hht = A[1:-1]

        # Plot instantaneous frequency curves
        if plot_inst_freq and plot_inst_freq is not None:
            fig, (ax0, ax1) = plt.subplots(nrows=2)
            ax0.plot(t, imf, label='signal')
            ax0.plot(t, A, label='envelope')
            ax0.set_xlabel("time (s)")
            ax0.set_ylabel("signal (units)")
            ax0.legend()
            ax1.plot(t_hht, f_inst)
            ax1.set_xlabel("time (s)")
            ax1.set_ylabel("frequency (Hz)")
            fig.tight_layout()
            plt.show()

        # Binning of frequency values
        binned_freq = pd.cut(f_inst, bin_edges)
        bin_inds = binned_freq.codes

        # Populate Hilbert spectrum matrix
        for i, bin_ind in enumerate(bin_inds):
            if bin_ind > 0:
                hhts[j][bin_ind][i] = A_hht[i]

    hht = np.sum(hhts, axis=0)

    # Smoothing - Downsample Frequency in HHT
    if smoothing_downsample_freq and smoothing_downsample_freq is not None:
        hht = _downsample_rows(hht, k_dwnsamp)
        f_hht = _downsample_rows(f_hht, k_dwnsamp)

    # Smoothing - Weighted Gaussian Filtering
    if smoothing_gauss_filt and smoothing_gauss_filt is not None:
        hht = cv2.GaussianBlur(hht, (k_gauss, k_gauss), 0)

    # Calculate marginal Hilbert spectrum
    marginal_spec = np.mean(hht, axis=1)

    # Plot Hilbert spectrum for all IMFs
    if plot_hilbert_spec and plot_hilbert_spec is not None:
        plt.figure()
        plt.pcolormesh(t_hht, f_hht, hht)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.show()

    # Plot marginal Hilbert spectrum
    if plot_marginal_hilbert_spec and plot_marginal_hilbert_spec is not None:
        plt.figure()
        plt.plot(f_hht, marginal_spec)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Marginal Hilbert Spectrum')
        plt.show()

    return hht, t_hht, f_hht, marginal_spec

def Appendix__imperfect_emd(x, t=None, tol_sd=0.2, max_IMFs=25, max_siftings=100, plot_emd=None):
    """
    Perform empirical mode decomposition on a signal 'x' as described in Huang et al. 1998.
    The decomposition terminates whence either the sifting process is unable to find local
    peaks or valleys in the residual signal or the max. no. of intended IMFs have already
    been extracted.

    Parameters
    ----------
    x : 1D array
        Signal of interest.
    t : 1D array
        Time (or space) at which elements of 'x' were measured.
    tol_sd : scalar
        Tolerance in standard deviation between two consecutive siftings. Used as a stopping
        criterion. See Eq. (5.5) in Huang et al. 1998 for more details.
    max_IMFs : scalar
        Max. no. of IMFs to be extracted.
    max_siftings : scalar
        Max. no. of siftings to be performed for extracting each IMF.

    Returns
    -------
    c : 2D array
        IMFs extracted from the EMD.
    r : 1D array
        Residual signal.
    """
    if t is None:
        t = np.array(range(len(x)))
        label_x = ''
    else:
        label_x = 'Time (s)'
    r = x
    c = np.zeros((max_IMFs, len(x)))
    i = 0; stop_emd = False
    while (i < max_IMFs) and not stop_emd:
        print(i + 1)
        h_km1 = r
        k = 1
        while k < max_siftings:
            # Find upper and lower extrema and include first and last point of the signal
            j_pks = signal.argrelmax(h_km1)[0]; j_pks = np.append(np.append(0, j_pks), -1)
            j_vls = signal.argrelmin(h_km1)[0]; j_vls = np.append(np.append(0, j_vls), -1)

            # Make upper and lower envelopes
            if len(j_pks) > 3:
                spl_up = sp.interpolate.InterpolatedUnivariateSpline(t[j_pks], h_km1[j_pks], k=3)
                envp_up = spl_up(t)
            elif len(j_pks) > 2:
                spl_up = sp.interpolate.InterpolatedUnivariateSpline(t[j_pks], h_km1[j_pks], k=2)
                envp_up = spl_up(t)
            else:
                print('No local peaks found! Stopping.')
                stop_emd = True
                break
            if len(j_vls) > 3:
                spl_lw = sp.interpolate.InterpolatedUnivariateSpline(t[j_vls], h_km1[j_vls], k=3)
                envp_lw = spl_lw(t)
            elif len(j_vls) > 2:
                spl_lw = sp.interpolate.InterpolatedUnivariateSpline(t[j_vls], h_km1[j_vls], k=2)
                envp_lw = spl_lw(t)
            else:
                print('No local valleys found! Stopping.')
                stop_emd = True
                break

            # Calculate mean envelope
            m_k = (envp_up + envp_lw) / 2

            # Find the next sifted signal
            h_k = h_km1 - m_k

            # Test for stopping criterion
            sd = np.sum((h_km1 - h_k) ** 2) / np.sum(h_km1 ** 2)
            print(sd)
            if sd < tol_sd:
                print('IMF found!')
                c[i] = h_k
                break
            else:
                k += 1
                h_km1 = h_k
        r = r - c[i]
        i += 1

    # Delete extra zero rows that haven't been populated
    c = np.delete(c, range(i-1, max_IMFs), axis=0)

    # Plot
    if plot_emd and plot_emd is not None:
        plt.figure(figsize=(12, 12))
        for i in range(len(c)):
            plt.subplot(len(c)+1, 1, i+1)
            plt.plot(t, x, color='0.8')
            plt.plot(t, c[i], 'k')
            plt.xlim([np.min(t), np.max(t)])
            plt.ylabel('IMF ' + str(i + 1))
        plt.subplot(len(c)+1, 1, i+2)
        plt.plot(t, x, color='0.8')
        plt.plot(t, r, 'k')
        plt.xlim([np.min(t), np.max(t)])
        plt.ylabel('Residual')
        plt.xlabel(label_x)
        plt.tight_layout()
        plt.show()

    return c, r

def Appendix__derivative_central_secondorder(x, fs):
    '''
    Calculate derivative of signal 'x' sampled at frequency 'fs'
    using central difference method of second order.
    '''
    n = len(x)
    dxdt = np.zeros((n, 1))
    dxdt[0] = (x[1] - x[0]) * fs # 1 / dt = fs
    for i in range(1, n-1):
        dxdt[i] = (x[i+1] - x[i-1]) / 2 * fs
    dxdt[n-1] = (x[n-1] - x[n-2]) * fs
    return dxdt

def Appendix__derivative_central_fourthorder(x, fs):
    '''
    Calculate derivative of signal 'x' sampled at frequency 'fs'
    using central difference method of second order.
    '''
    n = len(x)
    dxdt = np.zeros((n, 1))
    dxdt[0] = (x[1] - x[0]) * fs # 1 / dt = fs
    dxdt[1] = (x[2] - x[0]) * fs / 2
    for i in range(2, n-2):
        dxdt[i] = (- x[i+2] + 8*x[i+1] - 8*x[i-1] + x[i-2]) * fs / 12
    dxdt[n-2] = (x[n-1] - x[n-3]) * fs / 2
    dxdt[n-1] = (x[n-1] - x[n-2]) * fs
    return dxdt
