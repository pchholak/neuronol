import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt

def emd(x, t=None, tol_sd=0.2, max_IMFs=25, max_siftings=100, plot_emd=None):
    """
    Perform empirical mode decomposition on a signal 'x' as described in Huang et al. 1998.
    The decomposition terminates whence either the sifting process is unable to find local
    peaks or valleys in the residual signal or the max. no. of intended IMFs have already
    been extratced.

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
