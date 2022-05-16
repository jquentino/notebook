import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from siriuspy.devices import Screen


def extract_quad_ramp(boramp):
    """."""
    conf_times = np.sort(boramp.ps_normalized_configs_times)
    qf_ramp = np.zeros(len(conf_times))
    qd_ramp = qf_ramp.copy()
    for i, time in enumerate(conf_times):
        qf_ramp[i] = boramp[time]['BO-Fam:PS-QF']
        qd_ramp[i] = boramp[time]['BO-Fam:PS-QD']
    return qf_ramp, qd_ramp


def emittances_from_beam_size(sigmas, twiss, energy_spread):
    """."""
    etas = np.array([twiss.etax, twiss.etay])[:, None]
    betas = np.array([twiss.betax, twiss.betay])[:, None]
    emittances = (sigmas**2 - (etas*energy_spread)**2)/betas
    return emittances


def gauss(x, amp, x0, sigma, off):
    """."""
    return amp*np.exp(-(x-x0)**2/(2*sigma**2)) + off


def extraction_analysis(measures, ts_number):
    """."""
    # ts_screens = [
    #     Screen.DEVICES.TS_1, Screen.DEVICES.TS_2, Screen.DEVICES.TS_3,
    #     Screen.DEVICES.TS_4, Screen.DEVICES.TS_5, Screen.DEVICES.TS_6
    #     ]
    # ts_scrn = Screen(ts_screens[ts_number-1])
    -1.792273e-2
#     scl_fact_x = np.abs(ts_scrn.scale_factor_x)  # pixel to mm
#     scl_fact_y = np.abs(ts_scrn.scale_factor_y)
    scl_fact_x = scl_fact_y = 1.792273e-2
    sigma_arr = np.zeros([6, len(measures)])
    sigma_std_arr = np.zeros([2, len(measures)])

    for j, data in enumerate(measures):
        sigmax = []
        sigmay = []
        error_fitx = []
        error_fity = []
        n_images = len(data['images'])
        for i in range(n_images):
            image = data['images'][i]
            projx = np.sum(image, axis=0)
            projy = np.sum(image, axis=1)
            x_mean_idx = np.argmax(projx)
            y_mean_idx = np.argmax(projy)
            hline = image[y_mean_idx, :]
            vline = image[:, x_mean_idx]
            xx = np.arange(hline.size)
            xy = np.arange(vline.size)
            sigx = 10
            sigy = 10
            poptx, pcovx = curve_fit(
                gauss, xx, hline, p0=[
                    np.max(hline), x_mean_idx, sigx, hline[0]
                    ])
            popty, pcovy = curve_fit(
                gauss, xy, vline, p0=[
                    np.max(vline), y_mean_idx, sigy, vline[0]
                    ])

            sigmax.append(poptx[2] * scl_fact_x)  # sigmas stored in mm
            sigmay.append(popty[2] * scl_fact_y)  #

            error_fitx.append(np.sqrt(pcovx[2, 2]) * scl_fact_x)
            error_fity.append(np.sqrt(pcovy[2, 2]) * scl_fact_y)

        sigma_arr[:, j] = np.array([sigmax, sigmay]).ravel()

        sigma_std_arr[:, j] = np.array([
            np.sqrt(np.var(sigmax) + np.mean(error_fitx)**2),
            np.sqrt(np.var(sigmay) + np.mean(error_fity)**2)]).ravel()

    delay = - np.array(
        [measures[i]['delay_ramp'][0] for i in range(len(measures))])

    return delay, sigma_arr, sigma_std_arr


def plot_ext_time_scan(
        measures, ts_number, time_offset = 0,figname=None, legend=True, axis=None):
    """."""
    delay, sigmas, usigmas = extraction_analysis(measures, ts_number)
    sx = sigmas[:3, :]
    sy = sigmas[3:, :]
    mean_sx = sx.mean(axis=0)
    mean_sy = sy.mean(axis=0)
    rep_delay = np.tile(delay, 3)
    rep_delay += time_offset
    delay += time_offset
    with plt.style.context(['science', 'scatter']):
        if axis is None:
            fig, ax = plt.subplots()
        else:
            ax = axis
        ax.scatter(
            rep_delay, sx.ravel(), c='C0', s=1.5, label=r'measured $\sigma_x$',
            marker='v')
        ax.scatter(
            rep_delay, sy.ravel(), c='tab:orange', s=1.5,
            label=r'measured $\sigma_y$', marker='^')
        ax.errorbar(
            delay, mean_sx, yerr=usigmas[0], ls='-', marker='', c='C0',
            label=r'$<\sigma_x>$')
        ax.errorbar(
            delay, mean_sy, yerr=usigmas[1], ls='-', marker='', c='tab:orange',
            label=r'$<\sigma_y>$')
        ax.set_xlabel(r'$\Delta t_{ext}$ [ms]')
        ax.set_ylabel(r'$\sigma_{x,y}$ [mm]')
        if legend:
#             ax.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
              ax.legend(loc='best', fontsize=8, frameon=True)
        if figname is not None:
            plt.savefig(figname)
            plt.show()
    # return fig, ax
