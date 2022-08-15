import numpy as np
import matplotlib.pyplot as plt
from mathphys.functions import save_pickle, load_pickle
from apsuite.commisslib.emit_exchange import EmittanceExchangeSimul


def make_Rmap(S_list, C_list):
    """."""
    R_matrix = np.zeros([len(S_list), len(C_list)])
    init_delta = -0.073
    for i, s in enumerate(S_list):
        for j, c in enumerate(C_list):
            n_turns = int(np.abs(init_delta)/(s * c**2))
            if n_turns > 8000:
                print('Simulation canceled, N>8e3.')
                R_matrix[i, j] = np.nan
                continue
            else:
                simul = EmittanceExchangeSimul(
                    c=c, s=s, init_delta=init_delta, radiation=True)
                simul.dynamic_emit_exchange_envelope(verbose=False)
                r, rmax_idx = simul._calc_exchange_quality()
                r_max = r[rmax_idx]
                R_matrix[i, j] = r_max
    return R_matrix


def plot_rmap(S_list, C_list, R, initial_delta=-0.073,):
    """."""
    T_r = 1.6e-3  # ms
    S_mesh, C_mesh = np.meshgrid(S_list[:19], C_list, indexing='ij')
    with plt.style.context(['science', 'grid']):
        fig, ax = plt.subplots()
        t_mesh = np.abs(initial_delta)/(S_mesh*C_mesh**2) * T_r
        img = ax.pcolormesh(t_mesh, C_mesh*100, R, shading='auto')
        nivel_curves = [0.5, 1, 2, 3.5]
        cs = ax.contour(
            t_mesh, C_mesh*1e2, S_mesh, levels=nivel_curves, colors='k',
            linestyles='dashed', linewidths=1)
        ax.clabel(cs, fontsize=9, inline=1, fmt='S = %1.1f')
        ax.set_xlabel('$t_c$ [ms]')
        ax.set_ylabel(r'$|C| \;[\%]$')
        ax.set_xscale("log")
        ax.set_ylim([C_mesh.min()*1.1e2, C_mesh.max()*0.9e2])
        ax.tick_params(axis='both', which='both', direction='out')
        plt.colorbar(img, label='max$(R)$')
        plt.savefig(fname='new_R_map.pdf')


def load_and_plot():
    """."""
    data = load_pickle('r_map_data.pickle')
    S = data['S']
    C = data['C']
    R_matrix = data['R matrix']
    plot_rmap(S, C, R_matrix)


if __name__ == "__main__":
    """."""
    S_list = np.logspace(-1, 1.5, num=15)
    C_list = np.linspace(0.4, 1.5, num=15)*1e-2
    R_matrix = make_Rmap(S_list, C_list)
    data = dict()
    data['S'] = S_list
    data['C'] = C_list
    data['R matrix'] = R_matrix
    save_pickle(fname='r_map_data', data=data)
    plot_rmap(S_list, C_list, R_matrix)
