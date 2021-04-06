from datetime import datetime

import mesa_vs
import numpy as np
from astropy import constants as const
from astropy.io import ascii
from matplotlib import rcParams
from plots import Convective_parameter
from scipy.integrate import simps

rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['text.latex.preamble'] = [r'\usepackage[utf8]{inputenc}', r'\usepackage{amsfonts, amsmath, amsthm, amssymb}',
                                   r'\usepackage[english]{babel}']

G = const.G.cgs.value
M_sun = const.M_sun.cgs.value
R_sun = const.R_sun.cgs.value
c = const.c.cgs.value
pl_const = const.h.cgs.value
k_B = const.k_B.cgs.value
kpc = const.kpc.cgs.value


def plank(nu, T):
    a = 2 * pl_const * nu ** 3 / c ** 2
    return a / np.expm1(pl_const * nu / (k_B * T))


def main():
    abundance = {'h1': 0.75, 'he4': 0.25}
    # abundance = 'solar'
    path = 'fig/Aql-X_1_Shakura/075h1_+_025he4/01alpha/'

    time0 = datetime.now()
    data = ascii.read('F_xspec.dat')
    time = list(data['col1'])
    f = list(data['col3'])
    N = 100
    incl = 40
    d = 5 * kpc
    M = 1.5 * M_sun
    eta_accr = 0.1
    alpha = 0.1  # alpha = 0.1, 0.5

    Mdot_plot = []
    nuFnu_plot = np.empty((len(time), 8))
    for j, flux in enumerate(f):
        print('j = ', j)
        Mdot = 4 * np.pi * d ** 2 * flux / (eta_accr * c ** 2)
        Teff_plot, Tirr_plot, Tvis_plot, T_C_plot, Sigma0_plot = np.empty(N), [], [], [], []
        r_plot = np.geomspace(0.0001 * R_sun, 1.8 * R_sun, N)
        conv_param_z0_plot, conv_param_sigma_plot = [], []

        for i, r in enumerate(r_plot):
            print('i =', i)
            h = np.sqrt(G * M * r)
            rg = 2 * G * M / c ** 2
            r_in = 3 * rg
            func = 1 - np.sqrt(r_in / r)
            # vs = mesa_vs.MesaVerticalStructureRadConvExternalIrradiationZeroAssumption(M, alpha, r, Mdot * h * func,
            # abundance=abundance)
            vs = mesa_vs.MesaVerticalStructureRadConv(M, alpha, r, Mdot * h * func, abundance=abundance)
            vs.fit()

            conv_param_z0, conv_param_sigma = Convective_parameter(vs)

            conv_param_z0_plot.append(conv_param_z0)
            conv_param_sigma_plot.append(conv_param_sigma)
            Teff_plot[i] = vs.Teff
            # Tirr_plot.append(vs.Tirr)
            # Tvis_plot.append(vs.Tvis)
            T_C_plot.append(vs.parameters_C()[2])
            Sigma0_plot.append(vs.parameters_C()[4])
        Mdot_plot.append(Mdot)

        np.savetxt(path + 'conv_z0{}_non_irr.txt'.format(j), conv_param_z0_plot)
        np.savetxt(path + 'conv_sigma{}_non_irr.txt'.format(j), conv_param_sigma_plot)
        np.savetxt(path + 'Teff_plot{}_non_irr.txt'.format(j), Teff_plot)
        # np.savetxt(path + 'Tirr_plot{}_non_irr.txt'.format(j), Tirr_plot)
        # np.savetxt(path + 'Tvis_plot{}_non_irr.txt'.format(j), Tvis_plot)
        np.savetxt(path + 'T_C_plot{}_non_irr.txt'.format(j), T_C_plot)
        np.savetxt(path + 'Sigma0_plot{}_non_irr.txt'.format(j), Sigma0_plot)
        np.savetxt(path + 'r.txt', r_plot)

        nu_R = 4.56e14
        nu_J = 2.46e14
        nu_U = 8.22e14
        nu_B = 6.74e14
        nu_V = 5.44e14
        nu_UVW1 = 11.5e14
        nu_UVW2 = 15.7e14
        nu_UVM2 = 13.3e14
        nu_arr = np.array([nu_U, nu_B, nu_V, nu_R, nu_J, nu_UVW1, nu_UVW2, nu_UVM2])
        integral_arr = np.empty(8)

        for i, nu in enumerate(nu_arr):
            integral_arr[i] = simps(plank(nu, Teff_plot) * r_plot, r_plot)

        nuFnu_arr = 2 * np.pi * np.cos(np.radians(incl)) * nu_arr * integral_arr / d ** 2
        nuFnu_plot[j] = nuFnu_arr

    np.savetxt(path + 'Mdot_plot_non_irr.txt', Mdot_plot)
    np.savetxt(path + 'nuFnu_non_irr.txt', nuFnu_plot)
    np.savetxt(path + 't.txt', time)

    print(datetime.now() - time0)


if __name__ == '__main__':
    main()
