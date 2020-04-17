from plots import S_curve
from astropy import constants as const
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np

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


def main():
    M = 8.3e6 * M_sun
    # rg = 2 * G * M / c ** 2
    # r = 300 * rg
    # r = 1.86 * R_sun
    r = 2e14
    alpha = 0.02
    # S_curve(1e3, 1e4, M, alpha, r, structure='MesaRadConv', n=100, input='Teff', output='z0r', save=False, set_title=False)
    # alpha = 0.01
    # S_curve(2e3, 2e4, M, alpha=0.02, r=r, structure='MesaRadConvIrr', n=300, input='Teff', output='Teff',
    #         xscale='log', yscale='log', save=False, set_title=False)
    S_curve(1e15, 8e23, M, alpha=alpha, r=r, structure='MesaRadConv', n=100, input='Mdot', output='Mdot',
            xscale='parlog', yscale='parlog', save=False, set_title=True,
            title=r'M = {:g} Msun, r = {:g} cm, alpha = {:g}'.format(M / M_sun, r, alpha))

    Pi1 = 7
    Pi2 = 0.5
    Pi3 = 1.15
    Pi4 = 0.46
    Ledd = 1.3e38 * (M / M_sun)
    GM = G * M
    h = np.sqrt(GM * r)
    omegaK = np.sqrt(GM / r ** 3)

    xx = np.geomspace(3.8, 7, 100)
    x = np.geomspace(3.8, 6.2, 100)
    f = [np.log10(Pi3 * Pi4 ** 2 / (Pi1 * Pi2 ** 2) * 32 / (9 * np.pi * alpha * omegaK) *
                  (Ledd / GM) ** 2 * 1 / (10 ** sigma)) for sigma in x]

    g = [np.log10((2 * np.pi * alpha * h) / (Pi1 * Pi3) * (1 / 1) ** 2 * 10 ** sigma) for sigma in xx]

    plt.plot(x, f, '--')
    plt.plot(xx, g, '--')

    plt.savefig('fig/S-curve_SMBH.pdf')
    raise Exception

    sigma_down_sol = 74.6 * (alpha / 0.1) ** (-0.83) * (r / 1e10) ** 1.18 * (M / M_sun) ** (-0.4)
    sigma_up_sol = 39.9 * (alpha / 0.1) ** (-0.8) * (r / 1e10) ** 1.11 * (M / M_sun) ** (-0.37)

    mdot_down_sol = 2.64e15 * (alpha / 0.1) ** 0.01 * (r / 1e10) ** 2.58 * (M / M_sun) ** (-0.85)
    mdot_up_sol = 8.07e15 * (alpha / 0.1) ** (-0.01) * (r / 1e10) ** 2.64 * (M / M_sun) ** (-0.89)

    teff_up_sol = 6890 * (r / 1e10) ** (-0.09) * (M / M_sun) ** 0.03
    teff_down_sol = 5210 * (r / 1e10) ** (-0.10) * (M / M_sun) ** 0.04

    sigma_up_he = 589 * (alpha / 0.1) ** (-0.78) * (r / 1e10) ** 1.07 * (M / M_sun) ** (-0.36)
    teff_up_he = 13100 * (alpha / 0.1) ** (-0.01) * (r / 1e10) ** (-0.08) * (M / M_sun) ** 0.03
    mdot_up_he = 1.05e17 * (alpha / 0.1) ** (-0.05) * (r / 1e10) ** 2.69 * (M / M_sun) ** (-0.9)
    sigma_down_he = 1770 * (alpha / 0.1) ** (-0.83) * (r / 1e10) ** 1.2 * (M / M_sun) ** (-0.4)
    teff_down_he = 9700 * (r / 1e10) ** (-0.09) * (M / M_sun) ** 0.03
    mdot_down_he = 3.18e16 * (alpha / 0.1) ** (-0.01) * (r / 1e10) ** 2.65 * (M / M_sun) ** (-0.88)

    sigma_down_sol_ham = 13.4 * alpha ** (-0.83) * (r / 1e10) ** 1.14 * (M / M_sun) ** (-0.38)
    sigma_up_sol_ham = 8.3 * alpha ** (-0.77) * (r / 1e10) ** 1.11 * (M / M_sun) ** (-0.37)
    mdot_down_sol_ham = 4e15 * (alpha / 0.1) ** (-0.004) * (r / 1e10) ** 2.65 * (M / M_sun) ** (-0.88)
    mdot_up_sol_ham = 9.5e15 * (alpha / 0.1) ** 0.01 * (r / 1e10) ** 2.68 * (M / M_sun) ** (-0.89)

    # plt.scatter(sigma_down_sol, mdot_down_sol)
    # plt.scatter(sigma_up_sol, mdot_up_sol)

    # plt.savefig('fig/S-curve_with_dots.pdf')


if __name__ == '__main__':
    main()
