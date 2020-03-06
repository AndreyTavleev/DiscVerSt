import numpy as np
from scipy.integrate import simps
from astropy import constants as const
from matplotlib import rcParams
from astropy.io import ascii
import mesa_vs

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


def plank(nu, T):
    a = 2 * pl_const * nu ** 3 / c ** 2
    return a / np.expm1(pl_const * nu / (k_B * T))


def main():
    data = ascii.read('F_xspec.dat')
    t = list(data['col1'])
    f = list(data['col3'])
    N = 100
    incl = 40
    d = 5 * 3 * 1e21
    M = 1.5 * M_sun
    eta = 0.1
    alpha = 0.3

    jj = 0
    for flux in f:
        print('j = ', jj)
        jj = jj + 1
        Mdot = 4 * np.pi * d ** 2 * flux / (eta * c ** 2)
        Teff_plot = []
        r_plot = np.geomspace(0.0001 * R_sun, 1.87 * R_sun, N)

        for i, r in enumerate(r_plot):
            print('i =', i)
            h = (G * M * r) ** (1 / 2)
            vs = mesa_vs.MesaVerticalStructureRadConv(M, alpha, r, Mdot * h, irr_heat=True)
            vs.fit()
            Teff_plot.append(vs.Teff)

        nuFnu_plot = []
        nu_R = 4.56e14
        nu_J = 2.46e14
        nu_U = 8.22e14
        nu_B = 6.74e14
        nu_V = 5.44e14
        nu_UVW1 = 11.5e14
        nu_UVW2 = 15.7e14
        nu_UVM2 = 13.3e14
        nu = nu_R
        f_var = []
        for j, r in enumerate(r_plot):
            f_var.append(plank(nu, Teff_plot[j]) * r)

        integral = simps(f_var, r_plot)

        nuFnu = 2 * np.pi * np.cos(np.radians(incl)) * nu * integral / d ** 2
        nuFnu_plot.append(nuFnu)
    np.savetxt('fig/nuFnu_R.txt', nuFnu_plot)
    np.savetxt('fig/t.txt', t)


if __name__ == '__main__':
    main()
