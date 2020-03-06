import numpy as np
from scipy.integrate import simps
from astropy import constants as const
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
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
    f = list(data['col3'])
    N = 100
    incl = 40
    d = 5 * 3 * 1e21
    M = 1.5 * M_sun
    eta = 0.1
    alpha = 0.3
    maximum = max(f)
    for i, el in enumerate(f):
        if el == maximum:
            numer = i
            break

    flux = f[numer]
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
    nu_plot = np.geomspace(1e13, 2.5e18, 100)
    for i, nu in enumerate(nu_plot):
        print('ii =', i)

        f_var = []
        for j, r in enumerate(r_plot):
            f_var.append(plank(nu, Teff_plot[j]) * r)

        integral = simps(f_var, r_plot)

        nuFnu = 2 * np.pi * np.cos(np.radians(incl)) * nu * integral / d ** 2
        nuFnu_plot.append(nuFnu)

    plt.plot(nu_plot, nuFnu_plot, label=r'Mdot = {:g}'.format(Mdot))
    np.savetxt('fig/nu.txt', nu_plot)
    np.savetxt('fig/nuFnu.txt', nuFnu_plot)

    colours = list(mcolors.TABLEAU_COLORS) + list(np.random.rand(20, 4))
    plt.axvline(x=4.56e14, label='R', c=colours[2])
    plt.axvline(x=2.46e14, label='J', c=colours[3])
    plt.axvline(x=8.22e14, label='U', c=colours[4])
    plt.axvline(x=6.74e14, label='B', c=colours[5])
    plt.axvline(x=5.44e14, label='V', c=colours[6])
    plt.axvline(x=11.5e14, label='UVW1', c=colours[7])
    plt.axvline(x=15.7e14, label='UVW2', c=colours[8])
    plt.axvline(x=13.3e14, label='UVM2', c=colours[9])
    plt.xlabel(r'$\nu$')
    plt.ylabel(r'$\nu F\nu$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig('fig/nuFnu.pdf')


if __name__ == '__main__':
    main()
