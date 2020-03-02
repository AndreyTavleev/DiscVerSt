import numpy as np
from astropy import constants as const
from matplotlib import pyplot as plt
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
    f = list(data['col3'])
    N = 100
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
    Teff_plot2 = []
    ratio_plot = []
    z0_plot = []
    z0_plot2 = []
    r_plot = np.geomspace(0.0001 * R_sun, 1.87 * R_sun, N)

    for i, r in enumerate(r_plot):
        print('i =', i)
        h = (G * M * r) ** (1 / 2)
        vs = mesa_vs.MesaVerticalStructureRadConv(M, alpha, r, Mdot * h, irr_heat=True)
        z0r, result = vs.fit()
        Teff_plot.append(vs.Teff)
        z0_plot.append(z0r)
        ratio_plot.append(vs.Q_irr / vs.Q0)

    np.savetxt('fig/z0r.txt', z0_plot)
    np.savetxt('fig/r.txt', r_plot)
    np.savetxt('fig/Teff.txt', Teff_plot)
    np.savetxt('fig/ratio.txt', ratio_plot)

    for i, r in enumerate(r_plot):
        z0_plot2.append(z0_plot[0] * (r / r_plot[0]) ** (1 / 8))
        Teff_plot2.append(Teff_plot[0] * (r_plot[0] / r) ** (3 / 5))

    plt.plot(r_plot / R_sun, z0_plot2, '--', label=r'$\sim r^{1/8}$')
    plt.plot(r_plot / R_sun, z0_plot)
    plt.legend()
    plt.xlabel('$r/R_{sun}$')
    plt.ylabel('$z_0/r$')
    plt.xscale('log')
    plt.grid()
    plt.savefig('fig/Aql_X-1-z0r.pdf')
    plt.close()

    plt.plot(r_plot / R_sun, Teff_plot2, '--', label=r'$\sim r^{-3/5}$')
    plt.plot(r_plot / R_sun, Teff_plot)
    plt.legend()
    plt.xlabel('$r/R_{sun}$')
    plt.ylabel(r'$T_{\rm eff}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.savefig('fig/Aql_X-1-Teff-r.pdf')
    plt.close()

    plt.plot(r_plot / R_sun, ratio_plot)
    plt.xlabel('$r/R_{sun}$')
    plt.ylabel(r'$Q_{\rm irr}/Q_0$')
    plt.xscale('log')
    plt.grid()
    plt.savefig('fig/Aql_X-1-ratio-r.pdf')
    plt.close()


if __name__ == '__main__':
    main()
