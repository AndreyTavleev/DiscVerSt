from vs import IdealKramersVerticalStructure
import numpy as np
from astropy import constants as const
from matplotlib import pyplot as plt
from matplotlib import rcParams

try:
    import mesa_vs
except ImportError:
    mesa_vs = np.nan

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
    M = 10 * M_sun
    alpha = 0.3
    Mdot = 0.336e17
    rg = 2 * G * M / c ** 2
    mu = 0.62
    z0r_plot = []
    z0r2_plot = []
    z0r_teor_plot = []
    z0r_teor2_plot = []
    sigma0_plot = []
    sigma0_teor_plot = []
    sigma0_teor2_plot = []
    rho_c_plot = []
    rho_c_teor_plot = []
    rho_c_teor2_plot = []
    t_c_plot = []
    t_c_teor_plot = []
    t_c_teor2_plot = []
    r_plot = []

    for i, r in enumerate(np.geomspace(2 * rg, 3e4 * rg, 100)):
        print(i)
        h = (G * M * r) ** (1 / 2)
        z0r_teor = 0.020 * (M / M_sun) ** (-3 / 8) * alpha ** (-1 / 10) * (r / 1e10) ** (1 / 8) * (mu / 0.6) ** (
                -3 / 8) * (Mdot / 1e17) ** (3 / 20) * 2.6

        z0r_teor2 = 0.0092 * (M / M_sun) ** (-7 / 20) * alpha ** (-1 / 10) * (r / 1e7) ** (1 / 20) * (mu / 0.6) ** (
                -2 / 5) * (Mdot / 1e17) ** (1 / 5) * 2.6

        sigma0_teor = 33 * (M / M_sun) ** (1 / 4) * alpha ** (-4 / 5) * (r / 1e10) ** (-3 / 4) * (mu / 0.6) ** (
                3 / 4) * (Mdot / 1e17) ** (7 / 10) * 1.03

        sigma0_teor2 = 5.1e3 * (M / M_sun) ** (1 / 5) * alpha ** (-4 / 5) * (r / 1e7) ** (-3 / 5) * (mu / 0.6) ** (
                4 / 5) * (Mdot / 1e17) ** (3 / 5) * 0.96

        rho_c_teor = 8e-8 * (M / M_sun) ** (5 / 8) * alpha ** (-7 / 10) * (r / 1e10) ** (-15 / 8) * (mu / 0.6) ** (
                9 / 8) * (Mdot / 1e17) ** (11 / 20) * 0.76

        rho_c_teor2 = 2.8e-2 * (M / M_sun) ** (11 / 20) * alpha ** (-7 / 10) * (r / 1e7) ** (-33 / 20) * (mu / 0.6) ** (
                6 / 5) * (Mdot / 1e17) ** (2 / 5) * 0.67

        t_c_teor = 4e4 * (M / M_sun) ** (1 / 4) * alpha ** (-1 / 5) * (r / 1e10) ** (-3 / 4) * (mu / 0.6) ** (
                1 / 4) * (Mdot / 1e17) ** (3 / 10) * 1.09

        t_c_teor2 = 8.2e6 * (M / M_sun) ** (3 / 10) * alpha ** (-1 / 5) * (r / 1e7) ** (-9 / 10) * (mu / 0.6) ** (
                1 / 5) * (Mdot / 1e17) ** (2 / 5) * 1.2
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructure(M, alpha, r, Mdot * h)
            z0r, result = vs.fit()
            varkappa_C, rho_C, T_C, P_C, Sigma0 = vs.parameters_C()
            sigma0_plot.append(Sigma0)
            z0r_plot.append(z0r)
            rho_c_plot.append(rho_C)
            t_c_plot.append(T_C)
        vs2 = IdealKramersVerticalStructure(M, alpha, r, Mdot * h, mu=mu)
        z0r2, result2 = vs2.fit()
        sigma0_teor_plot.append(sigma0_teor)
        sigma0_teor2_plot.append(sigma0_teor2)
        z0r2_plot.append(z0r2)
        z0r_teor_plot.append(z0r_teor)
        z0r_teor2_plot.append(z0r_teor2)
        rho_c_teor_plot.append(rho_c_teor)
        rho_c_teor2_plot.append(rho_c_teor2)
        t_c_teor_plot.append(t_c_teor)
        t_c_teor2_plot.append(t_c_teor2)
        r_plot.append(r / rg)

    # np.savetxt('fig/z0r_plot.txt', z0r_plot)
    # np.savetxt('fig/r_plot.txt', r_plot)

    plt.plot(r_plot, z0r_plot, label='Mesa')
    plt.plot(r_plot, z0r2_plot, label='Kramers')
    plt.plot(r_plot, z0r_teor_plot, '--')
    plt.plot(r_plot, z0r_teor2_plot, '-.')
    plt.ylim(ymin=0, ymax=0.02)
    plt.grid()
    plt.xscale('log')
    plt.legend()
    plt.savefig('fig/r-z0r.pdf')
    plt.close()

    plt.plot(r_plot, sigma0_plot)
    plt.plot(r_plot, sigma0_teor_plot, '--')
    plt.plot(r_plot, sigma0_teor2_plot, '-.')
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('fig/r-Sigma0.pdf')
    plt.close()

    plt.plot(r_plot, rho_c_plot)
    plt.plot(r_plot, rho_c_teor_plot, '--')
    plt.plot(r_plot, rho_c_teor2_plot, '-.')
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('fig/r-Rho_c.pdf')
    plt.close()

    plt.plot(r_plot, t_c_plot)
    plt.plot(r_plot, t_c_teor_plot, '--')
    plt.plot(r_plot, t_c_teor2_plot, '-.')
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('fig/r-T.pdf')
    plt.close()


if __name__ == '__main__':
    main()
