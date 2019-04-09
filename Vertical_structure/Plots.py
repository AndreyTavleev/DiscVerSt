from enum import IntEnum

from vs import IdealKramersVerticalStructure, IdealBellLin1994VerticalStructure, MesaVerticalStructure
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
from astropy import constants as cnst
from scipy.interpolate import InterpolatedUnivariateSpline

try:
    from opacity import Opac

    opacity = Opac({b'he4': 0.25, b'h1': 0.75}, mesa_dir='/mesa')
except ImportError:
    class HasAnyAttr:
        def __getattr__(self, item):
            return None


    opacity = HasAnyAttr()

rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['text.latex.preamble'] = [r'\usepackage[utf8]{inputenc}', r'\usepackage{amsfonts, amsmath, amsthm, amssymb}']
# r'\usepackage[english,russian]{babel}'

sigmaSB = cnst.sigma_sb.cgs.value
G = cnst.G.cgs.value
c = cnst.c.cgs.value
M_sun = cnst.M_sun.cgs.value


class Vars(IntEnum):
    S = 0
    P = 1
    Q = 2
    T = 3


def Structure_Plot(M, alpha, r, Par, input='Teff'):
    h = (G * M * r) ** (1 / 2)

    if input == 'F':
        F = Par
    elif input == 'Teff':
        F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Par ** 4
    elif input == 'Mdot':
        F = Par * h
    else:
        raise IOError('Incorrect input, try Teff, Mdot of F')

    vs = MesaVerticalStructure(M, alpha, r, F)
    vs.fit()
    print('Teff = ', vs.Teff)
    t = np.linspace(0, 1, 100)
    S, P, Q, T = vs.integrate(t)[0]
    plt.plot(1 - t, S, label=r'$\Sigma$')
    plt.plot(1 - t, P, label='$P$')
    plt.plot(1 - t, Q, label='$Q$')
    plt.plot(1 - t, T, label='$T$')
    plt.grid()
    plt.legend()
    plt.xlabel('$z / z_0$')
    plt.title('Vertical structure, Teff = %d' % vs.Teff)
    plt.savefig('fig/vs%d.pdf' % vs.Teff)
    plt.close()


def S_curve(Par_min, Par_max, M, alpha, r, n=100, input='Teff', output='Mdot', save=True):
    Sigma_plot = []
    Plot = []

    h = (G * M * r) ** (1 / 2)

    Mdot_edd = 1.7e18 * M / M_sun
    rg = 2 * G * M / c ** 2

    for i, Par in enumerate(np.r_[Par_max:Par_min:complex(0, n)]):

        if input == 'Teff':
            Teff = Par
            F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Par ** 4
            Mdot = F / h
        elif input == 'Mdot':
            Mdot = Par
            F = Par * h
            Teff = (3 / (8 * np.pi) * (G * M) ** 4 * F / (sigmaSB * h ** 7)) ** (1 / 4)
        elif input == 'F':
            F = Par
            Mdot = Par / h
            Teff = (3 / (8 * np.pi) * (G * M) ** 4 * Par / (sigmaSB * h ** 7)) ** (1 / 4)
        else:
            raise IOError('Incorrect input, try Teff, Mdot of F')

        vs = MesaVerticalStructure(M, alpha, r, F)
        vs.fit()
        Sigma_plot.append(vs.y_c()[Vars.S] * vs.sigma_norm)

        if output == 'Teff':
            Plot.append(Teff)
        elif output == 'Mdot':
            Plot.append(Mdot / Mdot_edd)
        elif output == 'F':
            Plot.append(F)
        else:
            raise IOError('Incorrect output, try Teff, Mdot of F')
        print(i + 1)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\Sigma_0, \, g/cm^2$')
    plt.plot(Sigma_plot, Plot, label=r'M = {:g} Msun, r = {:g} rg'.format(M / M_sun, r / rg))
    if output == 'Teff':
        plt.ylabel(r'$T_{\rm eff}, \, K$')
    elif output == 'Mdot':
        plt.ylabel(r'$\dot{M}, \, g/s$')
    elif output == 'F':
        plt.ylabel(r'$F, \, g~cm^2$')
    plt.grid(True, which='both', ls='-')
    plt.legend()
    plt.title('S-curve')
    plt.tight_layout()
    if save:
        plt.savefig('fig/S-curve.pdf')
        plt.close()


def TempGrad_Plot(vs):
    vs.fit()
    n = 1000
    t = np.linspace(0, 1, n)
    y = vs.integrate(t)[0]
    S, P, Q, T = y
    grad_plot = InterpolatedUnivariateSpline(np.log(P), np.log(T)).derivative()
    rho, eos = opacity.rho(P * vs.P_norm, T * vs.T_norm, True)
    ion = np.exp(eos.lnfree_e)
    kappa = vs.opacity(y)
    plt.plot(1 - t, grad_plot(np.log(P)), label=r'$\nabla_{rad}$')
    plt.plot(1 - t, eos.grad_ad, label=r'$\nabla_{ad}$')
    plt.plot(1 - t, T * vs.T_norm / 2e5, label='T / 2e5K')
    plt.plot(1 - t, ion, label='free e')
    plt.plot(1 - t, kappa / kappa[-1], label=r'$\kappa / \kappa_C$')
    plt.legend()
    plt.xlabel('$z / z_0$')
    plt.title(r'$\frac{d(lnT)}{d(lnP)}, T_{\rm eff} = %d$' % vs.Teff)
    plt.hlines(0.4, *plt.xlim(), linestyles='--')
    plt.grid()
    plt.savefig('fig/TempGrad%d.pdf' % vs.Teff)
    plt.close()


def Opacity_Plot(Par_min, Par_max, M, alpha, r, n=100, input='Teff'):
    T_C_plot = []
    Opacity_Plot = []

    h = (G * M * r) ** (1 / 2)

    for i, Par in enumerate(np.r_[Par_max:Par_min:complex(0, n)]):

        if input == 'Teff':
            Teff = Par
            F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Par ** 4
        elif input == 'Mdot':
            Mdot = Par
            F = Par * h
        elif input == 'F':
            F = Par
        else:
            raise IOError('Incorrect input, try Teff, Mdot of F')

        vs = MesaVerticalStructure(M, alpha, r, F)
        vs.fit()
        y = vs.parameters_C()
        T_C_plot.append(y[2])
        Opacity_Plot.append(y[0])
        print(i + 1)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$T_C, K$')
    plt.ylabel(r'$\varkappa_C, cm^2/g$')
    plt.plot(T_C_plot, Opacity_Plot)
    plt.grid(True, which='both', ls='-')
    plt.title('Opacity')
    plt.tight_layout()
    plt.savefig('fig/Opacity.pdf')
    plt.close()


def main():
    c = cnst.c.cgs.value
    M = 6 * M_sun
    rg = 2 * G * M / c ** 2
    alpha = 0.5
    r = 8e10

    A = (1.7e18 * 3 * c ** 6 / (64 * np.pi * G ** 2)) ** (1 / 3) * (M * M_sun * sigmaSB * 1e16) ** (-1 / 3)

    # B = 3 / (64 * np.pi) * 1.7e18 * c ** 6 * 1e-2 / (G ** 2 * sigmaSB * 1e16)
    # A = (B / M) ** (1 / 3)
    # print('{:g}'.format(A*rg))
    # print('{:g}'.format((1.7e18*c**6/G**2)**(1/3) * (M*M_sun)**(-1/3)))
    # print('{:g}'.format(A/r))

    # for M in [6 * M_sun, 1e2 * M_sun, 1e3 * M_sun, 1e4 * M_sun, 1e5 * M_sun, 1e7 * M_sun, 1e8 * M_sun]:
    # for M in [6*M_sun, 60*M_sun, 600 * M_sun]:
    c = cnst.c.cgs.value
    M = 1e8 * M_sun
    rg = 2 * G * M / c ** 2
    for r in [30 * rg, 60 * rg, 100 * rg, 200 * rg, 300 * rg]:
        # c = cnst.c.cgs.value
        # rg = 2 * G * M / c ** 2
        # A = (1.7e18 * 3 * c ** 6 / (64 * np.pi * G ** 2)) ** (1 / 3) * (M * M_sun * sigmaSB * 1e16) ** (-1 / 3)
        # r = A * rg
        print('{:g}'.format(r))
        print('{:g}'.format(M))
        alpha = 0.5
        Mdot_edd = 1.7e18 * M / M_sun
        S_curve(Mdot_edd * 1e-4, 2 * Mdot_edd, M, alpha, r, 1000, input='Mdot', output='Mdot', save=False)

    plt.savefig('fig/S-curve-big1.pdf')
    plt.close()
    #
    #
    # for M in [6000*M_sun, 6e5*M_sun, 6e6 * M_sun]:
    #     c = cnst.c.cgs.value
    #     rg = 2 * G * M / c ** 2
    #     A = (1.7e18 * 3 * c ** 6 / (64 * np.pi * G ** 2)) ** (1 / 3) * (M * M_sun * sigmaSB * 1e16) ** (-1 / 3)
    #     r = A * rg
    #     print('{:g}'.format(r))
    #     print('{:g}'.format(M))
    #     alpha = 0.5
    #     Mdot_edd = 1.7e18 * M / M_sun
    #     S_curve(Mdot_edd * 1e-4, 2 * Mdot_edd, M, alpha, r, input='Mdot', output='Mdot')
    #
    # plt.savefig('fig/S-curve-big2.pdf')
    # plt.close()
    #
    # for M in [6e7*M_sun, 6e8*M_sun, 6e9 * M_sun]:
    #     c = cnst.c.cgs.value
    #     rg = 2 * G * M / c ** 2
    #     A = (1.7e18 * 3 * c ** 6 / (64 * np.pi * G ** 2)) ** (1 / 3) * (M * M_sun * sigmaSB * 1e16) ** (-1 / 3)
    #     r = A * rg
    #     print('{:g}'.format(r))
    #     print('{:g}'.format(M))
    #     alpha = 0.5
    #     Mdot_edd = 1.7e18 * M / M_sun
    #     S_curve(Mdot_edd * 1e-4, 2 * Mdot_edd, M, alpha, r, input='Mdot', output='Mdot')
    #
    # plt.savefig('fig/S-curve-big3.pdf')
    # plt.close()
    #
    #
    #
    #
    # for M in [6 * M_sun, 60 * M_sun, 600 * M_sun]:
    #     c = cnst.c.cgs.value
    #     rg = 2 * G * M / c ** 2
    #     A = (1.7e18 * 3 * c ** 6 / (64 * np.pi * G ** 2)) ** (1 / 3) * (M * M_sun * sigmaSB * 1e16) ** (-1 / 3)
    #     r = A * rg
    #     print('{:g}'.format(r))
    #     print('{:g}'.format(M))
    #     alpha = 0.5
    #     Mdot_edd = 1.7e18 * M / M_sun
    #     S_curve(Mdot_edd * 1e-4, 2 * Mdot_edd, M, alpha, r, input='Mdot', output='Teff')
    #
    # plt.savefig('fig/S-curve-big4.pdf')
    # plt.close()
    #
    # for M in [6000 * M_sun, 6e5 * M_sun, 6e6 * M_sun]:
    #     c = cnst.c.cgs.value
    #     rg = 2 * G * M / c ** 2
    #     A = (1.7e18 * 3 * c ** 6 / (64 * np.pi * G ** 2)) ** (1 / 3) * (M * M_sun * sigmaSB * 1e16) ** (-1 / 3)
    #     r = A * rg
    #     print('{:g}'.format(r))
    #     print('{:g}'.format(M))
    #     alpha = 0.5
    #     Mdot_edd = 1.7e18 * M / M_sun
    #     S_curve(Mdot_edd * 1e-4, 2 * Mdot_edd, M, alpha, r, input='Mdot', output='Teff')
    #
    # plt.savefig('fig/S-curve-big5.pdf')
    # plt.close()
    #
    # for M in [6e7 * M_sun, 6e8 * M_sun, 6e9 * M_sun]:
    #     c = cnst.c.cgs.value
    #     rg = 2 * G * M / c ** 2
    #     A = (1.7e18 * 3 * c ** 6 / (64 * np.pi * G ** 2)) ** (1 / 3) * (M * M_sun * sigmaSB * 1e16) ** (-1 / 3)
    #     r = A * rg
    #     print('{:g}'.format(r))
    #     print('{:g}'.format(M))
    #     alpha = 0.5
    #     Mdot_edd = 1.7e18 * M / M_sun
    #     S_curve(Mdot_edd * 1e-4, 2 * Mdot_edd, M, alpha, r, input='Mdot', output='Teff')
    #
    # plt.savefig('fig/S-curve-big6.pdf')
    # plt.close()

    # Opacity_Plot(1e21, 1e27, M, alpha, r, input='Mdot')

    # plot = []
    # plot_t = []
    # for i, Mdot in enumerate(np.r_[1e21:1e27:200j]):
    #     print(i + 1)
    #     h = (G * M * r) ** (1 / 2)
    #     F = Mdot * h
    #     vs = MesaVerticalStructure(M, alpha, r, F)
    #     vs.fit()
    #     print(vs.parameters_C()[1])
    #     oplya = opacity.kappa(3e-7, vs.parameters_C()[2])
    #     plot.append(oplya)
    #     plot_t.append(vs.parameters_C()[2])
    #
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.plot(plot_t, plot)
    # plt.savefig('fig/oplya.pdf')

    # for Mdot in np.linspace(1e21, 1e27, 20):
    #     Structure_Plot(M, alpha, r, Mdot, input='Mdot')

    # Structure_Plot(M, alpha, r, 2e4, input='Teff')

    # for r in [5.5e10, 6e10, 6.75e10]:
    #     S_curve(2.3e3, 1e4, M, alpha, r, output='Mdot')
    #
    # plt.hlines(1e17, *plt.xlim(), linestyles='--')
    # plt.grid(True, which='both', ls='-')
    # plt.legend()
    # plt.title('S-curve')
    # plt.show()

    # for Teff in np.linspace(2e3, 5e4, 20):
    # for Mdot in np.linspace(1e18, 1e19, 20):
    #     h = (G * M * r) ** (1 / 2)
    #     # F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Teff ** 4
    #     F = Mdot * h
    #     vs = MesaVerticalStructure(M, alpha, r, F)
    #     TempGrad_Plot(vs)
    #     print('Teff = %d' % vs.Teff)
    #     print('tau = %d' % vs.tau0())

    # S_curve(1e17, 1e20, M, alpha, r, 1000, input='Mdot', output='Mdot', save=False)
    #
    # Sigma_up = 589 * (alpha / 0.1) ** (-0.78) * (r / 1e10) ** 1.07 * (M / M_sun) ** (-0.36)
    # Teff_up = 13100 * (alpha / 0.1) ** (-0.01) * (r / 1e10) ** (-0.08) * (M / M_sun) ** 0.03
    # Mdot_up = 1.05e17 * (alpha / 0.1) ** (-0.05) * (r / 1e10) ** 2.69 * (M / M_sun) ** (-0.9)
    # Sigma_down = 1770 * (alpha / 0.1) ** (-0.83) * (r / 1e10) ** 1.2 * (M / M_sun) ** (-0.4)
    # Teff_down = 9700 * (r / 1e10) ** (-0.09) * (M / M_sun) ** 0.03
    # Mdot_down = 3.18e16 * (alpha / 0.1) ** (-0.01) * (r / 1e10) ** 2.65 * (M / M_sun) ** (-0.88)
    #
    # Mdot_down_another = 5.9e16 * (alpha / 0.1) ** (-0.41) * (r / 1e10) ** 2.62 * (M / M_sun) ** (-0.87)
    #
    # print(Sigma_up, Mdot_up)
    # print(Sigma_down, Mdot_down)
    #
    # plt.scatter(Sigma_down, Mdot_down)
    # plt.scatter(Sigma_up, Mdot_up)
    # plt.scatter(Sigma_down, Mdot_down_another, marker='*')
    #
    # plt.savefig('fig/Helium-S-curve.pdf')
    # plt.close()

    # Opacity_Plot(1e3, 3e4, M, alpha, r)

    # Teff = 11077.0
    # h = (G * M * r) ** (1 / 2)
    # F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Teff ** 4
    # vs = MesaVerticalStructure(M, alpha, r, F)
    # TempGrad_Plot(vs)
    # Structure_Plot(M, alpha, r, Teff, input='Teff')


if __name__ == '__main__':
    main()
