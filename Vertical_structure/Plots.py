from enum import IntEnum

from vs import IdealKramersVerticalStructure, IdealBellLin1994VerticalStructure, MesaVerticalStructure
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
from astropy import constants as cnst
from scipy.interpolate import InterpolatedUnivariateSpline

try:
    from opacity import Opac

    opacity = Opac(mesa_dir='/mesa')
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
R = cnst.R.cgs.value
G = cnst.G.cgs.value


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
    t = np.linspace(0, 1, 100)
    S, P, Q, T = vs.integrate(t)[0]
    plt.plot(1 - t, S, label=r'$\Sigma$')
    plt.plot(1 - t, P, label='$P$')
    plt.plot(1 - t, Q, label='$Q$')
    plt.plot(1 - t, T, label='$T$')
    plt.grid()
    plt.legend()
    plt.xlabel('$z / z_0$')
    plt.title('Vertical structure')
    plt.savefig('fig/vs.pdf')
    plt.close()


def S_curve(Par_min, Par_max, M, alpha, r, input='Teff', output='Teff'):
    Sigma_plot = []
    Plot = []

    h = (G * M * r) ** (1 / 2)

    for i, Par in enumerate(np.r_[Par_max:Par_min:100j]):

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
            Plot.append(Mdot)
        elif output == 'F':
            Plot.append(F)
        else:
            raise IOError('Incorrect output, try Teff, Mdot of F')
        print(i + 1)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\Sigma_0, \, g/cm^2$')
    plt.plot(Sigma_plot, Plot, label=r'r = {:g} cm'.format(r))
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


def main():
    M = 6 * cnst.M_sun.cgs.value
    r = 8e10
    alpha = 0.5

    Structure_Plot(M, alpha, r, 1e17, input='Mdot')
    # S_curve(2e3, 1e4, M, alpha, r, input='Teff', output='Mdot')

    # for r in [5.5e10, 6e10, 6.75e10]:
    #     S_curve(2.3e3, 1e4, M, alpha, r, output='Mdot')
    #
    # plt.hlines(1e17, *plt.xlim(), linestyles='--')
    # plt.grid(True, which='both', ls='-')
    # plt.legend()
    # plt.title('S-curve')
    # plt.show()

    # for Teff in np.linspace(2.3e3, 6e4, 5):
    #     h = (G * M * r) ** (1 / 2)
    #     F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Teff ** 4
    #     # print(F / h)
    #     vs = MesaVerticalStructure(M, alpha, r, F)
    #     TempGrad_Plot(vs)
    #     print('Teff = %d' % vs.Teff)
    #     print('tau = %d' % vs.tau0())


if __name__ == '__main__':
    main()
