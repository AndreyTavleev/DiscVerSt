from enum import IntEnum

from vs import IdealKramersVerticalStructure, IdealBellLin1994VerticalStructure
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
from astropy import constants as cnst


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


def S_curve(Teff_min, Teff_max, M, alpha, r, input='Teff', output='Teff'):
    Sigma_plot = []
    Plot = []

    h = (G * M * r) ** (1 / 2)

    for i, Teff in enumerate(np.r_[Teff_max:Teff_min:100j]):
        F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Teff ** 4
        Mdot = F / h
        vs = IdealKramersVerticalStructure(M, alpha, r, F)
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
    # plt.grid(True, which='both', ls='-')
    # plt.title('S-curve')
    # plt.tight_layout()
    # plt.savefig('fig/S-curve.pdf')


def main():
    M = 6 * cnst.M_sun.cgs.value
    r = 8e10
    alpha = 0.5
    S_curve(2.3e3, 1e4, M, alpha, r, 'Teff')

    # for r in [5.5e10, 6.75e10, 8e10]:
    #     S_curve(2.3e3, 1e4, M, alpha, r, 1)

    # plt.hlines(1e17, *plt.xlim(),linestyles='--')
    plt.grid(True, which='both', ls='-')
    plt.legend()
    plt.title('S-curve')
    plt.show()


if __name__ == '__main__':
    main()
