#!/usr/bin/env python3

from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize, root, newton, fsolve
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from astropy import constants as cnst
from enum import IntEnum
from matplotlib import rcParams
try:
    from opacity import Opac
    opacity = Opac(mesa_dir='/mesa')
except ImportError:
    class HasAnyAttr:
        def __getattr__(self, item):
            return None
    opacity = HasAnyAttr()

# rcParams['text.usetex'] = True
# rcParams['font.size'] = 14

sigmaSB = cnst.sigma_sb.cgs.value
R = cnst.R.cgs.value
G = cnst.G.cgs.value


class Vars(IntEnum):
    S = 0
    P = 1
    Q = 2
    T = 3


class BaseVerticalStructure:
    mu = 0.6

    def __init__(self, Mx, alpha, r, F):
        self.Mx = Mx
        self.GM = G * Mx
        self.alpha = alpha
        self.r = r
        self.F = F
        self.omegaK = np.sqrt(self.GM / self.r ** 3)

        self.Q_norm = self.Q0 = (3 / (8 * np.pi)) * F * self.omegaK / self.r ** 2
        self.Teff = (self.Q0 / sigmaSB) ** (1 / 4)
        self.z0 = self.z0_init()

    @property
    def z0(self):
        return self.__z0

    @z0.setter
    def z0(self, z0):
        self.__z0 = z0
        self.P_norm = (4 / 3) * self.Q_norm / (self.alpha * z0 * self.omegaK)
        self.T_norm = self.omegaK ** 2 * self.mu * z0 ** 2 / R
        self.sigma_norm = 16 * self.Q_norm / (3 * self.alpha * z0 ** 2 * self.omegaK ** 3)

    def law_of_viscosity(self, P):
        return self.alpha * P

    def law_of_rho(self, P, T):
        raise NotImplementedError

    def law_of_opacity(self, rho, T):
        raise NotImplementedError

    def viscosity(self, y):
        return self.law_of_viscosity(y[Vars.P] * self.P_norm)

    def rho(self, y):
        return self.law_of_rho(y[Vars.P] * self.P_norm, y[Vars.T] * self.T_norm)

    def opacity(self, y):
        return self.law_of_opacity(self.rho(y), y[Vars.T] * self.T_norm)

    def photospheric_pressure_equation(self, tau, y):
        rho = self.law_of_rho(y, self.Teff * (1 / 2 + 3 * tau / 4) ** (1 / 4))
        xi = self.law_of_opacity(rho, self.Teff * (1 / 2 + 3 * tau / 4) ** (1 / 4))
        return self.z0 * self.omegaK ** 2 / xi

    def initial(self):
        solution = solve_ivp(
            self.photospheric_pressure_equation,
            [0, 2 / 3],
            [1e-8]
        )
        # assert solution.success
        # integral = quad(lambda x: (1 + 3 * x / 2) ** (9 / 8), 0, 2 / 3)[0]
        # A = math.sqrt(omegaK ** 2 * z0 * R * integral * Teff ** (9 / 2) / (xi0_kram * mu * 2 ** (1 / 8))) / P0
        # A = root(lambda P1: 3 * P0 * P1 / 2 - wqe(3 / 2, P0 * P1, z0), [0.5]).x[0]
        y = np.empty(4, dtype=np.float)
        y[Vars.S] = 0
        y[Vars.P] = solution.y[0][-1] / self.P_norm
        y[Vars.Q] = 1
        y[Vars.T] = self.Teff / self.T_norm
        return y

    def dydt(self, t, y):
        dy = np.empty(4)
        rho = self.rho(y)
        w_r_phi = self.viscosity(y)
        xi = self.opacity(y)
        dy[Vars.S] = rho * 2 * self.z0 / self.sigma_norm
        dy[Vars.P] = rho * (1 - t) * self.omegaK ** 2 * self.z0 ** 2 / self.P_norm
        dy[Vars.Q] = -(3 / 2) * self.z0 * self.omegaK * w_r_phi / self.Q_norm
        dy[Vars.T] = ((abs(y[Vars.Q]) / y[Vars.T] ** 3)
                      * 3 * xi * rho * self.z0 * self.Q_norm / (16 * sigmaSB * self.T_norm ** 4))
        return dy

    def integrate(self, t):
        assert t[0] == 0
        solution = solve_ivp(self.dydt, (t[0], t[-1]), self.initial(), t_eval=t)
        # assert solution.success
        return [solution.y, solution.message]

    def y_c(self):
        y = self.integrate([0, 1])
        return y[0][:, -1]

    def tau0(self):
        y_c = self.y_c()
        Sigma0 = y_c[Vars.S] * self.sigma_norm
        T_C = y_c[Vars.T] * self.T_norm
        P_C = y_c[Vars.P] * self.P_norm
        rho_C = self.law_of_rho(P_C, T_C)
        xi_C = self.law_of_opacity(rho_C, T_C)
        return Sigma0 * xi_C / 2

    def Pi_finder(self):
        y_c = self.y_c()
        Sigma0 = y_c[Vars.S] * self.sigma_norm
        T_C = y_c[Vars.T] * self.T_norm
        P_C = y_c[Vars.P] * self.P_norm
        rho_C = self.law_of_rho(P_C, T_C)
        xi_C = self.law_of_opacity(rho_C, T_C)

        Pi_1 = (self.omegaK ** 2 * self.z0 ** 2 * rho_C) / P_C
        Pi_2 = Sigma0 / (2 * self.z0 * rho_C)
        Pi_3 = (3 / 4) * (self.alpha * self.omegaK * P_C * Sigma0) / (self.Q0 * rho_C)
        Pi_4 = (3 / 32) * (self.Teff / T_C) ** 4 * (Sigma0 * xi_C)

        Pi_real = np.array([Pi_1, Pi_2, Pi_3, Pi_4])

        return Pi_real

    def z0_init(self):
        return (self.r * 2.86e-7 * self.F ** (3 / 20) * (self.Mx / cnst.M_sun.cgs.value) ** (-12 / 35)
                * self.alpha ** (-1 / 10) * (self.r / 1e10) ** (1 / 20))

    def fit(self):
        def dq(z0r):
            self.z0 = z0r[0] * self.r
            q_c = self.y_c()[Vars.Q]
            return q_c

        result = fsolve(dq, self.z0 / self.r, full_output=True)
        return result


class IdealGasMixin:
    mu = 0.6

    def law_of_rho(self, P, T):
        return P * self.mu / (R * T)


class MesaGasMixin:
    law_of_rho = opacity.rho


class KramersOpacityMixin:
    xi0 = 5e24
    zeta = 1
    gamma = -7 / 2

    def law_of_opacity(self, rho, T):
        return self.xi0 * (rho ** self.zeta) * (T ** self.gamma)


class BellLin1994TwoComponentOpacityMixin:
    xi0_ff = 1.5e20  # BB AND FF, OPAL
    zeta_ff = 1
    gamma_ff = - 5 / 2
    xi0_h = 1.0e-36  # H-scattering
    zeta_h = 1 / 3
    gamma_h = 10

    def opacity_h(self, rho, T):
        return self.xi0_h * (rho ** self.zeta_h) * (T ** self.gamma_h)

    def opacity_ff(self, rho, T):
        return self.xi0_ff * (rho ** self.zeta_ff) * (T ** self.gamma_ff)

    def law_of_opacity(self, rho, T):
        return np.minimum(self.opacity_h(rho, T), self.opacity_ff(rho, T))


class MesaOpacityMixin:
    law_of_opacity = opacity.kappa


class IdealKramersVerticalStructure(IdealGasMixin, KramersOpacityMixin, BaseVerticalStructure):
    pass


class MesaVerticalStructure(MesaGasMixin, MesaOpacityMixin, BaseVerticalStructure):
    pass


def S_curve(Teff_min, Teff_max, M, alpha, r):
    porridge = []
    eggplant = []
    h = (G * M * r) ** (1 / 2)
    plt.xscale('log')
    plt.yscale('log')
    for i, Teff in enumerate(np.r_[Teff_max:Teff_min:10j]):
        F = 8 * np.pi / 3 * h**7 / (G*M)**4 * sigmaSB * Teff**4
        print(i+1)
        # print(F)
        vs = MesaVerticalStructure(M, alpha, r, F)
        result = vs.fit()
        if not result[2] or abs(result[1]['fvec']) > 1e-3:
            print(result[3])
            print(result[1]['fvec'])
        porridge.append(Teff)
        eggplant.append(vs.y_c()[Vars.S] * vs.sigma_norm)
    plt.plot(eggplant, porridge, 'x', label='F')
    plt.savefig('fig/s-curve.pdf')
    # plt.show()


def main():
    M = 6 * 2e33
    r = 1e11
    alpha = 0.5

    S_curve(2.3e3, 1e4, M, alpha, r)

    # F = 3e34
    # vs = vertical_structure(M, alpha, r, F)
    # print(vs.Pi_finder())


if __name__ == '__main__':
    main()
