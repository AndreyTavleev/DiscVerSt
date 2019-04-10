#!/usr/bin/env python3

from enum import IntEnum

from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import numpy as np
from astropy import constants as cnst

try:
    from opacity import Opac

    opacity = Opac({b'he4': 1.0}, mesa_dir='/mesa')
except ImportError:
    class HasAnyAttr:
        def __getattr__(self, item):
            return None


    opacity = HasAnyAttr()

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

    def __init__(self, Mx, alpha, r, F, eps=1e-4):
        self.Mx = Mx
        self.GM = G * Mx
        self.alpha = alpha
        self.r = r
        self.F = F
        self.omegaK = np.sqrt(self.GM / self.r ** 3)

        self.Q_norm = self.Q0 = (3 / (8 * np.pi)) * F * self.omegaK / self.r ** 2
        self.Teff = (self.Q0 / sigmaSB) ** (1 / 4)
        self.z0 = self.z0_init()

        self.eps = eps

    @property
    def z0(self):
        return self.__z0

    @z0.setter
    def z0(self, z0):
        self.__z0 = z0
        self.P_norm = (4 / 3) * self.Q_norm / (self.alpha * z0 * self.omegaK)
        self.T_norm = self.omegaK ** 2 * self.mu * z0 ** 2 / R
        self.sigma_norm = 28 * self.Q_norm / (3 * self.alpha * z0 ** 2 * self.omegaK ** 3)

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
            [1e-8 * self.P_norm], rtol=self.eps
        )
        # assert solution.success
        # integral = quad(lambda x: (1 + 3 * x / 2) ** (9 / 8), 0, 2 / 3)[0]
        # A = np.sqrt(self.omegaK ** 2 * self.z0 * R * integral * self.Teff ** (9 / 2) / (5e24 * self.mu * 2 ** (1 / 8)))
        # A = root(lambda P1: 3 * P0 * P1 / 2 - wqe(3 / 2, P0 * P1, z0), [0.5]).x[0]
        y = np.empty(4, dtype=np.float)
        y[Vars.S] = 0
        y[Vars.P] = solution.y[0][-1] / self.P_norm
        y[Vars.Q] = 1
        y[Vars.T] = self.Teff / self.T_norm
        return y

    def dlnTdlnP(self, y, t):
        raise NotImplementedError

    def dydt(self, t, y):
        dy = np.empty(4)
        rho = self.rho(y)
        xi = self.opacity(y)
        w_r_phi = self.viscosity(y)
        dy[Vars.S] = rho * 2 * self.z0 / self.sigma_norm
        dy[Vars.P] = rho * (1 - t) * self.omegaK ** 2 * self.z0 ** 2 / self.P_norm
        dy[Vars.Q] = -(3 / 2) * self.z0 * self.omegaK * w_r_phi / self.Q_norm
        # Coef = self.dlnTdlnP(y, t)
        # dy[Vars.T] = Coef * dy[Vars.P] * y[Vars.T] / y[Vars.P]
        dy[Vars.T] = ((abs(y[Vars.Q]) / y[Vars.T] ** 3)
                      * 3 * xi * rho * self.z0 * self.Q_norm / (16 * sigmaSB * self.T_norm ** 4))

        # dTdz_Rad = ((abs(y[Vars.Q]) / y[Vars.T] ** 3)
        #             * 3 * xi * rho * self.z0 * self.Q_norm / (16 * sigmaSB * self.T_norm ** 4))
        # rho_ad, eos = opacity.rho(y[Vars.P] * self.P_norm, y[Vars.T] * self.T_norm, True)
        #
        # if y[Vars.P] / y[Vars.T] * dTdz_Rad / dy[Vars.P] < eos.grad_ad:
        #     dy[Vars.T] = y[Vars.P] / y[Vars.T] * dTdz_Rad / dy[Vars.P]
        # else:
        #     dy[Vars.T] = eos.grad_ad * dy[Vars.P] * y[Vars.T] / y[Vars.P]
        return dy

    def integrate(self, t):
        assert t[0] == 0
        solution = solve_ivp(self.dydt, (t[0], t[-1]), self.initial(), t_eval=t, rtol=self.eps)
        # assert solution.success
        return [solution.y, solution.message]

    def y_c(self):
        y = self.integrate([0, 1])
        return y[0][:, -1]

    def parameters_C(self):
        y_c = self.y_c()
        Sigma0 = y_c[Vars.S] * self.sigma_norm
        T_C = y_c[Vars.T] * self.T_norm
        P_C = y_c[Vars.P] * self.P_norm
        rho_C = self.law_of_rho(P_C, T_C)
        varkappa_C = self.opacity(y_c)
        return varkappa_C, rho_C, T_C, P_C, Sigma0

    def tau0(self):
        y = self.parameters_C()
        Sigma0 = y[4]
        varkappa_C = y[0]
        return Sigma0 * varkappa_C / 2

    def Pi_finder(self):
        varkappa_C, rho_C, T_C, P_C, Sigma0 = self.parameters_C()

        Pi_1 = (self.omegaK ** 2 * self.z0 ** 2 * rho_C) / P_C
        Pi_2 = Sigma0 / (2 * self.z0 * rho_C)
        Pi_3 = (3 / 4) * (self.alpha * self.omegaK * P_C * Sigma0) / (self.Q0 * rho_C)
        Pi_4 = (3 / 32) * (self.Teff / T_C) ** 4 * (Sigma0 * varkappa_C)

        Pi_real = np.array([Pi_1, Pi_2, Pi_3, Pi_4])

        return Pi_real

    def z0_init(self):
        return (self.r * 2.86e-7 * self.F ** (3 / 20) * (self.Mx / cnst.M_sun.cgs.value) ** (-12 / 35)
                * self.alpha ** (-1 / 10) * (self.r / 1e10) ** (1 / 20))

    def fit(self):
        def dq(z0r):
            self.z0 = z0r * self.r
            q_c = self.y_c()[Vars.Q]
            return q_c

        z0r = self.z0 / self.r
        sign_dq = dq(z0r)
        if sign_dq > 0:
            factor = 2
        else:
            factor = 0.5

        while True:
            z0r *= factor
            if sign_dq * dq(z0r) < 0:
                break

        z0r, result = brentq(dq, z0r, z0r / factor, full_output=True)
        return z0r, result


class RadiativeTempGradient:
    def dlnTdlnP(self, y, t):
        rho = self.rho(y)
        xi = self.opacity(y)
        dTdz_Rad = ((abs(y[Vars.Q]) / y[Vars.T] ** 3)
                    * 3 * xi * rho * self.z0 * self.Q_norm / (16 * sigmaSB * self.T_norm ** 4))
        dPdz = rho * (1 - t) * self.omegaK ** 2 * self.z0 ** 2 / self.P_norm
        return y[Vars.P] / y[Vars.T] * dTdz_Rad / dPdz


class AdiabaticTempGradient:
    def dlnTdlnP(self, y, t):
        rho_ad, eos = opacity.rho(y[Vars.P] * self.P_norm, y[Vars.T] * self.T_norm, True)
        return eos.grad_ad


class FirstAssumptionRadiativeConvectiveGradient:
    def dlnTdlnP(self, y, t):
        rho = self.rho(y)
        xi = self.opacity(y)
        dTdz_Rad = ((abs(y[Vars.Q]) / y[Vars.T] ** 3)
                    * 3 * xi * rho * self.z0 * self.Q_norm / (16 * sigmaSB * self.T_norm ** 4))
        dPdz = rho * (1 - t) * self.omegaK ** 2 * self.z0 ** 2 / self.P_norm

        rho_ad, eos = opacity.rho(y[Vars.P] * self.P_norm, y[Vars.T] * self.T_norm, True)

        if y[Vars.P] / y[Vars.T] * dTdz_Rad / dPdz < eos.grad_ad:
            return y[Vars.P] / y[Vars.T] * dTdz_Rad / dPdz
        else:
            return eos.grad_ad


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


class IdealKramersVerticalStructure(IdealGasMixin, KramersOpacityMixin, RadiativeTempGradient, BaseVerticalStructure):
    pass


class MesaVerticalStructure(MesaGasMixin, MesaOpacityMixin, RadiativeTempGradient, BaseVerticalStructure):
    pass


class IdealBellLin1994VerticalStructure(IdealGasMixin, BellLin1994TwoComponentOpacityMixin, RadiativeTempGradient,
                                        BaseVerticalStructure):
    pass


def main():
    M = 6 * cnst.M_sun.cgs.value
    r = 8e10
    alpha = 0.5

    for Teff in np.linspace(2.3e4, 6e4, 5):
        h = (G * M * r) ** (1 / 2)
        F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Teff ** 4
        print('Mdot = %d' % F / h)
        vs = MesaVerticalStructure(M, alpha, r, F)
        print('Teff = %d' % vs.Teff)
        print('tau = %d' % vs.tau0())


if __name__ == '__main__':
    main()
