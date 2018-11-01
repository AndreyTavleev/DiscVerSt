from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize, root, newton, fsolve
from matplotlib import pyplot as plt
import numpy as np
from astropy import constants as cnst
import math
from enum import IntEnum
from Verstr import FindPi
from matplotlib import rcParams

rcParams['text.usetex'] = True
rcParams['font.size'] = 14

sigmaSB = cnst.sigma_sb.cgs.value
R = cnst.R.cgs.value
G = cnst.G.cgs.value

xi0_kram = 5e24
zeta_kram = 1
gamma_kram = - 7 / 2
xi0_ff = 1.5e20  # BB AND FF, OPAL
zeta_ff = 1
gamma_ff = - 5 / 2
xi0_h = 1.0e-36  # H-scattering
zeta_h = 1 / 3
gamma_h = 10


class Vars(IntEnum):
    S = 0
    P = 1
    Q = 2
    T = 3


class VerticalStructure:
    mu = 0.6

    def __init__(self, Mx, alpha, r, F, z0):
        self.Mx = Mx
        self.GM = G * Mx
        self.alpha = alpha
        self.r = r
        self.F = F
        self.z0 = z0
        self.omegaK = np.sqrt(self.GM / self.r ** 3)

        self.Q_norm = self.Q0 = (3 / 8 * np.pi) * F * self.omegaK / self.r ** 2
        self.P_norm = (4 / 3) * self.Q_norm / (self.alpha * self.z0 * self.omegaK)
        self.T_norm = self.omegaK ** 2 * self.mu * self.z0 ** 2 / R
        self.sigma_norm = 16 * self.Q_norm / (3 * self.alpha * self.z0 ** 2 * self.omegaK ** 3)

        self.Teff = (self.Q0 / sigmaSB) ** (1 / 4)

    def law_of_viscosity(self, P):
        return self.alpha * P

    def law_of_rho(self, P, T):
        return P * self.mu / (R * T)

    @staticmethod
    def opacity_h(rho, T):
        return xi0_h * (rho ** zeta_h) * (T ** gamma_h)

    @staticmethod
    def opacity_ff(rho, T):
        return xi0_ff * (rho ** zeta_ff) * (T ** gamma_ff)

    @staticmethod
    def opacity_kram(rho, T):
        return xi0_kram * (rho ** zeta_kram) * (T ** gamma_kram)

    def law_of_opacity(self, rho, T, kram=False):
        if kram:
            return self.opacity_kram(rho, T)
        if self.opacity_h(rho, T) < self.opacity_ff(rho, T):
            return self.opacity_h(rho, T)
        else:
            return self.opacity_ff(rho, T)

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
        dy[Vars.T] = ((y[Vars.Q] / y[Vars.T] ** 3)
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

    def Pi_finder(self):
        Sigma0 = self.y_c()[Vars.S] * self.sigma_norm
        T_C = self.y_c()[Vars.T] * self.T_norm
        P_C = self.y_c()[Vars.P] * self.P_norm
        rho_C = self.law_of_rho(P_C, T_C)
        xi_C = self.law_of_opacity(rho_C, T_C)

        # tau0 = (Sigma0 * xi_C) / 2
        # Pi_table = FindPi(tau0).getPi()

        Pi_1 = (self.omegaK ** 2 * self.z0 ** 2 * self.mu) / (R * T_C)
        Pi_2 = Sigma0 / (2 * self.z0 * rho_C)
        Pi_3 = (3 / 4) * (self.alpha * self.omegaK * R * T_C * Sigma0) / (self.Q0 * self.mu)
        Pi_4 = (3 / 32) * (self.Teff / T_C) ** 4 * (Sigma0 * xi_C)

        Pi_real = [Pi_1, Pi_2, Pi_3, Pi_4]

        return Pi_real


def vertical_structure(Mx, alpha, r, F):
    z0_init = r * 2.86e-7 * F ** (3 / 20) * (Mx / cnst.M_sun.cgs.value) ** (-12 / 35) * alpha ** (-1 / 10) * (
                r / 1e10) ** (1 / 20)

    def dq(z0r):
        vs = VerticalStructure(Mx, alpha, r, F, z0r * r)
        q_c = vs.y_c()[Vars.Q]
        return q_c

    # result = newton(dq, z0_init / r)

    result = fsolve(dq, z0_init / r, full_output=True)
    print(result[3])
    print(result[1]['fvec'])

    # result = minimize(dq, z0_init / r)
    # print(result.message)
    # print(result.fun)

    # assert result.success
    # assert result.fun < 1e-3

    z0 = result[0] * r
    return VerticalStructure(Mx, alpha, r, F, z0)


def S_curve(F_min, F_max, M, alpha, r):
    porridge = []
    eggplant = []
    i = 0
    h = (G * M * r) ** (1 / 2)
    for F in np.r_[F_max:F_min:100j]:
        i += 1
        print(i)
        print(F)
        vs = vertical_structure(M, alpha, r, F)
        porridge.append(F / h)
        eggplant.append(vs.y_c()[Vars.S] * vs.sigma_norm)
    plt.plot(eggplant, porridge, label='F')
    plt.show()


def main():
    M = 2e33
    r = 1e10
    alpha = 1
    F = 5e33

    vs = vertical_structure(M, alpha, r, F)
    t = np.r_[0:1:101j]  # np.linspace(0, 1, 101)
    y = vs.integrate(t)
    print(y[1])
    plt.plot(1 - t, y[0][Vars.T], label='T')
    plt.plot(1 - t, y[0][Vars.P], label='P')
    plt.plot(1 - t, y[0][Vars.Q], label='Q')
    plt.plot(1 - t, y[0][Vars.S], label='S')
    plt.grid()
    plt.legend()
    plt.show()

    S_curve(4.34e+33, 4.34e+34, M, alpha, r)

    # print(vs.Pi_finder())


if __name__ == '__main__':
    main()
