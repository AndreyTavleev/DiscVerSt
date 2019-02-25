#!/usr/bin/env python3

from enum import IntEnum

from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
from astropy import constants as cnst

try:
    from opacity import Opac
    opacity = Opac(mesa_dir='/mesa')
except ImportError:
    class HasAnyAttr:
        def __getattr__(self, item):
            return None
    opacity = HasAnyAttr()

# matplotlib.use("Agg")

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

    def parameters_C(self):
        y_c = self.y_c()
        Sigma0 = y_c[Vars.S] * self.sigma_norm
        T_C = y_c[Vars.T] * self.T_norm
        P_C = y_c[Vars.P] * self.P_norm
        rho_C = self.law_of_rho(P_C, T_C)
        varkappa_C = self.law_of_opacity(rho_C, T_C)
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

    def plot(self):
        self.fit()
        t = np.linspace(0, 1, 100)
        S, P, Q, T = self.integrate(t)[0]
        plt.plot(1 - t, S, label=r'$\Sigma$')
        plt.plot(1 - t, P, label='$P$')
        plt.plot(1 - t, Q, label='$Q$')
        plt.plot(1 - t, T, label='$T$')
        plt.grid()
        plt.legend()
        plt.xlabel('$z / z_0$')
        plt.title('Vertical structure')
        plt.savefig('fig/vs.pdf')

    def TempGrad(self):
        self.fit()
        n = 1000
        t = np.linspace(0, 1, n)
        y = self.integrate(t)[0]
        S, P, Q, T = y
        Grad_plot = CubicSpline(np.log(P), np.log(T)).derivative()(np.log(P))
        rho, eos = opacity.rho(P * self.P_norm, T * self.T_norm, True)
        ion = np.exp(eos.lnfree_e)
        kappa = self.opacity(y)
        plt.plot(1 - t, Grad_plot, label=r'$\nabla_{rad}$')
        plt.plot(1 - t, eos.grad_ad, label=r'$\nabla_{ad}$')
        plt.plot(1 - t, T * self.T_norm / 1e5, label='T / 1e5K')
        plt.plot(1 - t, ion, label='free e')
        plt.plot(1 - t, kappa / kappa[-1], label=r'$\kappa / \kappa_C$')
        plt.legend()
        plt.xlabel('$z / z_0$')
        plt.title(r'$\frac{d(lnT)}{d(lnP)}, \text{Teff} = %d$' % self.Teff)
        plt.hlines(0.4, *plt.xlim(), linestyles='--')
        plt.grid()
        plt.savefig('fig/TempGrad%d.pdf' % self.Teff)
        plt.close()


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


class IdealBellLin1994VerticalStructure(IdealGasMixin, BellLin1994TwoComponentOpacityMixin, BaseVerticalStructure):
    pass


def S_curve(Teff_min, Teff_max, M, alpha, r):
    porridge = []
    eggplant = []
    porridge_except = []
    eggplant_except = []
    porridge_except2 = []
    eggplant_except2 = []

    Mdot_graph = []

    x_graph = []
    x_graph_except = []
    x_graph_except2 = []

    graph_opacity = []
    graph_opacity_except = []
    graph_opacity_except2 = []

    graph_rho_C = []
    graph_rho_C_except = []
    graph_rho_C_except2 = []

    graph_T_C = []
    graph_T_C_except = []
    graph_T_C_except2 = []

    graph_P_C = []
    graph_P_C_except = []
    graph_P_C_except2 = []

    h = (G * M * r) ** (1 / 2)

    T_C_1 = 0
    T_C_2 = 0

    q = 1

    Sigma_1 = np.infty
    Sigma_2 = 0

    Teff_1 = 0
    Teff_2 = 0

    for i, Teff in enumerate(np.r_[Teff_max:Teff_min:100j]):
        F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Teff ** 4
        Mdot = F / h
        print(i + 1)
        # print(F)
        vs = MesaVerticalStructure(M, alpha, r, F)
        vs.fit()

        # Sigma_2 = vs.y_c()[Vars.S] * vs.sigma_norm
        #
        # if q == 1 and Sigma_2 > Sigma_1:
        #     q += 1
        #     Teff_1 = Teff
        #     T_C_1 = vs.parameters_C()[2]
        #
        # if q == 2 and Sigma_2 < Sigma_1:
        #     q += 1
        #     Teff_2 = Teff
        #     T_C_2 = vs.parameters_C()[2]
        #
        # Sigma_1 = Sigma_2

        porridge.append(Teff)
        Mdot_graph.append(Mdot)
        eggplant.append(vs.y_c()[Vars.S] * vs.sigma_norm)
        par = vs.parameters_C()
        graph_opacity.append(par[0])
        graph_rho_C.append(par[1])
        graph_T_C.append(par[2])
        graph_P_C.append(par[3])
        x_graph.append(i)

        Teff_1 = 5.4e3
        # Teff_1 = 5.5e3
        Teff_2 = 4.2e3
        # Teff_2 = 4.7e3

        # if Teff > 5.4e3:
        #     porridge.append(Teff)
        #     eggplant.append(vs.y_c()[Vars.S] * vs.sigma_norm)
        #     graph_opacity.append(vs.parameters_C()[0])
        #     graph_rho_C.append(vs.parameters_C()[1])
        #     graph_T_C.append(vs.parameters_C()[2])
        #     graph_P_C.append(vs.parameters_C()[3])
        #     x_graph.append(i)
        #
        # elif Teff >= 4.2e3 and Teff <= 5.4e3:
        #     porridge_except.append(Teff)
        #     eggplant_except.append(vs.y_c()[Vars.S] * vs.sigma_norm)
        #     graph_opacity_except.append(vs.parameters_C()[0])
        #     graph_rho_C_except.append(vs.parameters_C()[1])
        #     graph_T_C_except.append(vs.parameters_C()[2])
        #     graph_P_C_except.append(vs.parameters_C()[3])
        #     x_graph_except.append(i)
        #
        # else:
        #     porridge_except2.append(Teff)
        #     eggplant_except2.append(vs.y_c()[Vars.S] * vs.sigma_norm)
        #     graph_opacity_except2.append(vs.parameters_C()[0])
        #     graph_rho_C_except2.append(vs.parameters_C()[1])
        #     graph_T_C_except2.append(vs.parameters_C()[2])
        #     graph_P_C_except2.append(vs.parameters_C()[3])
        #     x_graph_except2.append(i)

    # # print(Teff_1, Teff_2, T_C_1, T_C_2)
    # porridge = np.array(porridge)
    # supper = np.ma.masked_where(porridge <= Teff_1, Mdot_graph)
    # slower = np.ma.masked_where(porridge >= Teff_2, Mdot_graph)
    # smiddle = np.ma.masked_where(np.logical_or(porridge <= Teff_2, porridge >= Teff_1), Mdot_graph)
    # fig, ax = plt.subplots()
    # # fig.set_size_inches(6.4, 9.6)
    #
    # ax.plot(eggplant, supper, 'g', eggplant, slower, 'b', eggplant, smiddle, 'c--')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_xlabel(r'$\Sigma_0, g/cm^2$')
    # # ax.set_ylabel(r'$T_{\rm eff}, K$')
    # ax.set_ylabel(r'$M$')
    # ax.set_title('S-curve')
    # ax.grid(True, which='both', ls='-')
    # # plt.plot(eggplant, porridge, 'g-')
    # # plt.plot(eggplant_except, porridge_except, 'c--')
    # # plt.plot(eggplant_except2, porridge_except2, 'b-')
    # plt.tight_layout()
    # plt.savefig('fig/s-curve.pdf')
    # plt.close()
    #
    # graph_opacity = np.array(graph_opacity)
    # graph_T_C = np.array(graph_T_C)
    # # supper = np.ma.masked_where(graph_T_C < T_C_1, graph_opacity)
    # # slower = np.ma.masked_where(graph_T_C > T_C_2, graph_opacity)
    # # smiddle = np.ma.masked_where(np.logical_or(graph_T_C <= T_C_2, graph_T_C >= T_C_1), graph_opacity)
    # supper = np.ma.masked_where(porridge <= Teff_1, graph_opacity)
    # slower = np.ma.masked_where(porridge >= Teff_2, graph_opacity)
    # smiddle = np.ma.masked_where(np.logical_or(porridge <= Teff_2, porridge >= Teff_1), graph_opacity)
    # fig, ax = plt.subplots()
    # ax.plot(graph_T_C, supper, 'g', graph_T_C, slower, 'b', graph_T_C, smiddle, 'c--')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_xlabel('$T_C, K$')
    # ax.set_ylabel(r'$\varkappa, cm^2/g$')
    # ax.set_title('MESA opacity')
    # ax.grid(True, which='both', ls='-')
    # # plt.plot(graph_T_C, graph_opacity, 'g-')
    # # plt.plot(graph_T_C_except, graph_opacity_except, 'c--')
    # # plt.plot(graph_T_C_except2, graph_opacity_except2, 'b-')
    # # plt.axvspan(T_C_1, T_C_2, color='grey', alpha=0.5)
    # plt.savefig('fig/T_C-opacity.pdf')
    # plt.close()
    #
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel(r'$\rho_C$')
    # plt.ylabel('Opacity')
    # plt.grid()
    # plt.plot(graph_rho_C, graph_opacity, 'bo')
    # plt.plot(graph_rho_C_except, graph_opacity_except, 'mo')
    # plt.plot(graph_rho_C_except2, graph_opacity_except2, 'bo')
    # plt.savefig('fig/rho_C-opacity.pdf')
    # plt.close()
    #
    # plt.plot(x_graph, graph_rho_C, 'bo', label='$rho_C$')
    # plt.plot(x_graph_except, graph_rho_C_except, 'mo', label='$rho_C$')
    # plt.plot(x_graph_except2, graph_rho_C_except2, 'bo', label='$rho_C$')
    # plt.legend()
    # plt.savefig('fig/rho_C.pdf')
    # plt.close()
    #
    # plt.plot(x_graph, graph_T_C, 'bo', label='$T_C$')
    # plt.plot(x_graph_except, graph_T_C_except, 'mo', label='$T_C$')
    # plt.plot(x_graph_except2, graph_T_C_except2, 'bo', label='$T_C$')
    # plt.legend()
    # plt.savefig('fig/T_C.pdf')
    # plt.close()
    #
    # plt.plot(x_graph, graph_P_C, 'bo', label='$P_C$')
    # plt.plot(x_graph_except, graph_P_C_except, 'mo', label='$P_C$')
    # plt.plot(x_graph_except2, graph_P_C_except2, 'bo', label='$P_C$')
    # plt.legend()
    # plt.savefig('fig/P_C.pdf')
    # plt.close()
    #
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Sigma0')
    # plt.ylabel('Opacity')
    # plt.grid()
    # plt.plot(eggplant, graph_opacity, 'bo')
    # plt.plot(eggplant_except, graph_opacity_except, 'mo')
    # plt.plot(eggplant_except2, graph_opacity_except2, 'bo')
    # plt.savefig('fig/sigma-opacity.pdf')
    # plt.close()

    # plt.show()

    # fig, axs = plt.subplots(1, 3, sharey=True)

    # plt.grid(True)
    plt.plot(eggplant, Mdot_graph, label=r'{:g} cm'.format(r))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\Sigma_0, g/cm^2$')
    # ax.set_ylabel(r'$T_{\rm eff}, K$')
    plt.ylabel(r'$\dot{M}, g/s$')
    # ax.set_title('S-curve')
    # plt.plot(eggplant, porridge, 'g-')
    # plt.plot(eggplant_except, porridge_except, 'c--')
    # plt.plot(eggplant_except2, porridge_except2, 'b-')
    # plt.tight_layout()


def main():
    M = 6 * cnst.M_sun.cgs.value
    r = 8e10
    alpha = 0.5

    # for r in [5.5e10, 6.75e10, 8e10]:
    #     S_curve(2.3e3, 1e4, M, alpha, r)
    #
    # plt.hlines(1e17, *plt.xlim(),linestyles='--')
    # plt.grid(True, which='both', ls='-')
    # plt.legend()
    # plt.title('S-curve')
    # plt.savefig('fig/s-curve2.pdf')
    # plt.close()

    for Teff in np.linspace(2.3e4, 6e4, 5):
        h = (G * M * r) ** (1 / 2)
        F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Teff ** 4
        # print(F / h)
        vs = MesaVerticalStructure(M, alpha, r, F)
        # vs.plot()
        vs.TempGrad()
        print('Teff = %d' % vs.Teff)
        print('tau = %d' % vs.tau0())


if __name__ == '__main__':
    main()
