#!/usr/bin/env python3
"""
Module contains several classes that represent vertical structure of accretion disc in case
of (tabular) Mesa opacity and(or) EOS.

Class MesaVerticalStructure --  for (tabular) Mesa opacities and EOS with radiative energy transport.
Class MesaIdealVerticalStructure -- for (tabular) Mesa opacities and ideal gas EOS with radiative energy transport.
Class MesaVerticalStructureAdiabatic -- for (tabular) Mesa opacities and EOS with adiabatic energy transport.
Class MesaVerticalStructureFirstAssumption -- for (tabular) Mesa opacities and EOS with radiative+adiabatic energy
    transport.
Class MesaVerticalStructureRadConv -- for (tabular) Mesa opacities and EOS with radiative+convective energy transport.

"""
import os

import numpy as np
from astropy import constants as const
from astropy.units import Hz
from scipy.integrate import simps
from scipy.optimize import root, least_squares

from vs import BaseVerticalStructure, Vars, IdealGasMixin, RadiativeTempGradient

sigmaSB = const.sigma_sb.cgs.value
sigmaT = const.sigma_T.cgs.value
mu = const.u.cgs.value
c = const.c.cgs.value
pl_const = const.h
proton_mass = const.m_p.cgs.value
G = const.G.cgs.value
M_sun = const.M_sun.cgs.value

try:
    from opacity import Opac
except ModuleNotFoundError as e:
    raise ModuleNotFoundError('Mesa2py is not installed') from e


class BaseMesaVerticalStructure(BaseVerticalStructure):
    def __init__(self, Mx, alpha, r, F, eps=1e-5, mu=0.6, abundance='solar'):
        super().__init__(Mx, alpha, r, F, eps=eps, mu=mu)
        self.mesaop = Opac(abundance)


class MesaGasMixin:
    def law_of_rho(self, P, T, full_output):
        if full_output:
            rho, eos = self.mesaop.rho(P, T, full_output=full_output)
            return rho, eos
        else:
            return self.mesaop.rho(P, T, full_output=full_output)


class MesaOpacityMixin:
    def law_of_opacity(self, rho, T, lnfree_e):
        return self.mesaop.kappa(rho, T, lnfree_e=lnfree_e)


class AdiabaticTempGradient:
    """
    Temperature gradient class. Return adiabatic d(lnT)/d(lnP) from Mesa.

    """

    def dlnTdlnP(self, y, t):
        rho, eos = self.mesaop.rho(y[Vars.P] * self.P_norm, y[Vars.T] * self.T_norm, full_output=True)
        return eos.grad_ad


class FirstAssumptionRadiativeConvectiveGradient:
    """
    Temperature gradient class. Calculate d(lnT)/d(lnP) in first assumption.
    If gradient is over-adiabatic, then return adiabatic gradient, else calculate radiative gradient.

    """

    def dlnTdlnP(self, y, t):
        varkappa = self.opacity(y)
        rho, eos = self.mesaop.rho(y[Vars.P] * self.P_norm, y[Vars.T] * self.T_norm, True)

        if t == 1:
            dlnTdlnP_rad = - self.dQdz(y, t) * (y[Vars.P] / y[Vars.T] ** 4) * 3 * varkappa * (
                    self.Q_norm * self.P_norm / self.T_norm ** 4) / (16 * sigmaSB * self.z0 * self.omegaK ** 2)
        else:
            dTdz = ((abs(y[Vars.Q]) / y[Vars.T] ** 3) * 3 * varkappa * rho * self.z0 * self.Q_norm / (
                    16 * sigmaSB * self.T_norm ** 4))
            dPdz = rho * (1 - t) * self.omegaK ** 2 * self.z0 ** 2 / self.P_norm

            dlnTdlnP_rad = (y[Vars.P] / y[Vars.T]) * (dTdz / dPdz)

        if dlnTdlnP_rad < eos.grad_ad:
            return dlnTdlnP_rad
        else:
            return eos.grad_ad


class RadConvTempGradient:
    """
    Temperature gradient class. Calculate d(lnT)/d(lnP) in presence of convection according to mixing length theory.

    """

    def dlnTdlnP(self, y, t):
        rho, eos = self.rho(y, True)
        varkappa = self.opacity(y, lnfree_e=eos.lnfree_e)

        if t == 1:
            dlnTdlnP_rad = - self.dQdz(y, t) * (y[Vars.P] / y[Vars.T] ** 4) * 3 * varkappa * (
                    self.Q_norm * self.P_norm / self.T_norm ** 4) / (16 * sigmaSB * self.z0 * self.omegaK ** 2)
        else:
            dTdz = (abs(y[Vars.Q]) / y[Vars.T] ** 3) * 3 * varkappa * rho * self.z0 * self.Q_norm / (
                    16 * sigmaSB * self.T_norm ** 4)
            dPdz = rho * (1 - t) * self.omegaK ** 2 * self.z0 ** 2 / self.P_norm
            dlnTdlnP_rad = (y[Vars.P] / y[Vars.T]) * (dTdz / dPdz)

        if dlnTdlnP_rad < eos.grad_ad:
            return dlnTdlnP_rad
        if t == 1:
            return dlnTdlnP_rad

        alpha_ml = 1.5
        H_p = y[Vars.P] * self.P_norm / (
                rho * self.omegaK ** 2 * self.z0 * (1 - t) + self.omegaK * np.sqrt(y[Vars.P] * self.P_norm * rho))
        H_ml = alpha_ml * H_p
        omega = varkappa * rho * H_ml
        A = 9 / 8 * omega ** 2 / (3 + omega ** 2)
        der = eos.dlnRho_dlnT_const_Pgas
        if der > 0:
            der = -1
            print('FUUUUU...!')

        VV = -((3 + omega ** 2) / (
                3 * omega)) ** 2 * eos.c_p ** 2 * rho ** 2 * H_ml ** 2 * self.omegaK ** 2 * self.z0 * (1 - t) / (
                     512 * sigmaSB ** 2 * y[Vars.T] ** 6 * self.T_norm ** 6 * H_p) * der * (
                     dlnTdlnP_rad - eos.grad_ad)
        V = 1 / np.sqrt(VV)

        coeff = [2 * A, V, V ** 2, - V]
        # print(coeff)
        x = np.roots(coeff)
        # try:
        #     x = np.roots(coeff)
        # except np.linalg.LinAlgError:
        #     print('LinAlgError')
        #     breakpoint()

        x = [a.real for a in x if a.imag == 0 and 0.0 < a.real < 1.0]
        # if len(x) != 1:
        #     print('not one x of there is no right x')
        #     breakpoint()
        x = x[0]
        # print(x)
        # print(H_p, H_ml, omega, eos.c_p, eos.dlnRho_dlnT_const_Pgas, dlnTdlnP_rad - eos.grad_ad, VV, rho,
        #       y[Vars.P] * self.P_norm, y[Vars.T] * self.T_norm, eos.grad_ad, dlnTdlnP_rad, varkappa)

        dlnTdlnP_conv = eos.grad_ad + (dlnTdlnP_rad - eos.grad_ad) * x * (x + V)
        return dlnTdlnP_conv


class Advection:
    def Q_adv(self, P):
        v_r = - self.alpha * self.r * self.omegaK * (self.z0 / self.r) ** 2
        beta = 1
        Q_adv = v_r * (self.z0 / self.r) * P * (-15 + 99 / 8 * beta)
        return Q_adv

    def initial(self):
        """
        Initial conditions.

        Returns
        -------
        array

        """

        Q_initial = self.Q_initial()
        y = np.empty(4, dtype=np.float)
        y[Vars.S] = 0
        # y[Vars.S] = 0.1 / self.sigma_norm
        # y[Vars.S] = self.P_ph() / (self.z0 * self.omegaK ** 2) / self.sigma_norm
        y[Vars.P] = self.P_ph() / self.P_norm
        y[Vars.Q] = Q_initial
        y[Vars.T] = ((Q_initial - self.Q_adv(y[Vars.P] * self.P_norm) / self.Q_norm) * self.Q_norm / sigmaSB) ** (
                1 / 4) / self.T_norm
        return y

    def photospheric_pressure_equation(self, tau, y):
        Teff = ((self.Q0 - self.Q_adv(y)) / sigmaSB) ** (1 / 4)
        T = Teff * (1 / 2 + 3 * tau / 4) ** (1 / 4)
        rho = self.law_of_rho(y, T)
        varkappa = self.law_of_opacity(rho, T)
        return self.z0 * self.omegaK ** 2 / varkappa


class ExternalIrradiation:

    def sigma_d_nu(self, nu):  # cross-section in cm2 (Morrison & McCammon, 1983)
        E = (pl_const * nu * Hz).to('keV').value

        if 0.030 <= E <= 0.100:
            c_0 = 17.3
            c_1 = 608.1
            c_2 = -2150.0
        elif 0.100 < E <= 0.284:
            c_0 = 34.6
            c_1 = 267.9
            c_2 = -476.1
        elif 0.284 < E <= 0.400:
            c_0 = 78.1
            c_1 = 18.8
            c_2 = 4.3
        elif 0.400 < E <= 0.532:
            c_0 = 71.4
            c_1 = 66.8
            c_2 = -51.4
        elif 0.532 < E <= 0.707:
            c_0 = 95.5
            c_1 = 145.8
            c_2 = -61.1
        elif 0.707 < E <= 0.867:
            c_0 = 308.9
            c_1 = -380.6
            c_2 = 294.0
        elif 0.867 < E <= 1.303:
            c_0 = 120.6
            c_1 = 169.3
            c_2 = -47.7
        elif 1.303 < E <= 1.840:
            c_0 = 141.3
            c_1 = 146.8
            c_2 = -31.5
        elif 1.840 < E <= 2.471:
            c_0 = 202.7
            c_1 = 104.7
            c_2 = -17.0
        elif 2.471 < E <= 3.210:
            c_0 = 342.7
            c_1 = 18.7
            c_2 = 0.0
        elif 3.210 < E <= 4.038:
            c_0 = 352.2
            c_1 = 18.7
            c_2 = 0.0
        elif 4.038 < E <= 7.111:
            c_0 = 433.9
            c_1 = -2.4
            c_2 = 0.75
        elif 7.111 < E <= 8.331:
            c_0 = 629.0
            c_1 = 30.9
            c_2 = 0.0
        elif 8.331 < E <= 10.000:
            c_0 = 701.2
            c_1 = 25.2
            c_2 = 0.0

        result = (c_0 + c_1 * E + c_2 * E ** 2) * E ** (-3) * 1e-24  # cross-section in cm2
        return result

    def J_tot(self, F_nu, y, tau_Xray, t):
        sigma = 0.34
        try:
            _ = len(self.nu_irr)
            k_d_nu = np.array([self.sigma_d_nu(nu) for nu in self.nu_irr]) / proton_mass
        except TypeError:
            k_d_nu = self.sigma_d_nu(self.nu_irr) / proton_mass  # cross-section per proton mass
        tau = (sigma + k_d_nu) * y[Vars.S] * self.sigma_norm / 2  # tau = varkappa * mass_coord (Sigma / 2)
        lamb = sigma / (sigma + k_d_nu)
        k = np.sqrt(3 * (1 - lamb))
        zeta_0 = self.cos_theta_irr

        D_nu = 3 * lamb * zeta_0 ** 2 / (1 - k ** 2 * zeta_0 ** 2)

        C_nu = D_nu * (1 + np.exp(-tau_Xray / zeta_0) + 2 / (3 * zeta_0) * (1 + np.exp(-tau_Xray / zeta_0))) / (
                1 + np.exp(-k * tau_Xray) + 2 * k / 3 * (1 + np.exp(-k * tau_Xray)))

        J_tot = F_nu / (4 * np.pi) * (
                C_nu * (np.exp(-k * tau) + np.exp(-k * (tau_Xray - tau))) +
                (1 - D_nu) * (np.exp(-tau / zeta_0) + np.exp(-(tau_Xray - tau) / zeta_0))
        )

        i = 0
        if (tau_Xray[i] - tau[i]) < 0:
            print('tau0 < tau, J, min nu: ', t, sigma + k_d_nu[i],
                  self.Sigma0_par, y[Vars.S] * self.sigma_norm / 2,
                  self.Sigma0_par - y[Vars.S] * self.sigma_norm / 2,
                  tau_Xray[0], tau[0], tau_Xray[i] - tau[i],
                  -k[i] * (tau_Xray[i] - tau[i]), -(tau_Xray[i] - tau[i]) / zeta_0, J_tot[i])
            raise Exception

        # for i, JJ in enumerate(J_tot):
        #     if (tau_Xray[i] - tau[i]) < 0:
        #         print('tau0 < tau, J: ', self.nu_irr[i], sigma + k_d_nu[i],
        #               self.Sigma0_par - y[Vars.S] * self.sigma_norm / 2,
        #               tau_Xray[i], tau[i], tau_Xray[i] - tau[i],
        #               -k[i] * (tau_Xray[i] - tau[i]), -(tau_Xray[i] - tau[i]) / zeta_0, JJ)
        #     if np.isinf(JJ) or np.isnan(JJ) or JJ < 0:
        #         print('J_tot = ', JJ, self.nu_irr[i], tau_Xray[i], tau[i],
        #               -k[i] * (tau_Xray[i] - tau[i]), -tau[i] / zeta_0, -(tau_Xray[i] - tau[i]) / zeta_0)
        #         raise Exception

        # if np.isinf(J_tot) or np.isnan(J_tot) or J_tot < 0:
        #     print('J_tot = ', J_tot, tau_Xray, tau, -k * (tau_Xray - tau), -tau / zeta_0, -(tau_Xray - tau) / zeta_0)

        return J_tot

    def H_tot(self, F_nu, tau_Xray):
        sigma = 0.34
        try:
            _ = len(self.nu_irr)
            k_d_nu = np.array([self.sigma_d_nu(nu) for nu in self.nu_irr]) / proton_mass
        except TypeError:
            k_d_nu = self.sigma_d_nu(self.nu_irr) / proton_mass  # cross-section per proton mass
        Sigma_ph = self.P_ph() / (self.z0 * self.omegaK ** 2)
        tau = (sigma + k_d_nu) * Sigma_ph  # varkappa * Sigma_ph
        lamb = sigma / (sigma + k_d_nu)
        k = np.sqrt(3 * (1 - lamb))
        zeta_0 = self.cos_theta_irr

        D_nu = 3 * lamb * zeta_0 ** 2 / (1 - k ** 2 * zeta_0 ** 2)

        C_nu = D_nu * (1 + np.exp(-tau_Xray / zeta_0) + 2 / (3 * zeta_0) * (1 + np.exp(-tau_Xray / zeta_0))) / (
                1 + np.exp(-k * tau_Xray) + 2 * k / 3 * (1 + np.exp(-k * tau_Xray)))

        H_tot = F_nu * (
                k * C_nu / 3 * (np.exp(-k * tau) - np.exp(-k * (tau_Xray - tau))) +
                (zeta_0 - D_nu / (3 * zeta_0)) * (np.exp(-tau / zeta_0) - np.exp(-(tau_Xray - tau) / zeta_0))
        )

        i = 0
        if (tau_Xray[i] - tau[i]) < 0:
            print('tau0 < tau, H, min nu: ', self.nu_irr[i], sigma + k_d_nu[i],
                  self.Sigma0_par - Sigma_ph,
                  tau_Xray[i], tau[i], tau_Xray[i] - tau[i],
                  -k[i] * (tau_Xray[i] - tau[i]), -(tau_Xray[i] - tau[i]) / zeta_0, H_tot[i])
            raise Exception

        # for i, HH in enumerate(H_tot):
        #     if (tau_Xray[i] - tau[i]) < 0:
        #         print('tau0 < tau, H: ', self.nu_irr[i], sigma + k_d_nu[i],
        #               self.Sigma0_par - Sigma_ph,
        #               tau_Xray[i], tau[i], tau_Xray[i] - tau[i],
        #               -k[i] * (tau_Xray[i] - tau[i]), -(tau_Xray[i] - tau[i]) / zeta_0, HH)
        #     if np.isinf(HH) or np.isnan(HH) or HH < 0:
        #         print('H_tot = ', HH, self.nu_irr[i], tau_Xray[i], tau[i],
        #               -k[i] * (tau_Xray[i] - tau[i]), -tau[i] / zeta_0, -(tau_Xray[i] - tau[i]) / zeta_0)
        #         raise Exception

        # if np.isinf(H_tot) or np.isnan(H_tot) or H_tot < 0:
        #     print('H_tot = ', H_tot, tau_Xray, tau, -k * (tau_Xray - tau), -tau / zeta_0, -(tau_Xray - tau) / zeta_0)

        # self.values = np.array([C_nu, D_nu, lamb, k, H_tot, tau_Xray - tau,
        #                         np.exp(-tau / zeta_0) - np.exp(-(tau_Xray - tau) / zeta_0),
        #                         np.exp(-k * tau) - np.exp(-k * (tau_Xray - tau)),
        #                         tau, tau_Xray, self.P_ph() / (self.z0 * self.omegaK ** 2)])

        self.values = Sigma_ph

        # print('coeff_tot2 = ', (np.exp(-tau / zeta_0) - np.exp(-(tau_Xray - tau) / zeta_0)))
        # print('coeff_tot1 = ', (np.exp(-k * tau) - np.exp(-k * (tau_Xray - tau))))

        return H_tot

    def epsilon(self, y, t):
        rho = self.rho(y, full_output=False)
        sigma = 0.34
        try:
            _ = len(self.nu_irr)
            k_d_nu = np.array([self.sigma_d_nu(nu) for nu in self.nu_irr]) / proton_mass
        except TypeError:
            k_d_nu = self.sigma_d_nu(self.nu_irr) / proton_mass  # cross-section per proton mass

        tau_Xray = (sigma + k_d_nu) * self.Sigma0_par

        try:
            _ = len(self.F_nu_irr)
            epsilon = 4 * np.pi * rho * simps(k_d_nu * self.J_tot(self.F_nu_irr, y, tau_Xray, t), self.nu_irr)
            self.tau_Xray = simps(tau_Xray, self.nu_irr)
        except TypeError:
            epsilon = 4 * np.pi * rho * k_d_nu * self.J_tot(self.F_nu_irr, y, tau_Xray, t)
            self.tau_Xray = tau_Xray
        return epsilon

    def Q_irr_ph(self):
        sigma = 0.34
        try:
            _ = len(self.nu_irr)
            k_d_nu = np.array([self.sigma_d_nu(nu) for nu in self.nu_irr]) / proton_mass
        except TypeError:
            k_d_nu = self.sigma_d_nu(self.nu_irr) / proton_mass  # cross-section per proton mass

        tau_Xray = (sigma + k_d_nu) * self.Sigma0_par

        try:
            _ = len(self.F_nu_irr)
            Qirr = simps(self.H_tot(self.F_nu_irr, tau_Xray), self.nu_irr)
            self.tau_Xray = simps(tau_Xray, self.nu_irr)
        except TypeError:
            Qirr = self.H_tot(self.F_nu_irr, tau_Xray)
            self.tau_Xray = tau_Xray

        print('T_irr = {:g}'.format((Qirr / sigmaSB) ** (1 / 4)))

        h = np.sqrt(self.GM * self.r)
        eta_accr = 0.1
        rg = 2 * self.GM / c ** 2
        r_in = 3 * rg
        func = 1 - np.sqrt(r_in / self.r)
        Mdot = self.F / (h * func)

        self.T_irr = (Qirr / sigmaSB) ** (1 / 4)
        self.C_irr = Qirr / (eta_accr * Mdot * c ** 2) * (4 * np.pi * self.r ** 2)
        self.Q_irr = Qirr
        return Qirr

    def Q_initial(self):
        result = 1 + self.Q_irr_ph() / self.Q_norm
        # print('Q_irr, Q_norm =', self.Q_irr_ph(), self.Q_norm)
        # if result != 1:
        #     print('Initial =', result)
        return result

    def initial(self):
        """
        Initial conditions.

        Returns
        -------
        array

        """

        Q_initial = self.Q_initial()
        y = np.empty(4, dtype=np.float64)
        y[Vars.S] = 2 * self.P_ph() / (self.z0 * self.omegaK ** 2) / self.sigma_norm  # 2 -> full mass coordinate Sigma
        y[Vars.P] = self.P_ph() / self.P_norm
        y[Vars.Q] = Q_initial
        y[Vars.T] = (Q_initial * self.Q_norm / sigmaSB) ** (1 / 4) / self.T_norm  # Tph^4 = Teff^4 + Tirr^4
        return y

    def dQdz(self, y, t):
        w_r_phi = self.viscosity(y)
        result = -(3 / 2) * self.z0 * self.omegaK * w_r_phi / self.Q_norm - self.epsilon(y, t) * self.z0 / self.Q_norm
        # print('dQdz =', result)
        # print('dQdz_non = ', -(3 / 2) * self.z0 * self.omegaK * w_r_phi / self.Q_norm)
        # if self.epsilon(y) != 0:
        #     print('epsilon = ', self.epsilon(y))
        return result

    def Sigma0_init(self):
        return 9.73e-24 * (self.r / 1e10) ** (-11 / 10) * self.F ** (10 / 10) * (self.Mx / M_sun) ** (
                -1 / 10) * self.alpha ** (-4 / 5) / 1e6

    def dq(self, x, norm):
        self.Sigma0_par = abs(x[1]) * norm
        self.z0 = abs(x[0]) * self.r
        print('[{:g}, {:g}, {:g}]'.format(x[0], self.Sigma0_par, x[1]))
        q_c = np.array([self.y_c()[Vars.Q], self.Sigma0_par / self.parameters_C()[4] - 1])
        return q_c

    def fit(self, start_estimation_z0r=None, start_estimation_Sigma0=None):
        """
        Solve optimization problem and calculate the vertical structure.

        Returns
        -------
        double and result
            The value of normalized unknown free parameter z_0 / r and result of optimization.

        """

        if start_estimation_z0r is None:
            x0_z0r = self.z0_init() / self.r
        else:
            x0_z0r = start_estimation_z0r

        if start_estimation_Sigma0 is None:
            norm = self.Sigma0_init()
        else:
            norm = start_estimation_Sigma0

        # result = root(self.dq, x0=np.array([x0_z0r, 1]), args=norm, method='hybr')
        # print(result)

        result = least_squares(self.dq, x0=np.array([x0_z0r, 1]), args=(norm,),
                               verbose=2, loss='linear')
        result.x = abs(result.x) * np.array([1, norm])
        self.Sigma0_par = result.x[1]
        self.z0 = result.x[0] * self.r
        return result


class ExternalIrradiationZeroAssumption:
    def initial(self):
        """
        Initial conditions.

        Returns
        -------
        array

        """

        Q_initial = self.Q_initial()
        y = np.empty(4, dtype=np.float64)
        # y[Vars.S] = 0  # Whether to take into account Sigma_ph
        y[Vars.S] = 2 * self.P_ph() / (self.z0 * self.omegaK ** 2) / self.sigma_norm  # 2 -> full mass coordinate Sigma
        y[Vars.P] = self.P_ph() / self.P_norm
        y[Vars.Q] = Q_initial
        y[Vars.T] = (self.Teff ** 4 + self.T_irr ** 4) ** (1 / 4) / self.T_norm
        self.values = self.P_ph() / (self.z0 * self.omegaK ** 2)
        return y


class MesaVerticalStructure(MesaGasMixin, MesaOpacityMixin, RadiativeTempGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for (tabular) Mesa opacities and EOS with radiative energy transport.

    """
    pass


class MesaIdealVerticalStructure(IdealGasMixin, MesaOpacityMixin, RadiativeTempGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for (tabular) Mesa opacities and ideal gas EOS with radiative energy transport.

    """
    pass


class MesaVerticalStructureAdiabatic(MesaGasMixin, MesaOpacityMixin, AdiabaticTempGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for (tabular) Mesa opacities and EOS with adiabatic energy transport.

    """
    pass


class MesaVerticalStructureFirstAssumption(MesaGasMixin, MesaOpacityMixin, FirstAssumptionRadiativeConvectiveGradient,
                                           BaseMesaVerticalStructure):
    """
    Vertical structure class for (tabular) Mesa opacities and EOS with radiative+adiabatic energy transport.

    """
    pass


class MesaVerticalStructureRadConv(MesaGasMixin, MesaOpacityMixin, RadConvTempGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for (tabular) Mesa opacities and EOS with radiative+convective energy transport.

    """
    pass


class MesaVerticalStructureRadConvExternalIrradiation(MesaGasMixin, MesaOpacityMixin, RadConvTempGradient,
                                                      ExternalIrradiation, BaseMesaVerticalStructure):
    def __init__(self, Mx, alpha, r, F, nu_irr, F_nu_irr, cos_theta_irr=None, eps=1e-5, abundance='solar'):
        super().__init__(Mx, alpha, r, F, eps=eps, mu=0.6, abundance=abundance)
        self.nu_irr = nu_irr
        self.F_nu_irr = F_nu_irr
        self.Sigma0_par = self.Sigma0_init()
        if cos_theta_irr is None:
            self.cos_theta_irr = 1 / 8 * (self.z0 / self.r)
        else:
            self.cos_theta_irr = cos_theta_irr
        self.T_irr = None
        self.C_irr = None
        self.tau_Xray = None
        self.Q_irr = None
        self.values = None


class MesaVerticalStructureRadConvExternalIrradiationZeroAssumption(MesaGasMixin, MesaOpacityMixin, RadConvTempGradient,
                                                                    ExternalIrradiationZeroAssumption,
                                                                    BaseMesaVerticalStructure):
    def __init__(self, Mx, alpha, r, F, C_irr=None, T_irr=None, eps=1e-5, abundance='solar'):
        super().__init__(Mx, alpha, r, F, eps=eps, mu=0.6, abundance=abundance)
        h = np.sqrt(self.GM * self.r)
        eta_accr = 0.1
        rg = 2 * self.GM / c ** 2
        r_in = 3 * rg
        func = 1 - np.sqrt(r_in / r)
        Mdot = self.F / (h * func)
        if C_irr is None and T_irr is None:
            raise Exception('C_irr or T_irr is required.')
        elif T_irr is not None and C_irr is None:
            self.T_irr = T_irr
            self.Q_irr = sigmaSB * self.T_irr ** 4
            self.C_irr = self.Q_irr / (eta_accr * Mdot * c ** 2) * (4 * np.pi * self.r ** 2)
        elif C_irr is not None and T_irr is None:
            self.C_irr = C_irr
            self.Q_irr = self.C_irr * eta_accr * Mdot * c ** 2 / (4 * np.pi * self.r ** 2)
            self.T_irr = (self.Q_irr / sigmaSB) ** (1 / 4)
        else:
            raise Exception('Only one of (C_irr, T_irr) is required.')
        print('C_irr, T_irr = ', self.C_irr, self.T_irr)


class MesaVerticalStructureAdvection(MesaGasMixin, MesaOpacityMixin, RadiativeTempGradient, Advection,
                                     BaseMesaVerticalStructure):
    pass


class MesaIdealVerticalStructureAdvection(IdealGasMixin, MesaOpacityMixin, RadiativeTempGradient, Advection,
                                          BaseMesaVerticalStructure):
    pass


class MesaVerticalStructureRadConvAdvection(MesaGasMixin, MesaOpacityMixin, RadConvTempGradient, Advection,
                                            BaseMesaVerticalStructure):
    pass


def main():
    M = 10 * M_sun
    alpha = 0.01
    Mdot = 1e17
    rg = 2 * G * M / c ** 2
    r = 400 * rg
    print('Calculating structure and making a structure plot. '
          '\nStructure with tabular MESA opacity and EOS.')
    print('M = {:g} grams \nr = {:g} cm = {:g} rg \nalpha = {:g} \nMdot = {:g} g/s'.format(M, r, r / rg, alpha, Mdot))
    h = np.sqrt(G * M * r)
    r_in = 3 * rg
    F = Mdot * h * (1 - np.sqrt(r_in / r))
    vs = MesaVerticalStructureRadConv(M, alpha, r, F)
    z0r, result = vs.fit()
    if result.converged:
        print('The vertical structure has been calculated successfully.')
    print('z0/r = ', z0r)
    t = np.linspace(0, 1, 100)
    S, P, Q, T = vs.integrate(t)[0]
    import matplotlib.pyplot as plt
    plt.plot(1 - t, S, label=r'$\hat{\Sigma}$')
    plt.plot(1 - t, P, label=r'$\hat{P}$')
    plt.plot(1 - t, Q, label=r'$\hat{Q}$')
    plt.plot(1 - t, T, label=r'$\hat{T}$')
    plt.xlabel('$z / z_0$')
    plt.title(r'$M = {:g}\, M_{{\odot}},\, \dot{{M}} = {:g}\, {{\rm g/s}},\, \alpha = {:g}, r = {:g} \,\rm cm$'.format(
        M / M_sun, Mdot, alpha, r))
    plt.grid()
    plt.legend()
    if not os.path.exists('fig/'):
        os.makedirs('fig/')
    plt.savefig('fig/vs_mesa.pdf')
    print('Plot of structure is successfully saved to fig/vs_mesa.pdf.')
    return


if __name__ == '__main__':
    main()
