#!/usr/bin/env python3
"""
Module contains several classes that represent vertical structure of accretion disc in case
of (tabular) MESA opacity and(or) EOS.

Class MesaVerticalStructure --  for MESA opacities and EoS with radiative energy transport.
Class MesaIdealVerticalStructure -- for MESA opacities and ideal gas EoS with radiative energy transport.
Class MesaVerticalStructureAd -- for MESA opacities and EoS with adiabatic energy transport.
Class MesaVerticalStructureRadAd -- for MESA opacities and EoS with radiative+adiabatic energy transport.
Class MesaVerticalStructureRadConv -- for MESA opacities and EoS with radiative+convective energy transport.

Class MesaVerticalStructureExternalIrradiation -- for MESA opacities and EoS with radiative energy transport
    and advanced external irradiation scheme from (Mescheryakov et al. 2011).
Class MesaVerticalStructureRadAdExternalIrradiation -- for MESA opacities and EOS
    with radiative+adiabatic energy transport and advanced external irradiation scheme
    from (Mescheryakov et al. 2011).
Class MesaVerticalStructureRadConvExternalIrradiation -- for MESA opacities and EOS
    with radiative+convective energy transport and advanced external irradiation scheme
    from (Mescheryakov et al. 2011).

Class MesaVerticalStructureExternalIrradiationZeroAssumption -- for MESA opacities and EOS
    with radiative energy transport and simple external irradiation scheme via T_irr or C_irr.
Class MesaVerticalStructureRadAdExternalIrradiationZeroAssumption -- for MESA opacities and EOS
    with radiative+adiabatic energy transport and simple external irradiation scheme via T_irr or C_irr.
Class MesaVerticalStructureRadConvExternalIrradiationZeroAssumption -- for MESA opacities and EOS
    with radiative+convective energy transport and simple external irradiation scheme via T_irr or C_irr.

"""
import os
from types import FunctionType

import numpy as np
from astropy import constants as const
from astropy import units
from scipy.integrate import simps
from scipy.optimize import root, least_squares, brentq

from vs import BaseVerticalStructure, Vars, IdealGasMixin, RadiativeTempGradient

sigmaSB = const.sigma_sb.cgs.value
sigmaT = const.sigma_T.cgs.value
atomic_mass = const.u.cgs.value
c = const.c.cgs.value
proton_mass = const.m_p.cgs.value
G = const.G.cgs.value
M_sun = const.M_sun.cgs.value

try:
    from opacity import Opac
except ModuleNotFoundError as e:
    raise ModuleNotFoundError('Mesa2py is not installed') from e


class NotConvergeError(Exception):
    def __init__(self, Sigma0_par, z0r):
        self.Sigma0_par = Sigma0_par
        self.z0r = z0r


class PphNotConvergeError(Exception):
    def __init__(self, func_Pph):
        self.func_Pph = abs(func_Pph)


class BaseMesaVerticalStructure(BaseVerticalStructure):
    def __init__(self, Mx, alpha, r, F, eps=1e-5, mu=0.6, abundance='solar'):
        super().__init__(Mx, alpha, r, F, eps=eps, mu=mu)
        self.mesaop = Opac(abundance)


class BaseExternalIrradiation(BaseMesaVerticalStructure):
    def __init__(self, Mx, alpha, r, F, nu_irr, spectrum_irr, L_X_irr, spectrum_irr_par,
                 args_spectrum_irr=(), kwargs_spectrum_irr={}, cos_theta_irr=None, cos_theta_irr_exp=1 / 12,
                 eps=1e-5, abundance='solar', P_ph_0=None):
        super().__init__(Mx, alpha, r, F, eps=eps, mu=0.6, abundance=abundance)

        if spectrum_irr is None:
            raise Exception("spectrum_irr must be a function or an array-like, not None.")
        if L_X_irr is None:
            raise Exception("L_X_irr must be a double, not None.")
        if nu_irr is None:
            raise Exception("nu_irr must be an array-like, not None.")

        if isinstance(spectrum_irr, FunctionType):
            if spectrum_irr_par == 'nu':
                self.nu_irr = nu_irr
                spectrum_irr = spectrum_irr(self.nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr) / simps(
                    spectrum_irr(self.nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr), self.nu_irr)
            elif spectrum_irr_par == 'E_in_keV':
                self.nu_irr = (nu_irr * units.keV).to('Hz', equivalencies=units.spectral()).value
                spectrum_irr = spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr) / simps(
                    spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr), nu_irr) * \
                               units.Hz.to('keV', equivalencies=units.spectral())
            else:
                raise Exception("spectrum_irr_par must be 'nu' or 'E_in_keV', not None or anything else.")
        else:
            if len(nu_irr) != len(spectrum_irr):
                raise Exception("'nu_irr' and 'spectrum_irr' must have the same size.")
            if spectrum_irr_par == 'nu':
                self.nu_irr = nu_irr
                spectrum_irr = spectrum_irr
            elif spectrum_irr_par == 'E_in_keV':
                self.nu_irr = (nu_irr * units.keV).to('Hz', equivalencies=units.spectral()).value
                spectrum_irr = spectrum_irr * units.Hz.to('keV', equivalencies=units.spectral())
            else:
                raise Exception("spectrum_irr_par must be 'nu' or 'E_in_keV', not None or anything else.")

        F_nu_irr = L_X_irr / (4 * np.pi * r ** 2) * spectrum_irr
        self.F_nu_irr = F_nu_irr
        self.Sigma0_par = self.Sigma0_init()
        if cos_theta_irr is None:
            self.cos_theta_irr_key = False
            self.cos_theta_irr_exp = cos_theta_irr_exp
        else:
            self.cos_theta_irr_key = True
            self.cos_theta_irr = cos_theta_irr
        self.T_irr = None
        self.C_irr = None
        self.Q_irr = None
        self.Sigma_ph = None
        self.P_ph_0 = P_ph_0
        self.P_ph_key = False
        self.P_ph_parameter = None
        self.converged = False

    @property
    def cos_theta_irr(self):
        if self.cos_theta_irr_key:
            return self.__cos_theta_irr
        else:
            return self.cos_theta_irr_exp * self.z0 / self.r

    @cos_theta_irr.setter
    def cos_theta_irr(self, value):
        self.__cos_theta_irr = value


class BaseExternalIrradiationZeroAssumption(BaseMesaVerticalStructure):
    def __init__(self, Mx, alpha, r, F, C_irr=None, T_irr=None, eps=1e-5, abundance='solar', P_ph_0=None):
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
        self.P_ph_0 = P_ph_0
        self.P_ph_key = False
        self.P_ph_parameter = None


class MesaGasMixin:
    def law_of_rho(self, P, T, full_output):
        if full_output:
            rho, eos = self.mesaop.rho(P, T, full_output=full_output)
            return rho, eos
        else:
            return self.mesaop.rho(P, T, full_output=full_output)


class MesaOpacityMixin:
    def law_of_opacity(self, rho, T, lnfree_e, return_grad):
        if return_grad:
            kappa, dlnkap_dlnRho, dlnkap_dlnT = self.mesaop.kappa(rho, T, lnfree_e=lnfree_e, return_grad=return_grad)
            return kappa, dlnkap_dlnRho, dlnkap_dlnT
        else:
            return self.mesaop.kappa(rho, T, lnfree_e=lnfree_e, return_grad=return_grad)


class AdiabaticTempGradient:
    """
    Temperature gradient class. Returns adiabatic d(lnT)/d(lnP) from Mesa.

    """

    def dlnTdlnP(self, y, t):
        rho, eos = self.mesaop.rho(y[Vars.P] * self.P_norm, y[Vars.T] * self.T_norm, full_output=True)
        return eos.grad_ad


class RadiativeAdiabaticGradient:
    """
    Temperature gradient class.
    If gradient is over-adiabatic, then returns adiabatic d(lnT)/d(lnP), else calculates radiative d(lnT)/d(lnP).

    """

    def dlnTdlnP(self, y, t):
        rho, eos = self.rho(y, True)
        varkappa = self.opacity(y, lnfree_e=eos.lnfree_e, return_grad=False)

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
    Temperature gradient class. Calculates d(lnT)/d(lnP) in presence of convection according to mixing length theory.

    """

    def dlnTdlnP(self, y, t):
        rho, eos = self.rho(y, True)
        varkappa = self.opacity(y, lnfree_e=eos.lnfree_e, return_grad=False)

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

        VV = -((3 + omega ** 2) / (
                3 * omega)) ** 2 * eos.c_p ** 2 * rho ** 2 * H_ml ** 2 * self.omegaK ** 2 * self.z0 * (1 - t) / (
                     512 * sigmaSB ** 2 * y[Vars.T] ** 6 * self.T_norm ** 6 * H_p) * der * (
                     dlnTdlnP_rad - eos.grad_ad)
        V = 1 / np.sqrt(VV)

        coeff = [2 * A, V, V ** 2, - V]

        try:
            x = np.roots(coeff)
        except np.linalg.LinAlgError:
            print('LinAlgError, coeff[2A, V, V ** 2, -V] = ', coeff)
            raise

        x = [a.real for a in x if a.imag == 0 and 0.0 < a.real < 1.0]
        if len(x) != 1:
            print('not one x of there is no right x')
            raise Exception
        x = x[0]
        dlnTdlnP_conv = eos.grad_ad + (dlnTdlnP_rad - eos.grad_ad) * x * (x + V)
        return dlnTdlnP_conv


class ExternalIrradiation:
    @staticmethod
    def sigma_d_nu(nu):  # cross-section in cm2 (Morrison & McCammon, 1983) from 0.03 to 10 keV
        E = (nu * units.Hz).to('keV', equivalencies=units.spectral()).value

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

    def J_tot(self, F_nu, y, tau_Xray, t):  # mean intensity as function of tau
        sigma_sc = 0.34  # not exactly sigma_sc in case of not fully ionized gas
        k_d_nu = np.array([self.sigma_d_nu(nu) for nu in self.nu_irr]) / proton_mass  # cross-section per proton mass
        tau = (sigma_sc + k_d_nu) * \
              (y[Vars.S] * self.sigma_norm + 2 * self.Sigma_ph) / 2  # tau = varkappa * (Sigma + 2 * Sigma_ph) / 2

        lamb = sigma_sc / (sigma_sc + k_d_nu)
        k = np.sqrt(3 * (1 - lamb))
        zeta_0 = self.cos_theta_irr

        D_nu = 3 * lamb * zeta_0 ** 2 / (1 - k ** 2 * zeta_0 ** 2)

        C_nu = D_nu * (1 + np.exp(-tau_Xray / zeta_0) + 2 / (3 * zeta_0) * (1 + np.exp(-tau_Xray / zeta_0))) / (
                1 + np.exp(-k * tau_Xray) + 2 * k / 3 * (1 + np.exp(-k * tau_Xray)))

        J_tot = F_nu / (4 * np.pi) * (
                C_nu * (np.exp(-k * tau) + np.exp(-k * (tau_Xray - tau))) +
                (1 - D_nu) * (np.exp(-tau / zeta_0) + np.exp(-(tau_Xray - tau) / zeta_0))
        )

        if (tau_Xray[0] - tau[0]) < 0:
            raise NotConvergeError(self.Sigma0_par, self.z0 / self.r)
        return J_tot

    def H_tot(self, F_nu, tau_Xray, Pph):  # eddington flux at the photosphere
        sigma_sc = 0.34  # not exactly sigma_sc in case of not fully ionized gas
        k_d_nu = np.array([self.sigma_d_nu(nu) for nu in self.nu_irr]) / proton_mass  # cross-section per proton mass
        tau = (sigma_sc + k_d_nu) * Pph / (self.z0 * self.omegaK ** 2)  # tau at the photosphere
        lamb = sigma_sc / (sigma_sc + k_d_nu)
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
            raise NotConvergeError(self.Sigma0_par, self.z0 / self.r)
        return H_tot

    def epsilon(self, y, t):
        rho, eos = self.rho(y, full_output=True)
        sigma_sc = 0.34  # not exactly sigma_sc in case of not fully ionized gas
        k_d_nu = np.array([self.sigma_d_nu(nu) for nu in self.nu_irr]) / proton_mass  # cross-section per proton mass
        tau_Xray = (sigma_sc + k_d_nu) * (self.Sigma0_par + 2 * self.Sigma_ph)
        epsilon = 4 * np.pi * rho * simps(k_d_nu * self.J_tot(self.F_nu_irr, y, tau_Xray, t), self.nu_irr)
        return epsilon

    def Q_irr_ph(self, Pph):
        sigma_sc = 0.34  # not exactly sigma_sc in case of not fully ionized gas
        k_d_nu = np.array([self.sigma_d_nu(nu) for nu in self.nu_irr]) / proton_mass  # cross-section per proton mass
        tau_Xray = (sigma_sc + k_d_nu) * (self.Sigma0_par + 2 * Pph / (self.z0 * self.omegaK ** 2))
        Qirr = simps(self.H_tot(self.F_nu_irr, tau_Xray, Pph), self.nu_irr)
        return Qirr

    def photospheric_pressure_equation_irr(self, tau, P, Pph):
        T = (self.Teff ** 4 * (1 / 2 + 3 * tau / 4) + self.Q_irr_ph(Pph) / sigmaSB) ** (1 / 4)
        rho, eos = self.law_of_rho(P, T, True)
        varkappa = self.law_of_opacity(rho, T, lnfree_e=eos.lnfree_e, return_grad=False)
        return self.z0 * self.omegaK ** 2 / varkappa

    def P_ph_irr(self, Pph):
        solution = self.photospheric_pressure_equation_irr(tau=2 / 3, P=Pph, Pph=Pph) * 2 / 3
        return solution

    def Q_initial(self, Pph):
        result = 1 + self.Q_irr_ph(Pph) / self.Q_norm
        return result

    def initial(self):
        """
        Initial conditions.

        Returns
        -------
        array

        """

        if self.P_ph_0 is None:
            self.P_ph_0 = self.P_ph()

        def fun_P_ph(x):
            return abs(x) - self.P_ph_irr(abs(x))

        if not self.P_ph_key:
            sign_P_ph = fun_P_ph(self.P_ph_0)
            if sign_P_ph > 0:
                factor = 0.5
            else:
                factor = 2.0

            while True:
                self.P_ph_0 *= factor
                if sign_P_ph * fun_P_ph(self.P_ph_0) < 0:
                    break
            P_ph, res = brentq(fun_P_ph, self.P_ph_0, self.P_ph_0 / factor, full_output=True)
            if abs(fun_P_ph(P_ph)) > 1e-7:
                raise PphNotConvergeError(fun_P_ph(P_ph))
            self.P_ph_0 = P_ph + 1
            P_ph = abs(res.root)
            if self.fitted:
                self.P_ph_parameter = abs(res.root)
                P_ph = self.P_ph_parameter
                self.P_ph_key = True
        else:
            P_ph = self.P_ph_parameter

        self.Sigma_ph = P_ph / (self.z0 * self.omegaK ** 2)
        Q_initial = self.Q_initial(Pph=P_ph)
        Qirr = (Q_initial - 1) * self.Q_norm
        self.C_irr = Qirr / simps(self.F_nu_irr, self.nu_irr)
        self.T_irr = (Qirr / sigmaSB) ** (1 / 4)
        self.Q_irr = Qirr
        y = np.empty(4, dtype=np.float64)
        y[Vars.S] = 0
        y[Vars.P] = P_ph / self.P_norm
        y[Vars.Q] = Q_initial
        y[Vars.T] = (Q_initial * self.Q_norm / sigmaSB) ** (1 / 4) / self.T_norm  # Tph^4 = Teff^4 + Tirr^4
        return y

    def dQdz(self, y, t):
        w_r_phi = self.viscosity(y)
        result = -(3 / 2) * self.z0 * self.omegaK * w_r_phi / self.Q_norm - self.epsilon(y, t) * self.z0 / self.Q_norm
        return result

    def Sigma0_init(self):
        return 9.73e-24 * (self.r / 1e10) ** (-11 / 10) * self.F ** (10 / 10) * (self.Mx / M_sun) ** (
                -1 / 10) * self.alpha ** (-4 / 5) / 1e6

    def dq(self, x, norm):
        self.Sigma0_par = abs(x[1]) * norm
        self.z0 = abs(x[0]) * self.r
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

        cost_root = -1
        try:
            result = root(self.dq, x0=np.array([x0_z0r, 1]), args=norm, method='hybr')
            cost_root = result.fun[0] ** 2 + result.fun[1] ** 2
        except (NotConvergeError, PphNotConvergeError):
            try:
                result = least_squares(self.dq, x0=np.array([x0_z0r, 1]), args=(norm,),
                                       verbose=0, loss='linear')
                if result.cost <= 1e-16:
                    self.converged = True
            except NotConvergeError as err:
                print('Not converged, try larger Sigma0_par or smaller z0r approximations. '
                      'Current approximations are Sigma0_par = {:g}, z0r = {:g}.'.format(err.Sigma0_par, err.z0r))
                raise err
            except PphNotConvergeError as err:
                print('Not converged')
                raise err

        if cost_root > 1e-16:
            try:
                result_least_squares = least_squares(self.dq, x0=np.array([x0_z0r, 1]), args=(norm,),
                                                     verbose=0, loss='linear')
                if result_least_squares.cost * 2 < cost_root:
                    result = result_least_squares
                    if result_least_squares.cost <= 1e-16:
                        self.converged = True
            except (NotConvergeError, PphNotConvergeError):
                pass
        else:
            self.converged = True

        result.x = abs(result.x) * np.array([1, norm])
        self.Sigma0_par = result.x[1]
        self.z0 = result.x[0] * self.r
        self.fitted = True  # doesn't mean that structure converged successfully, need only for Pph
        return result


class ExternalIrradiationZeroAssumption:
    def photospheric_pressure_equation(self, tau, P):
        T = (self.Teff ** 4 * (1 / 2 + 3 * tau / 4) + self.T_irr) ** (1 / 4)
        rho, eos = self.law_of_rho(P, T, True)
        varkappa = self.law_of_opacity(rho, T, lnfree_e=eos.lnfree_e, return_grad=False)
        return self.z0 * self.omegaK ** 2 / varkappa

    def P_ph_irr(self, Pph):
        solution = self.photospheric_pressure_equation(tau=2 / 3, P=Pph) * 2 / 3
        return solution

    def initial(self):
        """
        Initial conditions.

        Returns
        -------
        array

        """

        if self.P_ph_0 is None:
            self.P_ph_0 = self.P_ph()

        def fun_P_ph(x):
            return abs(x) - self.P_ph_irr(abs(x))

        if not self.P_ph_key:
            sign_P_ph = fun_P_ph(self.P_ph_0)
            if sign_P_ph > 0:
                factor = 0.5
            else:
                factor = 2.0

            while True:
                self.P_ph_0 *= factor
                if sign_P_ph * fun_P_ph(self.P_ph_0) < 0:
                    break
            P_ph, res = brentq(fun_P_ph, self.P_ph_0, self.P_ph_0 / factor, full_output=True)
            if abs(fun_P_ph(P_ph)) > 1e-7:
                raise PphNotConvergeError(fun_P_ph(P_ph))
            self.P_ph_0 = P_ph + 1
            P_ph = abs(res.root)
            if self.fitted:
                self.P_ph_parameter = abs(res.root)
                P_ph = self.P_ph_parameter
                self.P_ph_key = True
        else:
            P_ph = self.P_ph_parameter

        Q_initial = self.Q_initial()
        y = np.empty(4, dtype=np.float64)
        y[Vars.S] = 0
        y[Vars.P] = P_ph / self.P_norm
        y[Vars.Q] = Q_initial
        y[Vars.T] = (self.Teff ** 4 + self.T_irr ** 4) ** (1 / 4) / self.T_norm
        return y


# Classes with MESA opacity without Irradiation
class MesaVerticalStructure(MesaGasMixin, MesaOpacityMixin, RadiativeTempGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for MESA opacities and EoS with radiative energy transport.

    """
    pass


class MesaIdealGasVerticalStructure(IdealGasMixin, MesaOpacityMixin, RadiativeTempGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for MESA opacities and ideal gas EoS with radiative energy transport.

    """
    pass


class MesaVerticalStructureAd(MesaGasMixin, MesaOpacityMixin, AdiabaticTempGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for MESA opacities and EoS with adiabatic energy transport.

    """
    pass


class MesaVerticalStructureRadAd(MesaGasMixin, MesaOpacityMixin, RadiativeAdiabaticGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for MESA opacities and EoS with radiative+adiabatic energy transport.

    """
    pass


class MesaVerticalStructureRadConv(MesaGasMixin, MesaOpacityMixin, RadConvTempGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for MESA opacities and EoS with radiative+convective energy transport.

    """
    pass


# Classes with MESA opacity and with advanced Irradiation
class MesaVerticalStructureExternalIrradiation(MesaGasMixin, MesaOpacityMixin, RadiativeTempGradient,
                                               ExternalIrradiation, BaseExternalIrradiation):
    """
    Vertical structure class for MESA opacities and EoS with radiative energy transport
    and advanced external irradiation scheme from (Mescheryakov et al. 2011).

    """
    pass


class MesaVerticalStructureRadAdExternalIrradiation(MesaGasMixin, MesaOpacityMixin, RadiativeAdiabaticGradient,
                                                    ExternalIrradiation, BaseExternalIrradiation):
    """
    Vertical structure class for MESA opacities and EoS with radiative+adiabatic energy transport
    and advanced external irradiation scheme from (Mescheryakov et al. 2011).

    """
    pass


class MesaVerticalStructureRadConvExternalIrradiation(MesaGasMixin, MesaOpacityMixin, RadConvTempGradient,
                                                      ExternalIrradiation, BaseExternalIrradiation):
    """
    Vertical structure class for MESA opacities and EoS with radiative+convective energy transport
    and advanced external irradiation scheme from (Mescheryakov et al. 2011).

    """
    pass


# Classes with MESA opacity and with Zero Assumption Irradiation
class MesaVerticalStructureExternalIrradiationZeroAssumption(MesaGasMixin, MesaOpacityMixin, RadiativeTempGradient,
                                                             ExternalIrradiationZeroAssumption,
                                                             BaseExternalIrradiationZeroAssumption):
    """
    Vertical structure class for MESA opacities and EoS with radiative energy transport
    and simple external irradiation scheme via T_irr or C_irr.

    """
    pass


class MesaVerticalStructureRadAdExternalIrradiationZeroAssumption(MesaGasMixin, MesaOpacityMixin,
                                                                  RadiativeAdiabaticGradient,
                                                                  ExternalIrradiationZeroAssumption,
                                                                  BaseExternalIrradiationZeroAssumption):
    """
    Vertical structure class for MESA opacities and EoS with radiative+adiabatic energy transport
    and simple external irradiation scheme via T_irr or C_irr.

    """
    pass


class MesaVerticalStructureRadConvExternalIrradiationZeroAssumption(MesaGasMixin, MesaOpacityMixin, RadConvTempGradient,
                                                                    ExternalIrradiationZeroAssumption,
                                                                    BaseExternalIrradiationZeroAssumption):
    """
    Vertical structure class for MESA opacities and EoS with radiative+convective energy transport
    and simple external irradiation scheme via T_irr or C_irr.

    """
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
    os.makedirs('fig/', exist_ok=True)
    plt.savefig('fig/vs_mesa.pdf')
    print('Plot of structure is successfully saved to fig/vs_mesa.pdf.')
    return


if __name__ == '__main__':
    main()
