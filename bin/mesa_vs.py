import numpy as np
from astropy import constants as const
from astropy.units import Hz
from scipy.integrate import simps, solve_ivp
from scipy.optimize import newton

from vs import BaseVerticalStructure, Vars, IdealGasMixin, RadiativeTempGradient, BellLin1994TwoComponentOpacityMixin

sigmaSB = const.sigma_sb.cgs.value
sigmaT = const.sigma_T.cgs.value
mu = const.u.cgs.value
c = const.c.cgs.value
pl_const = const.h

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
    def dlnTdlnP(self, y, t):
        rho, eos = self.mesaop.rho(y[Vars.P] * self.P_norm, y[Vars.T] * self.T_norm, full_output=True)
        return eos.grad_ad


class FirstAssumptionRadiativeConvectiveGradient:
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
    def dlnTdlnP(self, y, t):
        rho, eos = self.rho(y, True)
        varkappa = self.opacity(y, lnfree_e=eos.lnfree_e)

        if t == 1:
            dlnTdlnP_rad = - self.dQdz(y, t) * (y[Vars.P] / y[Vars.T] ** 4) * 3 * varkappa * (
                    self.Q_norm * self.P_norm / self.T_norm ** 4) / (16 * sigmaSB * self.z0 * self.omegaK ** 2)
        else:
            # dTdz = (abs(y[Vars.Q] + self.Q_adv(y[Vars.P] * self.P_norm) / self.Q_norm) / y[Vars.T] ** 3) * 3 * varkappa * rho * self.z0 * self.Q_norm / (
            #         16 * sigmaSB * self.T_norm ** 4)
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

        VV = -((3 + omega ** 2) / (
                3 * omega)) ** 2 * eos.c_p ** 2 * rho ** 2 * H_ml ** 2 * self.omegaK ** 2 * self.z0 * (1 - t) / (
                     512 * sigmaSB ** 2 * y[Vars.T] ** 6 * self.T_norm ** 6 * H_p) * der * (
                     dlnTdlnP_rad - eos.grad_ad)
        V = 1 / np.sqrt(VV)

        coeff = [2 * A, V, V ** 2, - V]
        # print(coeff)
        try:
            x = np.roots(coeff)
        except np.linalg.LinAlgError:
            print('LinAlgError')
            breakpoint()

        x = [a.real for a in x if a.imag == 0 and 0.0 < a.real < 1.0]
        if len(x) != 1:
            print('not one x of there is no right x')
            breakpoint()
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

    def k_d_nu(self, nu):  # cross-section in cm2
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

    def J_tot(self, F_nu, y, tau_0):
        sigma = 0.34
        k_d_nu = 2
        tau = (sigma + k_d_nu) * y[Vars.S] * self.sigma_norm
        lamb = sigma / (sigma + k_d_nu)
        k = np.sqrt(3 * (1 - lamb))
        zeta_0 = np.cos(self.theta_irr)

        D_nu = 3 * lamb * zeta_0 ** 2 / (1 - k ** 2 * zeta_0 ** 2)

        C_nu = D_nu * (1 + np.exp(-tau_0 / zeta_0) + 2 / (3 * zeta_0) * (1 + np.exp(-tau_0 / zeta_0))) / (
                1 + np.exp(-k * tau_0) + 2 * k / 3 * (1 + np.exp(-k * tau_0)))

        J_tot = F_nu / (4 * np.pi) * (
                C_nu * (np.exp(-k * tau) + np.exp(-k * (tau_0 - tau))) +
                (1 - D_nu) * (np.exp(-tau / zeta_0) + np.exp(-(tau_0 - tau) / zeta_0))
        )

        return J_tot

    def H_tot(self, F_nu, tau_0):
        sigma = 0.34
        k_d_nu = 2
        tau = (sigma + k_d_nu) * 0.1
        # tau = (sigma + k_d_nu) * self.P_ph() / (self.z0 * self.omegaK ** 2)
        lamb = sigma / (sigma + k_d_nu)
        k = np.sqrt(3 * (1 - lamb))
        zeta_0 = np.cos(self.theta_irr)

        D_nu = 3 * lamb * zeta_0 ** 2 / (1 - k ** 2 * zeta_0 ** 2)

        C_nu = D_nu * (1 + np.exp(-tau_0 / zeta_0) + 2 / (3 * zeta_0) * (1 + np.exp(-tau_0 / zeta_0))) / (
                1 + np.exp(-k * tau_0) + 2 * k / 3 * (1 + np.exp(-k * tau_0)))

        H_tot = F_nu * (
                k * C_nu / 3 * (np.exp(-k * tau) - np.exp(-k * (tau_0 - tau))) +
                (zeta_0 - D_nu / (3 * zeta_0)) * (np.exp(-tau / zeta_0) - np.exp(-(tau_0 - tau) / zeta_0))
        )

        return H_tot

    def epsilon(self, y):
        rho = self.rho(y)
        tau_0 = np.infty
        k_d_nu = 2
        epsilon = 4 * np.pi * rho * simps(k_d_nu * self.J_tot(self.F_nu_irr, y, tau_0), self.nu_irr)
        return epsilon

    def Q_irr_ph(self):
        tau_0 = np.infty
        Qirr = simps(self.H_tot(self.F_nu_irr, tau_0), self.nu_irr)
        return Qirr

    def Q_initial(self):
        result = 1 + self.Q_irr_ph() / self.Q_norm
        # print('Q_irr, Q_norm =', self.Q_irr_ph(), self.Q_norm)
        # print('Initial =', result)
        return result

    def dQdz(self, y, t):
        w_r_phi = self.viscosity(y)
        result = -(3 / 2) * self.z0 * self.omegaK * w_r_phi / self.Q_norm - self.epsilon(y) * self.z0 / self.Q_norm
        # print('dQdz =', result, -(3 / 2) * self.z0 * self.omegaK * w_r_phi / self.Q_norm, self.epsilon(y))
        return result


class ExternalIrradiationZeroAssumption:
    def Q_initial(self):
        return 1 + self.Q_irr / self.Q_norm


class MesaVerticalStructure(MesaGasMixin, MesaOpacityMixin, RadiativeTempGradient, BaseMesaVerticalStructure):
    pass


class MesaIdealVerticalStructure(IdealGasMixin, MesaOpacityMixin, RadiativeTempGradient, BaseMesaVerticalStructure):
    pass


class MesaVerticalStructureAdiabatic(MesaGasMixin, MesaOpacityMixin, AdiabaticTempGradient, BaseMesaVerticalStructure):
    pass


class MesaVerticalStructureFirstAssumption(MesaGasMixin, MesaOpacityMixin, FirstAssumptionRadiativeConvectiveGradient,
                                           BaseMesaVerticalStructure):
    pass


class MesaVerticalStructureRadConv(MesaGasMixin, MesaOpacityMixin, RadConvTempGradient, BaseMesaVerticalStructure):
    pass


class MesaVerticalStructureRadConvExternalIrradiation(MesaGasMixin, MesaOpacityMixin, RadConvTempGradient,
                                                      ExternalIrradiation, BaseMesaVerticalStructure):
    def __init__(self, Mx, alpha, r, F, nu_irr, F_nu_irr, theta_irr=None, eps=1e-5, abundance='solar'):
        super().__init__(Mx, alpha, r, F, eps=eps, mu=0.6, abundance=abundance)
        self.nu_irr = nu_irr
        self.F_nu_irr = F_nu_irr
        if theta_irr is None:
            self.theta_irr = np.arccos(1 / 8 * (self.z0 / self.r))
        else:
            self.theta_irr = theta_irr


class MesaVerticalStructureRadConvExternalIrradiationZeroAssumption(MesaGasMixin, MesaOpacityMixin, RadConvTempGradient,
                                                                    ExternalIrradiationZeroAssumption,
                                                                    BaseMesaVerticalStructure):
    def __init__(self, Mx, alpha, r, F, eps=1e-5, abundance='solar'):
        super().__init__(Mx, alpha, r, F, eps=eps, mu=0.6, abundance=abundance)
        h = np.sqrt(self.GM * self.r)
        albedo = 0.5
        eta_accr = 0.1
        rg = 2 * self.GM / c ** 2
        r_in = 3 * rg
        func = 1 - np.sqrt(r_in / r)
        Mdot = self.F / (h * func)
        q = 1 / 8
        C_irr = (1 - albedo) * q * self.z0 / self.r
        self.Q_irr = C_irr * eta_accr * Mdot * c ** 2 / (4 * np.pi * self.r ** 2)

        self.Teff = ((self.Q0 + self.Q_irr) / sigmaSB) ** (1 / 4)
        self.Tirr = (self.Q_irr / sigmaSB) ** (1 / 4)
        self.Tvis = (self.Q0 / sigmaSB) ** (1 / 4)


class MesaVerticalStructureRadConvAdvection(MesaGasMixin, MesaOpacityMixin, RadConvTempGradient, Advection,
                                            BaseMesaVerticalStructure):
    pass


class MesaVerticalStructureAdvection(MesaGasMixin, MesaOpacityMixin, RadiativeTempGradient, Advection,
                                     BaseMesaVerticalStructure):
    pass


class MesaIdealVerticalStructureAdvection(IdealGasMixin, MesaOpacityMixin, RadiativeTempGradient, Advection,
                                          BaseMesaVerticalStructure):
    pass
