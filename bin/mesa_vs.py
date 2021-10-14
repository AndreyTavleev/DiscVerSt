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
import numpy as np
from astropy import constants as const

from vs import BaseVerticalStructure, Vars, IdealGasMixin, RadiativeTempGradient, BellLin1994TwoComponentOpacityMixin
from vs import Prad

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


class RadConvTempGradientPrad:
    def dlnTdlnP(self, y, t):
        rho, eos = self.rho(y, True)
        varkappa = self.opacity(y, lnfree_e=eos.lnfree_e)

        if t == 1:
            dTdz_der = (self.dQdz(y, t) / y[Vars.T] ** 3) * 3 * varkappa * rho * self.z0 * self.Q_norm / (
                    16 * sigmaSB * self.T_norm ** 4)
            A_der = - rho * self.omegaK ** 2 * self.z0 ** 2 / self.P_norm
            B = 16 * sigmaSB / (3 * c) * self.T_norm ** 4 * y[Vars.T] ** 3 / self.P_norm
            dPdz = A_der - B * dTdz_der
            dlnTdlnP_rad = (y[Vars.P] / y[Vars.T]) * (dTdz_der / dPdz)
            # if dlnTdlnP_rad < 0.0:
            #     print('t = 1, ', dlnTdlnP_rad)
        else:
            dTdz = (abs(y[Vars.Q]) / y[Vars.T] ** 3) * 3 * varkappa * rho * self.z0 * self.Q_norm / (
                    16 * sigmaSB * self.T_norm ** 4)

            A = rho * (1 - t) * self.omegaK ** 2 * self.z0 ** 2 / self.P_norm
            B = 16 * sigmaSB / (3 * c) * self.T_norm ** 4 * y[Vars.T] ** 3 / self.P_norm
            dPdz = A - B * dTdz
            dlnTdlnP_rad = (y[Vars.P] / y[Vars.T]) * (dTdz / dPdz)

            # if dlnTdlnP_rad < 0.0:
            #     print('t = {}, '.format(t), dlnTdlnP_rad)

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


class MesaVerticalStructureRadConvPrad(MesaGasMixin, MesaOpacityMixin, RadConvTempGradientPrad, Prad,
                                       BaseMesaVerticalStructure):
    pass
