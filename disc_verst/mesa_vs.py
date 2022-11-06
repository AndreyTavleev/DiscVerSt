#!/usr/bin/env python3
"""
Module contains several classes that represent vertical structure of accretion disc in case
of (tabular) MESA opacity and(or) EOS.

Class MesaVerticalStructure --  for MESA opacities and EoS with radiative energy transport.
Class MesaIdealVerticalStructure -- for MESA opacities and ideal gas EoS with radiative energy transport.
Class MesaVerticalStructureAd -- for MESA opacities and EoS with adiabatic energy transport.
Class MesaVerticalStructureRadAd -- for MESA opacities and EoS with radiative+adiabatic energy transport.
Class MesaVerticalStructureRadConv -- for MESA opacities and EoS with radiative+convective energy transport.

"""
import os

import numpy as np
from astropy import constants as const

from disc_verst.vs import BaseVerticalStructure, Vars, IdealGasMixin, RadiativeTempGradient
from disc_verst.vs import BellLin1994TwoComponentOpacityMixin
from disc_verst.vs import Prad

sigmaSB = const.sigma_sb.cgs.value
sigmaT = const.sigma_T.cgs.value
mu = const.u.cgs.value
c = const.c.cgs.value
pl_const = const.h
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
            print('FUUUUUU!!!')

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
        varkappa = self.opacity(y, lnfree_e=eos.lnfree_e, return_grad=False)

        if t == 1:
            dTdz_der = (self.dQdz(y, t) / y[Vars.T] ** 3) * 3 * varkappa * rho * self.z0 * self.Q_norm / (
                    16 * sigmaSB * self.T_norm ** 4)
            P_full = y[Vars.P] * self.P_norm + 4 * sigmaSB / (3 * c) * y[Vars.T] ** 4 * self.T_norm ** 4
            dP_full_der = - rho * self.omegaK ** 2 * self.z0 ** 2
            dlnTdlnP_rad = (P_full / y[Vars.T]) * (dTdz_der / dP_full_der)
            # # if dlnTdlnP_rad < 0.0:
            # #     print('t = 1, ', dlnTdlnP_rad)
        else:
            dTdz = (abs(y[Vars.Q]) / y[Vars.T] ** 3) * 3 * varkappa * rho * self.z0 * self.Q_norm / (
                    16 * sigmaSB * self.T_norm ** 4)

            P_full = y[Vars.P] * self.P_norm + 4 * sigmaSB / (3 * c) * y[Vars.T] ** 4 * self.T_norm ** 4
            dP_full = rho * (1 - t) * self.omegaK ** 2 * self.z0 ** 2
            dlnTdlnP_rad = (P_full / y[Vars.T]) * (dTdz / dP_full)

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
        if der > 0:
            der = -1

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
            print('LinAlgError, coeff[2A, V, V ** 2, -V] = ', coeff)
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


class MesaVerticalStructureRadConvPrad(MesaGasMixin, MesaOpacityMixin, RadConvTempGradientPrad, Prad,
                                       BaseMesaVerticalStructure):
    pass


def main():
    M = 10 * M_sun
    alpha = 0.01
    Mdot = 1e17
    rg = 2 * G * M / c ** 2
    r = 400 * rg
    print('Calculating structure and making a structure plot. '
          '\nStructure with tabular MESA opacity and EOS.'
          '\nChemical composition is solar.')
    print('M = {:g} grams \nr = {:g} cm = {:g} rg '
          '\nalpha = {:g} \nMdot = {:g} g/s'.format(M, r, r / rg, alpha, Mdot))
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
    plt.title(r'$M = {:g}\, M_{{\odot}},\, \dot{{M}} = {:g}\, {{\rm g/s}},\, '
              r'\alpha = {:g}, r = {:g} \,\rm cm$'.format(M / M_sun, Mdot, alpha, r))
    plt.grid()
    plt.legend()
    os.makedirs('fig/', exist_ok=True)
    plt.savefig('fig/vs_mesa.pdf')
    print('Plot of structure is successfully saved to fig/vs_mesa.pdf.')
    return


if __name__ == '__main__':
    main()
