from vs import BaseVerticalStructure, Vars, IdealGasMixin, RadiativeTempGradient
import numpy as np
from scipy.optimize import root_scalar
from astropy import constants as const

sigmaSB = const.sigma_sb.cgs.value

try:
    from opacity import Opac
except ModuleNotFoundError:
    print('Mesa2py is not installed')
    exit(22)

opacity = Opac('solar', mesa_dir='/mesa')
# opacity = Opac({b'he4': 0.25, b'h1': 0.75}, mesa_dir='/mesa')
# opacity = Opac({b'he4': 1.0}, mesa_dir='/mesa')


class MesaGasMixin:
    law_of_rho = opacity.rho


class MesaOpacityMixin:
    law_of_opacity = opacity.kappa


class AdiabaticTempGradient:
    def dlnTdlnP(self, y, t):
        rho_ad, eos = opacity.rho(y[Vars.P] * self.P_norm, y[Vars.T] * self.T_norm, True)
        return eos.grad_ad


class FirstAssumptionRadiativeConvectiveGradient:
    def dlnTdlnP(self, y, t):
        xi = self.opacity(y)

        if t == 1:
            dlnTdlnP_rad = - self.dQdz(y, t) * (y[Vars.P] / y[Vars.T] ** 4) * 3 * xi * (
                    self.Q_norm * self.P_norm / self.T_norm ** 4) / (16 * sigmaSB * t * self.omegaK ** 2)
        else:
            rho = self.rho(y)
            dTdz = ((abs(y[Vars.Q]) / y[Vars.T] ** 3) * 3 * xi * rho * self.z0 * self.Q_norm / (
                    16 * sigmaSB * self.T_norm ** 4))
            dPdz = rho * (1 - t) * self.omegaK ** 2 * self.z0 ** 2 / self.P_norm

            dlnTdlnP_rad = (y[Vars.P] / y[Vars.T]) * (dTdz / dPdz)

        rho_ad, eos = opacity.rho(y[Vars.P] * self.P_norm, y[Vars.T] * self.T_norm, True)

        if dlnTdlnP_rad < eos.grad_ad:
            return dlnTdlnP_rad
        else:
            return eos.grad_ad


class RadConvTempGradient:
    def dlnTdlnP(self, y, t):
        xi = self.opacity(y)
        rho = self.rho(y)

        if t == 1:
            dlnTdlnP_rad = - self.dQdz(y, t) * (y[Vars.P] / y[Vars.T] ** 4) * 3 * xi * (
                    self.Q_norm * self.P_norm / self.T_norm ** 4) / (16 * sigmaSB * t * self.omegaK ** 2)
        else:
            dTdz = ((abs(y[Vars.Q]) / y[Vars.T] ** 3) * 3 * xi * rho * self.z0 * self.Q_norm / (
                    16 * sigmaSB * self.T_norm ** 4))
            dPdz = rho * (1 - t) * self.omegaK ** 2 * self.z0 ** 2 / self.P_norm

            dlnTdlnP_rad = (y[Vars.P] / y[Vars.T]) * (dTdz / dPdz)

        rho_ad, eos = opacity.rho(y[Vars.P] * self.P_norm, y[Vars.T] * self.T_norm, True)
        alpha_ml = 1.5
        H_p = y[Vars.P] * self.P_norm / (rho * self.omegaK ** 2 * (1 - t) + np.sqrt(y[Vars.P] * self.P_norm * rho))
        H_ml = alpha_ml * H_p
        omega = xi * rho * H_ml
        A = 9 / 8 * omega ** 2 / (3 + omega ** 2)
        VV = -((3 + omega ** 2) / (3 * omega ** 2)) ** 2 * eos.c_p ** 2 * rho ** 2 * H_ml ** 2 * self.omegaK ** 2 * (
                1 - t) / (
                     512 * sigmaSB ** 2 * y[Vars.T] ** 6 * self.T_norm ** 6 * H_p) * eos.dlnRho_dlnT_const_Pgas * (
                     dlnTdlnP_rad - eos.grad_ad)
        V = VV ** (-1 / 2)

        y = root_scalar(lambda x: 2 * A * x ** 3 + V * x ** 2 + V ** 2 * x - V, bracket=[-1, 1]).root

        dlnTdlnP_conv = eos.grad_ad + (dlnTdlnP_rad - eos.grad_ad) * y * (y + V)

        if dlnTdlnP_rad < eos.grad_ad:
            return dlnTdlnP_rad
        else:
            return dlnTdlnP_conv


class MesaVerticalStructure(MesaGasMixin, MesaOpacityMixin, RadiativeTempGradient, BaseVerticalStructure):
    pass


class MesaIdealVerticalStructure(IdealGasMixin, MesaOpacityMixin, RadiativeTempGradient, BaseVerticalStructure):
    pass


class MesaVerticalStructureAdiabatic(MesaGasMixin, MesaOpacityMixin, AdiabaticTempGradient, BaseVerticalStructure):
    pass


class MesaVerticalStructureFirstAssumption(MesaGasMixin, MesaOpacityMixin, FirstAssumptionRadiativeConvectiveGradient,
                                           BaseVerticalStructure):
    pass


class MesaVerticalStructureRadConv(MesaGasMixin, MesaOpacityMixin, RadConvTempGradient, BaseVerticalStructure):
    pass
