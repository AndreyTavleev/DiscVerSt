import numpy as np
from astropy import constants as const
from scipy.integrate import solve_ivp, simps
from scipy.optimize import newton

from vs import BaseVerticalStructure, Vars, IdealGasMixin, RadiativeTempGradient

sigmaSB = const.sigma_sb.cgs.value
sigmaT = const.sigma_T.cgs.value
mu = const.u.cgs.value
c = const.c.cgs.value
pl_const = const.h.cgs.value
m_e = const.m_e.cgs.value
k_B = const.k_B.cgs.value

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


class AnotherPph:
    def P_ph(self):
        def func(tau_end):
            # print('Start')
            solution = solve_ivp(
                self.photospheric_pressure_equation,
                t_span=[0, tau_end], t_eval=np.linspace(0, tau_end, 100),
                y0=[1e-7 * self.P_norm], rtol=self.eps
            )
            # print('End')
            P_temp = solution.y[0]
            tau_temp = solution.t
            T_temp = self.Teff * (1 / 2 + 3 * tau_temp / 4) ** (1 / 4)
            integral_plot = []
            for j in range(len(P_temp)):
                rho_temp, eos_temp = self.law_of_rho(P_temp[j], T_temp[j], full_output=True)
                lnfree_e = eos_temp.lnfree_e
                kappa = self.law_of_opacity(rho_temp, T_temp[j], lnfree_e)

                e_fermi = pl_const ** 2 / (2 * m_e) * (3 / (8 * np.pi) * np.exp(lnfree_e) * rho_temp / mu) ** (2 / 3)
                degeneracy = (e_fermi - m_e * c ** 2) / (k_B * T_temp[j])
                psi = np.exp(0.8168 * degeneracy - 0.05522772 * degeneracy ** 2)
                kT_mc2 = k_B * T_temp[j] / (m_e * c ** 2)
                G_inverse = 1.129 + 0.2965 * psi - 0.005594 * psi ** 2 + (
                            11.47 + 0.3570 * psi + 0.1078 * psi ** 2) * kT_mc2 + (
                                        -3.249 + 0.1678 * psi - 0.04706 * psi ** 2) * kT_mc2 ** 2
                mesa_teor_kappa_sc = sigmaT / mu * np.exp(lnfree_e) / G_inverse

                integral_plot.append(np.sqrt((kappa - mesa_teor_kappa_sc) / kappa))

                # if (kappa - sigmaT / mu * np.exp(lnfree_e)) < 0:
                #     # breakpoint()
                #     integral_plot.append(np.sqrt(abs(kappa - sigmaT / mu * np.exp(lnfree_e)) / kappa))
                # else:
                #     integral_plot.append(np.sqrt((kappa - sigmaT / mu * np.exp(lnfree_e)) / kappa))
                print(kappa, sigmaT / mu * np.exp(lnfree_e), mesa_teor_kappa_sc, (kappa - mesa_teor_kappa_sc) / kappa)
                # kappa - sigmaT / mu * np.exp(lnfree_e),
                # (kappa - sigmaT / mu * np.exp(lnfree_e)) / kappa, lnfree_e, P_temp[j], T_temp[j],
                # rho_temp, 5e24 * rho_temp * T_temp[j] ** (-7 / 2))

            tau_eff = simps(integral_plot, tau_temp)
            # print('tau_eff = ', tau_eff, 2 / 3 - tau_eff)
            return 2 / 3 - tau_eff

        root, info = newton(func, x0=2 / 3, x1=2, full_output=True)
        # print('root = ', root)
        solution = solve_ivp(
            self.photospheric_pressure_equation,
            [0, root],
            [1e-7 * self.P_norm], rtol=self.eps
        )
        return solution.y[0][-1], root

    def initial(self):
        Q_initial = self.Q_initial()
        y = np.empty(4, dtype=np.float64)
        y[Vars.S] = 0
        P_ph, root = self.P_ph()
        print('root = ', root)
        y[Vars.P] = P_ph / self.P_norm
        y[Vars.Q] = Q_initial
        y[Vars.T] = self.Teff / self.T_norm * (1 / 2 + 3 * root / 4) ** (1 / 4)
        return y


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


class MesaVerticalStructureRadConvAnotherPph(MesaGasMixin, MesaOpacityMixin, RadConvTempGradient, AnotherPph,
                                             BaseMesaVerticalStructure):
    pass
