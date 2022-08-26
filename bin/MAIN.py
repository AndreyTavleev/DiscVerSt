import sys
from datetime import datetime

import numpy as np
from astropy import constants as const
from astropy.io import ascii
from scipy.integrate import simps
from astropy import units

from plots_vs import S_curve, Structure_Plot, Radial_Plot
import mesa_vs

G = const.G.cgs.value
M_sun = const.M_sun.cgs.value
R_sun = const.R_sun.cgs.value
c = const.c.cgs.value
pl_const = const.h.cgs.value
k_B = const.k_B.cgs.value
sigmaSB = const.sigma_sb.cgs.value
pl_const_unit = const.h


def plank(T, nu):
    coeff = (2 * pl_const * nu ** 3 / c ** 2) / np.expm1(pl_const * nu / (k_B * T))
    # print(coeff)
    return coeff


def main():
    M = 1.4 * M_sun  # 1.4 * M_sun
    alpha = 0.1  # 0.5
    time0 = datetime.now()
    nu_irr = np.geomspace(7.3e15, 2.4e18, 30)  # from 0.03 to 10 keV
    # nu_irr = 2.4e18
    # nu_irr = 2.4e16
    # nu_irr = 2.4e17
    eta_accr = 0.1
    Mdot_edd = 1.39e18 * M / M_sun

    E = (pl_const_unit * nu_irr * units.Hz).to('keV').value
    spectra = (E / 8) ** (-0.4) * np.exp(-E / 8) / simps((E / 8) ** (-0.4) * np.exp(-E / 8), E) * \
              (pl_const_unit * units.Hz).to('keV') / units.keV  # spectra as f(nu), not f(E/keV)
    print(spectra)

    # M = 1.4 * M_sun
    # # for T_irr in np.array([0, 1, 3, 6, 9, 12, 20, 60]) * 1e3:
    # for T_irr in np.array([0, 3, 9, 11, 12, 20]) * 1e3:
    #     # Structure_Plot(M, alpha, r, 8700, input='Teff',  # 8700 4500
    #     #                # structure='MesaRadConvIrr',
    #     #                structure='MesaRadConvIrrZero',
    #     #                abundance='solar', n=300,
    #     #                path_dots='fig/Tuchman_etal1989/vs_Tirr{:g}_zero_Q_initial_10Msun_upper.dat'.format(T_irr),
    #     #                add_Pi_values=True, make_pic=False,
    #     #                T_irr=T_irr)
    #     r = 3e10
    #     S_curve(2e3, 3e4, M, alpha, r, input='Teff',
    #             structure='MesaRadConvIrrZero',
    #             abundance='solar', n=100,
    #             tau_break=True,
    #             path_dots='fig/Dubus_etal1998/S-curve_irr_Tirr{}_zero_test_with_Sigma_ph.dat'.format(T_irr),
    #             add_Pi_values=True, make_pic=False,
    #             T_irr=T_irr)
    #
    # print(datetime.now() - time0)
    # return

    # for Mdot in np.array([0.1, 1, 50, 100, 700, 1000, 7000, 1e4, 4e4, 7e4, 1e5, 3e5]) * Mdot_edd:
    # for Mdot in np.array([0.1, 1, 50]) * Mdot_edd:
    #     for r in [3e10]:
    #         F_nu_irr = eta_accr * Mdot * c ** 2 / (4 * np.pi * r ** 2)
    #         Structure_Plot(M, alpha, r, 1, input='Mdot_Mdot_edd',  # lower = 1e-2
    #                        structure='MesaRadConv',
    #                        abundance='solar', n=300,
    #                        path_dots='fig/Mescheryakov_etal2011/vs_upper_non_irr.dat',
    #                        add_Pi_values=True, make_pic=False)
    #         Structure_Plot(M, alpha, r, 1, input='Mdot_Mdot_edd',  # lower = 1e-2
    #                        structure='MesaRadConvIrr',
    #                        # structure='MesaRadConvIrrZero',
    #                        abundance='solar', n=300,
    #                        path_dots='fig/Mescheryakov_etal2011/vs_upper_{:1.1e}_edd_irr.dat'.format(Mdot / Mdot_edd),
    #                        add_Pi_values=True, make_pic=False,
    #                        nu_irr=nu_irr, F_nu_irr=F_nu_irr)
    # for Mdot in np.array([1]) * Mdot_edd:
    # for Mdot in np.array([1, 10, 50, 100, 1000]) * Mdot_edd:
    # for Mdot in np.array([10, 50]) * Mdot_edd:
    # for Mdot in np.array([1, 1e-1]) * Mdot_edd:
    for Mdot in np.array([1e-1]) * Mdot_edd:
    # for Mdot in np.array([1e-2, 1e-1, 1]) * Mdot_edd:
        r = 3e10
        F_nu_irr = eta_accr * Mdot * c ** 2 / (4 * np.pi * r ** 2) * spectra
        # F_nu_irr = eta_accr * Mdot * c ** 2 / (4 * np.pi * r ** 2) * np.ones(nu_irr.size) / (nu_irr[-1] - nu_irr[0])
        print(F_nu_irr)
        S_curve(1e-4, 1, M, alpha, r, input='Mdot_Mdot_edd',  # 1e-5, 1
                structure='MesaRadConvIrr',
                abundance='solar', n=100,
                tau_break=True,
                path_dots='fig/Mescheryakov_etal2011/S-curve_Dubus_etal1998_{}edd_debug_z01_'
                          'spectra_1.dat'.format(Mdot / Mdot_edd),
                add_Pi_values=True, make_pic=False,
                nu_irr=nu_irr, F_nu_irr=F_nu_irr,
                cos_theta_irr=1/80)
        # Radial_Plot(M, alpha, r_start=1e9, r_end=1e12, Par=1, input='Mdot_Mdot_edd',
        #             structure='MesaRadConvIrr',
        #             abundance='solar', n=100,
        #             tau_break=True,
        #             path_dots='fig/Dubus_etal1998/r-plot_edd_1keV.dat',
        #             add_Pi_values=True,
        #             nu_irr=nu_irr, F_nu_irr=F_nu_irr)
        #             cos_theta_irr=1/40)
        # Radial_Plot(M, alpha, r_start=1e9, r_end=1e12, Par=1, input='Mdot_Mdot_edd',
        #             structure='MesaRadConv',
        #             abundance='solar', n=100,
        #             tau_break=True,
        #             path_dots='fig/Dubus_etal1998/r-plot.dat',
        #             add_Pi_values=True)
        # S_curve(1e-5, 1, M, alpha, r, input='Mdot_Mdot_edd',
        #         structure='MesaRadConvIrr',
        #         abundance='solar', n=100,
        #         tau_break=True,
        #         path_dots='fig/Mescheryakov_etal2011/S-curve_Dubus_etal1998.dat',
        #         add_Pi_values=True, make_pic=False,
        #         nu_irr=nu_irr, F_nu_irr=None)
        # S_curve(1e-5, 1, M, alpha, r, input='Mdot_Mdot_edd',
        #         structure='MesaRadConv',
        #         abundance='solar', n=100,
        #         tau_break=True,
        #         path_dots='fig/Mescheryakov_etal2011/S-curve_Dubus_etal1998_non_irr.dat',
        #         add_Pi_values=True, make_pic=False)
    print(datetime.now() - time0)
    return

    for T_irr in np.array([0, 1, 3, 6, 9, 12, 20, 60]) * 1e3:
        # Structure_Plot(M, alpha, r, 8700, input='Teff',  # 8700 4500
        #                # structure='MesaRadConvIrr',
        #                structure='MesaRadConvIrrZero',
        #                abundance='solar', n=300,
        #                path_dots='fig/Tuchman_etal1989/vs_Tirr{:g}_zero_Q_initial_upper.dat'.format(T_irr),
        #                add_Pi_values=True, make_pic=False,
        #                T_irr=T_irr)
        S_curve(2e3, 3e4, M, alpha, r, input='Teff',
                # structure='MesaRadConvIrr',
                structure='MesaRadConvIrrZero',
                abundance='solar', n=100,
                tau_break=True,
                path_dots='fig/Dubus_etal1998/S-curve_irr_Tirr{}_zero_Q_initial.dat'.format(T_irr),
                add_Pi_values=True, make_pic=False,
                T_irr=T_irr)

    M = 10 * M_sun
    for T_irr in np.array([0, 1, 3, 6, 9, 12, 20, 60]) * 1e3:
        # Structure_Plot(M, alpha, r, 8700, input='Teff',  # 8700 4500
        #                # structure='MesaRadConvIrr',
        #                structure='MesaRadConvIrrZero',
        #                abundance='solar', n=300,
        #                path_dots='fig/Tuchman_etal1989/vs_Tirr{:g}_zero_Q_initial_10Msun_upper.dat'.format(T_irr),
        #                add_Pi_values=True, make_pic=False,
        #                T_irr=T_irr)
        S_curve(2e3, 3e4, M, alpha, r, input='Teff',
                # structure='MesaRadConvIrr',
                structure='MesaRadConvIrrZero',
                abundance='solar', n=100,
                tau_break=True,
                path_dots='fig/Dubus_etal1998/S-curve_irr_Tirr{}_zero_Q_initial_10Msun.dat'.format(T_irr),
                add_Pi_values=True, make_pic=False,
                T_irr=T_irr)

    print(datetime.now() - time0)
    raise Exception

    # nu_irr = 7.3e15
    # nu_irr = 2.4e18
    # nu_irr = np.geomspace(7.3e15, 2.4e18, 100)  # from 0.03 to 10 keV
    # nu_irr = np.linspace(2.4e17, 2.4e18, 2)
    # eta_accr = 0.1
    # E = (const.h * nu_irr * Hz).to('keV').value
    # spectra = (E / 7.5) ** (-0.4) * np.exp(-E/7.5)
    # T_irr = 6e4
    # F_nu_irr = sigmaSB * T_irr ** 4 * np.pi * plank(T_irr, nu_irr)
    # F_nu_irr = sigmaSB * T_irr ** 4
    # Mdot = 7e16

    # Q_irr = sigmaSB * T_irr ** 4
    # C_irr = Q_irr / (0.1 * Mdot * c ** 2) * (4 * np.pi * r ** 2)

    # F_nu_irr = eta_accr * Mdot * c ** 2 / (4 * np.pi * r ** 2)

    for i, nu_irr in enumerate(np.array([1, 2, 3, 6, 10]) * 2.4e17):
        # if i in [0, 1]:
        #     continue
        # for i, nu_irr in enumerate(np.array([10]) * 2.4e17):
        print('i = ', i)
        for T_irr in np.array([1, 3, 6, 10, 20, 60]) * 1e3:
            # for T_irr in np.array([6, 7, 8]) * 1e5:
            # Q_irr = sigmaSB * T_irr ** 4
            F_nu_irr = sigmaSB * T_irr ** 4
            # C_irr = Q_irr / (0.1 * Mdot * c ** 2) * (4 * np.pi * r ** 2)
            S_curve(2e3, 3e4, M, alpha, r, input='Teff',
                    # structure='MesaRadConvIrr',
                    structure='BellLinIrr',
                    abundance='solar', n=100,
                    tau_break=True, path_dots='fig/S-curve_irr_Tirr{}_{}_BellLin.dat'.format(T_irr, i),
                    add_Pi_values=True, make_pic=False,
                    # C_irr=C_irr)
                    nu_irr=nu_irr, F_nu_irr=F_nu_irr)
            # Structure_Plot(M, alpha, r, 2e4, input='Teff', structure='MesaRadConvIrr',
            #                path_dots='fig/vvss_Tirr{:g}_{}.dat'.format(T_irr, i),
            #                make_pic=False, nu_irr=nu_irr, F_nu_irr=F_nu_irr)
            #                # make_pic=False, C_irr=C_irr)
    # S_curve(2e3, 3e4, M, alpha, r, input='Teff', structure='MesaRadConv', abundance='solar', n=300,
    #         tau_break=True, path_dots='fig/S-curve_non_irr.dat',
    #         add_Pi_values=True, make_pic=False)

    # for T_irr in [1e4, 4e4, 6e4]:
    #     nu_irr = 2.4e18
    #     F_nu_irr = sigmaSB * T_irr ** 4
    #     # Mdot = 7e16
    #     # Q_irr = sigmaSB * T_irr ** 4
    #     # C_irr = Q_irr / (0.1 * Mdot * c ** 2) * (4 * np.pi * r ** 2)
    #     Structure_Plot(M, alpha, r, 2e4, input='Teff', structure='MesaRadConvIrr',
    #                    path_dots='fig/vvss_Tirr_{:g}_2e4.dat'.format(T_irr),
    #                    make_pic=False, nu_irr=nu_irr, F_nu_irr=F_nu_irr)
    #                    # make_pic=False, C_irr=C_irr)
    # Structure_Plot(M, alpha, r, 2e4, input='Teff', structure='MesaRadConvIrr',
    #                path_dots='fig/vvss_145713645.dat',
    #                make_pic=False, nu_irr=nu_irr, F_nu_irr=F_nu_irr)
    # make_pic=False, C_irr=C_irr)
    # Structure_Plot(M, alpha, r, 2e4, input='Teff', structure='MesaRadConv', path_dots='fig/vvss_Tirr_0_2e4.dat',
    #                make_pic=False)

    print(datetime.now() - time0)
    raise Exception

    # from opacity import Opac
    # mesaop = Opac({'h1': 1.0})
    #
    # data = ascii.read('S-curve_test_test2.dat')
    # P = data['P_c']
    # T = data['T_c']
    #
    # grad = np.zeros(len(P))
    # for i in range(len(grad)):
    #     rho, eos = mesaop.rho(P[i], T[i], full_output=True)
    #     grad[i] = eos.grad_ad
    # np.savetxt('fig/grad_ad_test2.dat', grad)
    # raise Exception

    # Structure_Plot(M, alpha, r, 1, input='Mdot_Mdot_edd', structure='MesaRadConv', abundance='solar', n=100,
    #                add_Pi_values=True, path_dots='fig/vs_non_irr.dat', make_pic=False)
    # Structure_Plot(M, alpha, r, 1, input='Mdot_Mdot_edd', structure='MesaRadConvIrr', abundance='solar', n=100,
    #                add_Pi_values=True, path_dots='fig/vs_irr_max_nu.dat', make_pic=False)
    # Structure_Plot(M, alpha, r, 1, input='Mdot_Mdot_edd', structure='MesaRadConvIrrZero', abundance='solar', n=100,
    #                add_Pi_values=True, path_dots='fig/vs_irr_zero.dat', make_pic=False)

    # S_curve(6e3, 6e4, M, alpha, r, input='Teff', structure='MesaRadConv', abundance='solar', n=400, tau_break=True,
    #         path_dots='fig/S-curve_non_irr.dat', add_Pi_values=True, make_pic=False)
    # S_curve(6e3, 6e4, M, alpha, r, input='Teff', structure='MesaRadConvIrr', abundance='solar', n=400, tau_break=True,
    #         path_dots='fig/S-curve_irr.dat', add_Pi_values=True, make_pic=False)
    # S_curve(6e3, 6e4, M, alpha, r, input='Teff', structure='MesaRadConvIrrZero', abundance='solar', n=400,
    #         tau_break=True, path_dots='fig/S-curve_irr_zero.dat', add_Pi_values=True, make_pic=False)

    # rg = 2 * G * M / c ** 2
    # r = 50 * rg
    # mmdot = 1e-10
    # Mdot = mmdot * M_sun / 3.154e7
    S_curve(107700, 1.2e5, M, alpha, r, input='Teff', structure='MesaPrad', abundance='solar', n=10,
            tau_break=True, path_dots='fig/S-curve_test_test3.dat',
            add_Pi_values=True, make_pic=False)
    # # Structure_Plot(M, alpha, r, 2e4, input='Teff', structure='MesaRadConv', abundance={'he4': 1.0}, n=100,
    # #                add_Pi_values=True, path_dots='fig/vs_test_test.dat', make_pic=False)
    #
    # print(datetime.now() - time0)
    # raise Exception

    # for r in np.geomspace(6500 * rg, 1e5 * rg, 20):
    #     S_curve(1e14, 2e17, M, alpha, r, input='Mdot', structure='MesaRadConv', abundance='solar', n=200,
    #             tau_break=True, path_dots='fig/S-curve_Shakura_stab{:g}.dat'.format(r/rg),
    #             add_Pi_values=True, make_pic=False)

    # for mmdot in [1e-7, 1e-8, 1e-9, 1e-10, 1e-11]:
    # for mmdot in [1e-7, 1e-8, 1e-9]:
    # for mmdot in [1e-11]:
    #     Mdot = mmdot * M_sun / 3.154e7
    #     rg = 2 * G * M / c ** 2
    #     alpha = 0.5
    #     Radial_Plot(M, alpha, 3.1 * rg, 1e5 * rg, Mdot, input='Mdot', structure='MesaRadConv', abundance={'he4': 1.0},
    #                 n=300, tau_break=True, path_dots='fig/r_table_Shakura_{:g}_he4.dat'.format(mmdot))
    # 3.1 * rg, 1e5 * rg 300
    # 1e5 * rg, 1e7 * rg

    print(datetime.now() - time0)
    raise Exception

    # M = 1.4 * M_sun
    # # alpha = 0.74  # alpha_hot
    # alpha = 0.01  # alpha_cold
    #
    # T_C_plot, Teff_plot = [], []
    # F = np.loadtxt('fig/Aql_X-1/Aql-F-{}days_074end.txt'.format(days))
    # r = np.loadtxt('fig/Aql_X-1/Aql-r-{}days_074end.txt'.format(days))
    #
    # # if days == 6:
    # #     cold_index = 182
    # # if days == 45:
    # #     cold_index = 186
    # # if days == 55:
    # #     cold_index = 134
    #
    # for i, rr in enumerate(r):
    #     print(i)
    #     # if i > cold_index:
    #     #     alpha = 0.01  # alpha_cold
    #     vs = MesaVerticalStructureRadConv(M, alpha, rr, F[i])
    #     vs.fit()
    #     varkappa_C, rho_C, T_C, P_C, Sigma0 = vs.parameters_C()
    #     T_C_plot.append(T_C)
    #     Teff_plot.append(vs.Teff)
    #
    # np.savetxt('fig/Aql_X-1/Aql-T_C-{}days_074end.txt'.format(days), T_C_plot)
    # np.savetxt('fig/Aql_X-1/Aql-Teff-{}days_074end.txt'.format(days), Teff_plot)
    # return

    # sigmaT = const.sigma_T.cgs.value
    # mu = const.u.cgs.value
    # from opacity import Opac
    # mesaop = Opac('solar')
    # for i in range(335):
    #     if i < 107:
    #         continue
    #     P = np.loadtxt('fig/lnfree_e/P_plot_lnfree_e_{}.txt'.format(i))
    #     T = np.loadtxt('fig/lnfree_e/T_plot_lnfree_e_{}.txt'.format(i))
    #     integral_plot = []
    #     for j in range(len(P)):
    #         rho, eos = mesaop.rho(P[j], T[j], full_output=True)
    #         kappa = mesaop.kappa(rho, T[j], lnfree_e=eos.lnfree_e)
    #         integral_plot.append(np.sqrt((kappa - sigmaT / mu * np.exp(eos.lnfree_e))/kappa))
    #
    #     root = newton(func, x0=1, args=(integral_plot,))
    #     print(root)

    # tau_eff = simps(integral_plot, np.linspace(0, 1, len(P)))
    #
    #
    # if i not in [235, 320, 321, 322, 323, 324, 325, 326, 327]:
    #     print(tau_eff)
    # else:
    #     print(tau_eff, i)
    # raise Exception

    # M = 6 * M_sun
    # alpha = 0.01
    # rg = 2 * G * M / c ** 2
    # r = 100 * rg
    # F_plot = ascii.read('fig/Approx_formulas/diploma1.dat')['F']
    # nomer = -1
    # for F in F_plot:
    #     print(nomer)
    #     vs = MesaVerticalStructureRadConv(M, alpha, r, F)
    #     vs.fit()
    #     Pi = vs.Pi_finder()
    #     y = vs.parameters_C()
    #     delta = (4 * sigmaSB) / (3 * c) * y[2] ** 4 / y[3]
    #     string_for_Pi = str(Pi) + ' ' + str(delta) + ' ' + str(r / (3 * rg)) + ' ' + str(
    #         nomer) + ' ' + str(M / M_sun) + ' ' + str(alpha) + '\n'
    #     nomer -= 1
    #     with open('fig/Approx_formulas/Pi_param3.txt', 'a') as g:
    #         g.write(string_for_Pi)
    # raise Exception

    # time0 = datetime.now()
    # M = 6 * M_sun
    # alpha = 0.01
    # rg = 2 * G * M / c ** 2
    # r = 100 * rg
    # # Teff = 7e3
    # Teff = 1e4
    # # Teff = 1.3e4
    # # h = np.sqrt(G * M * r)
    # # F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Teff ** 4
    # abundance = {'h1': 1.0}
    # # abundance = 'solar'
    # # vs = MesaVerticalStructureRadConvPrad(M, alpha, r, F, abundance=abundance)
    # # print(vs.fit())
    # Structure_Plot(M, alpha, r, Teff, input='Teff', structure='Prad', abundance=abundance, n=500,
    #                path_output='fig/Approx_formulas/vs1e4_h1.dat', make_pic=False)
    # print(datetime.now() - time0)
    # raise Exception


if __name__ == '__main__':
    # for days in [45, 55]:
    #     main(days)
    main()
