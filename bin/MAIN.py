import sys
from datetime import datetime

import numpy as np
from astropy import constants as const
from astropy.io import ascii
from matplotlib import rcParams
from scipy.integrate import simps

from plots import Structure_Plot, S_curve

rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['text.latex.preamble'] = r'\usepackage[utf8]{inputenc} \usepackage{amsfonts, amsmath, amsthm, amssymb} ' \
                                  r'\usepackage[english]{babel} '

G = const.G.cgs.value
M_sun = const.M_sun.cgs.value
R_sun = const.R_sun.cgs.value
c = const.c.cgs.value
pl_const = const.h.cgs.value
k_B = const.k_B.cgs.value
sigmaSB = const.sigma_sb.cgs.value


def plank(nu, T):
    a = 2 * pl_const * nu ** 3 / c ** 2
    return a / np.expm1(pl_const * nu / (k_B * T))


def func(tau_end, integral_plot):
    tau_eff = simps(integral_plot, np.linspace(0, tau_end, len(integral_plot)))
    return 1.0 - tau_eff


def main(days):
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

    # from opacity import Opac
    # mesaop = Opac('solar')
    # p = np.geomspace(1e-1, 1e28, 1000)
    # t = np.geomspace(1e2, 1e10, 900)
    # kappa = np.empty((900, 1000))
    # rho = np.empty((900, 1000))
    # lnfree_e = np.empty((900, 1000))
    # der = np.empty((900, 1000))
    # for i, tt in enumerate(t):
    #     for j, pp in enumerate(p):
    #         rho1, eos = mesaop.rho(pp, tt, full_output=True)
    #         kappa1 = mesaop.kappa(rho1, tt, lnfree_e=40.0)
    #         rho[i][j] = rho1
    #         kappa[i][j] = kappa1
    #         lnfree_e[i][j] = eos.lnfree_e
    #         der[i][j] = eos.dlnRho_dlnT_const_Pgas
    # np.savetxt('fig/rho_mesa.txt', rho)
    # np.savetxt('fig/lnfree_e_mesa.txt', lnfree_e)
    # np.savetxt('fig/kappa_40_mesa.txt', kappa)
    # np.savetxt('fig/der_mesa.txt', der)
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
    # Structure_Plot(M, alpha, r, Teff, input='Teff', structure='Prad', abundance=abundance, n=500, savedots=True,
    #                path_output='fig/Approx_formulas/vs1e4_h1.dat', make_pic=False)
    # print(datetime.now() - time0)
    # raise Exception

    time0 = datetime.now()
    M = 6 * M_sun  # 6 * M_sun
    alpha = 0.01  # 0.5
    rg = 2 * G * M / c ** 2
    r = 100 * rg  # 100 * rg
    S_curve(6e3, 4e4, M, alpha, r, structure='BellLin', abundance='solar', n=400, input='Teff',
            savedots=True, save_indexes=True,
            path_output='fig/Approx_formulas/diploma5.dat', path_indexes='fig/Approx_formulas/diploma5_indexes.dat',
            make_Pi_table=True, Pi_table_path='fig/Approx_formulas/Pi_param_diploma5.txt', make_pic=False)  # 2e3 1e7 400
    print(datetime.now() - time0)
    raise Exception

    Sigma_plot, z0r_plot, T_C_plot, Teff_plot, rho_c_plot, = [], [], [], [], []
    delta_plot, tau_plot, Prad_C_plot, Pgas_C_plot = [], [], [], []

    r_plot = np.geomspace(50 * rg, 3e4 * rg, 100)
    for r in r_plot[29:]:
        # print('Finding Pi parameters of structure and making a structure plot.')
        # print('M = {:g} grams \nr = {:g} cm \nalpha = {:g} \nMdot = {:g} g/s'.format(M, r, alpha, Mdot))
        # print('Mdot = {:g} g/s'.format(Mdot))
        print('r = {:g} rg'.format(r / rg))
        h = np.sqrt(G * M * r)
        r_in = 3 * rg
        F = Mdot * h * (1 - np.sqrt(r_in / r))
        vs = MesaVerticalStructureRadConvPrad(M, alpha, r, F)
        z0r, result = vs.fit()
        if result.converged:
            print('The vertical structure has been calculated successfully.')
        # Pi = vs.Pi_finder()
        # print('Pi parameters =', Pi)
        # t = np.linspace(0, 1, 100)
        # S, P, Q, T = vs.integrate(t)[0]
        # P_gas = P[0] * vs.P_norm
        # P_rad = 4 * sigmaSB / (3 * c) * (T[0] * vs.T_norm) ** 4
        #
        # P_gas_c = P[-1] * vs.P_norm
        # P_rad_c = 4 * sigmaSB / (3 * c) * (T[-1] * vs.T_norm) ** 4

        varkappa_C, rho_C, T_C, Pgas_C, Sigma0 = vs.parameters_C()
        Prad_C = (4 * sigmaSB) / (3 * c) * T_C ** 4
        delta = Prad_C / Pgas_C
        Teff = vs.Teff
        tau = vs.tau()
        delta_plot.append(delta)
        Teff_plot.append(Teff)
        z0r_plot.append(z0r)
        tau_plot.append(tau)
        rho_c_plot.append(rho_C)
        T_C_plot.append(T_C)
        Sigma_plot.append(Sigma0)
        Prad_C_plot.append(Prad_C)
        Pgas_C_plot.append(Pgas_C)

        # print('P_rad / P_gas = {:g}'.format(P_rad / P_gas))
        # print('P_rad_c / P_gas_c = {:g}'.format(P_rad_c / P_gas_c))
        # print('P_rad_c / P_gas_c, delta = {:g}'.format(delta))
        # print(Sigma0)
        np.savetxt('fig/Prad_test/z0r1.txt', z0r_plot)
        np.savetxt('fig/Prad_test/T_C1.txt', T_C_plot)
        np.savetxt('fig/Prad_test/rho_C1.txt', rho_c_plot)
        np.savetxt('fig/Prad_test/Teff1.txt', Teff_plot)
        np.savetxt('fig/Prad_test/tau1.txt', tau_plot)
        np.savetxt('fig/Prad_test/Sigma1.txt', Sigma_plot)
        np.savetxt('fig/Prad_test/delta1.txt', delta_plot)
        np.savetxt('fig/Prad_test/r1.txt', r_plot)
        np.savetxt('fig/Prad_test/Prad_C1.txt', Prad_C_plot)
        np.savetxt('fig/Prad_test/Pgas_C1.txt', Pgas_C_plot)
    raise Exception

    data = ascii.read('F_xspec.dat')
    f = list(data['col3'])
    d = 5 * 3 * 1e21
    M = 1.5 * M_sun
    eta = 0.1
    alpha = 0.3
    maximum = max(f)
    for i, el in enumerate(f):
        if el == maximum:
            numer = i
            break

    flux = f[numer]
    Mdot = 4 * np.pi * d ** 2 * flux / (eta * c ** 2)
    print(Mdot)
    r = 1.87 * R_sun
    h = (G * M * r) ** (1 / 2)
    import mesa_vs
    vs = mesa_vs.MesaVerticalStructure(M, alpha, r, Mdot * h, irr_heat=True)
    z0r_init = 2.86e-7 * vs.F ** (3 / 20) * (vs.Mx / M_sun) ** (-9 / 20) * vs.alpha ** (-1 / 10) * (vs.r / 1e10) ** (
            1 / 20)
    print(z0r_init)
    print(vs.fit())
    print(vs.Teff)
    sys.exit()

    Structure_Plot(M, alpha, r, 1e18, input='Mdot', structure='Mesa')

    S_curve(2e3, 1.2e4, M, alpha, r, structure='Mesa', n=300, input='Teff', output='Teff', save=True,
            path='fig/S-curve-Ab.pdf', savedots=True)

    # Structure_Plot(M, alpha, r, 1e4, structure='Mesa')

    raise Exception

    # for Teff in [2000, 3000, 4000, 5000, 7000, 10000, 12000]:
    #     F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Teff ** 4
    #
    #     vs = MesaVerticalStructure(M, alpha, r, F)
    #     print(Teff, 'K')
    #     print(vs.tau0())
    #     TempGrad_Plot(vs)

    # F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * 2000 ** 4
    # vs = MesaVerticalStructure(M, alpha, r, F)
    # print(vs.tau0())

    # plot = []
    # for Teff in np.linspace(4e3, 2e4, 200):
    #     print(Teff)
    #     h = (G * M * r) ** (1 / 2)
    #     F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Teff ** 4
    #
    #     vs = MesaVerticalStructure(M, alpha, r, F)
    #
    #     plot.append(TempGrad_Plot(vs))
    # teff_plot = np.linspace(4e3, 2e4, 200)
    # plt.plot(teff_plot, plot)
    # plt.grid(True)
    # plt.xlabel('Teff')
    # plt.ylabel('Conv. parameter (z0)')
    # plt.savefig('fig/plot.pdf')

    raise Exception
    # Opacity_Plot(2e3, 1e4, M, alpha, r, structure='Mesa', n=1000, input='Teff', path='fig/Opacity-new.pdf')
    ######
    # S_curve(2e3, 1e4, M, alpha, r, structure='Mesa', n=300, input='Teff', output='Teff', save=False, title=False, lolita=1)
    # S_curve(2e3, 1e4, M, alpha, r, structure='MesaIdeal', n=300, input='Teff', output='Teff', save=False, mu=0.4, title=False, lolita=1)
    # S_curve(2e3, 1e4, M, alpha, r, structure='MesaIdeal', n=300, input='Teff', output='Teff', save=False, mu=0.8, title=False, lolita=1)
    # S_curve(2e3, 1e4, M, alpha, r, structure='MesaIdeal', n=300, input='Teff', output='Teff', save=False, mu=1.0, title=False, lolita=1)
    # plt.legend()
    # plt.savefig('fig/Full-S-curve-2.pdf')
    # plt.close()
    #
    # S_curve(2e3, 1e4, M, alpha, r, structure='Mesa', n=1000, input='Teff', output='Teff', save=False, title=False, lolita=2)
    # S_curve(2e3, 1e4, M, alpha, r, structure='Kramers', n=300, input='Teff', output='Teff', save=False, title=False, lolita=2)
    # S_curve(2e3, 1e4, M, alpha, r, structure='BellLin', n=1000, input='Teff', output='Teff', save=False, title=False, lolita=2)
    # plt.legend()
    # plt.savefig('fig/Full-S-curve-1.pdf')
    # plt.close()


if __name__ == '__main__':
    for days in [45, 55]:
        main(days)
