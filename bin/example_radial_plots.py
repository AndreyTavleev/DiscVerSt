import numpy as np
from astropy import constants as const
from matplotlib import pyplot as plt
from matplotlib import rcParams, cm

import mesa_vs

rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['text.latex.preamble'] = [r'\usepackage[utf8]{inputenc}', r'\usepackage{amsfonts, amsmath, amsthm, amssymb}',
                                   r'\usepackage[english]{babel}']

G = const.G.cgs.value
M_sun = const.M_sun.cgs.value
R_sun = const.R_sun.cgs.value
sigmaSB = const.sigma_sb.cgs.value
c = const.c.cgs.value
pl_const = const.h.cgs.value
k_B = const.k_B.cgs.value


def main(dotM, mass, alpha, index):
    M = mass * M_sun
    Mdot_edd = 1.39e18 * M / M_sun
    # Mdot = dotM * Mdot_edd
    Mdot = dotM
    rg = 2 * G * M / c ** 2
    # r_plot = np.geomspace(5 * rg, 1000 * rg, 100)
    r_plot = np.geomspace(1001 * rg, 2000 * rg, 50)
    # r_plot = np.geomspace(2001 * rg, 6000 * rg, 60)
    # r_plot = np.geomspace(6001 * rg, 10000 * rg, 60)
    # r_plot = np.geomspace(10001 * rg, 14000 * rg, 60)
    # r_plot = np.geomspace(14001 * rg, 20000 * rg, 60)
    # np.savetxt('fig/TDE_SMBH/{:g}Msun/additional_r_plot{}1.txt'.format(M / M_sun, index), r_plot)
    Sigma0_plot = []
    z0r_plot = []
    print(mass, dotM / Mdot_edd, alpha)
    # for i, r in enumerate(r_plot[index:]):
    for i, r in enumerate(r_plot):
        print('i = ', i)
        h = np.sqrt(M * G * r)
        func = 1 - np.sqrt(3 * rg / r)
        F = Mdot * h * func
        vs = mesa_vs.MesaVerticalStructureRadConv(M, alpha, r, F)
        z0r, result = vs.fit()
        varkappa_C, rho_C, T_C, P_C, Sigma0 = vs.parameters_C()
        Sigma0_plot.append(Sigma0)
        z0r_plot.append(z0r)
        delta = (4 * sigmaSB) / (3 * c) * T_C ** 4 / P_C
        if delta > 1.0:
            print('delta > 1.0')
        if vs.tau() < 1.0:
            print('tau < 1.0')

        np.savetxt('fig/TDE_SMBH/{:g}Msun/Sigma{}_alpha{:g}_additional.txt'.
                   format(M / M_sun, index, alpha), Sigma0_plot)
        np.savetxt('fig/TDE_SMBH/{:g}Msun/z0r{}_alpha{:g}_additional.txt'.
                   format(M / M_sun, index, alpha), z0r_plot)
    #     mesaop = mesa_vs.Opac('solar')
    #     rho, eos = mesaop.rho(P_C, T_C, True)
    # print('func = ', func)
    # print('omegaK = ', vs.omegaK)
    # print('z0r, sigma0 = {}, {}'.format(z0r, Sigma0))
    # print('P_c, T_c = {}, {}'.format(P_C, T_C))
    # print(rho, eos)
    # print(varkappa_C)
    return
    raise Exception

    Mdot = 3.184241601504656222e+24
    rg = 2 * G * M / c ** 2
    r = 100 * 3 * rg
    h = np.sqrt(M * G * r)
    F = Mdot * h * (1 - np.sqrt(3 * rg / r))
    vs = mesa_vs.MesaVerticalStructureRadConv(M, alpha, r, F)
    z0r, result = vs.fit()
    t = np.linspace(0, 1, 100)
    S, P, Q, T = vs.integrate(t)[0]
    varkappa_C, rho_C, T_C, P_C, Sigma0 = vs.parameters_C()
    S = S * vs.sigma_norm / Sigma0
    P = P[::-1] * vs.P_norm / P_C
    T = T[::-1] * vs.T_norm / T_C
    Q = Q[::-1]
    plt.plot(S, P, label=r'$p$')
    plt.plot(S, T, label=r'$\theta$', ls='-.')
    plt.plot(S, Q, label=r'$q$', ls=':')
    plt.plot(S, t, label=r'$z/z_0$', ls='--')
    plt.legend()
    plt.grid()
    plt.xlim(xmin=0, xmax=1)
    plt.ylim(ymin=0, ymax=1)
    plt.xlabel(r'$\sigma$')
    plt.tight_layout()
    plt.savefig('fig/VAZNO.pdf')
    raise Exception

    # p = np.geomspace(1e-12, 1e10, 1000)
    # t = np.geomspace(178, 3.16e10, 1000)
    # # arr = np.empty((1000, 1000))
    # # arr2 = np.empty((1000, 1000))
    # # for i, el in enumerate(p):
    # #     for j, el2 in enumerate(t):
    # #         print('{}, {}'.format(i, j))
    # #         mesaop = mesa_vs.Opac('solar')
    # #         rho, eos = mesaop.rho(el, el2, True)
    # #         arr[i][j] = eos.dlnRho_dlnT_const_Pgas
    # #         arr2[i][j] = rho
    # #
    # # np.savetxt('fig/TEXT.txt', arr)
    # # np.savetxt('fig/TEXT2.txt', arr2)
    # # raise Exception
    # # # print(np.amax(arr), np.amin(arr))
    # # t = t[:500]
    # # p = p[:900]
    # p, t = np.meshgrid(np.log10(t), np.log10(p))
    # # fig = plt.figure()
    # # ax = fig.gca(projection='3d')
    # arr = np.loadtxt('/Users/andrey/verstr/fig/TEXT.txt')
    # # rho = np.loadtxt('/Users/andrey/verstr/fig/TEXT2.txt')
    # # rho = np.delete(rho, [i for i in range(900, np.shape(rho)[0])], 0)
    # # arr = np.delete(arr, [i for i in range(500, np.shape(arr)[1])], 1)
    # print(np.shape(p), np.shape(t), np.shape(arr))
    # # print(np.shape(arr))
    # # sum = 0
    # for i, row in enumerate(arr):
    #     for j, el in enumerate(row):
    #         # if el < -1e10:
    #         #     sum = sum + 1
    #         if np.isnan(el):
    #             arr[i][j] = -1
    #         if el > 0.0:
    #             arr[i][j] = 10
    #         if el < 0.0:
    #             arr[i][j] = 0
    #             # sum = sum + 1
    #             # if p[i] > 350:
    #             #     print('qqqqq')
    #             #     raise Exception
    #             # if t[j] > 2.5e5:
    #             #     print('ggggg')
    #             #     raise Exception
    #             # print(i, j, p[i], t[j])
    #
    # # arr2 = arr > 0.0
    # # print(np.nanmax(rho), np.nanmin(rho))
    # fig, ax = plt.subplots()
    # cs = ax.contourf(p, t, arr, cmap=cm.coolwarm, levels=50)
    # fig.colorbar(cs)
    # # ax.plot_surface(p, t, arr)
    # plt.show()
    # raise Exception

    M = 1e8 * M_sun
    alpha = 0.5
    Mdot_edd = 1.39e18 * M / M_sun
    Mdot = 1 * Mdot_edd
    Mdot = 3.184241601504656222e+24
    rg = 2 * G * M / c ** 2
    mu = 0.62
    z0r_plot = []
    z0r_teor_plot = []  # zone C
    z0r_teor2_plot = []  # zone B
    sigma0_plot, sigma0_teor_plot, sigma0_teor2_plot = [], [], []
    rho_c_plot, rho_c_teor_plot, rho_c_teor2_plot = [], [], []
    t_c_plot, t_c_teor_plot, t_c_teor2_plot = [], [], []
    r_plot = []
    delta_plot = []
    varkappa_C_plot = []
    tau_plot = []
    t_vis_plot = []

    for i, r in enumerate(np.geomspace(1.5 * 3 * rg, 200 * 3 * rg, 200)):
        print(i)
        h = (G * M * r) ** (1 / 2)
        z0r_teor = 0.020 * (M / M_sun) ** (-3 / 8) * alpha ** (-1 / 10) * (r / 1e10) ** (1 / 8) * (mu / 0.6) ** (
                -3 / 8) * (Mdot / 1e17) ** (3 / 20) * 2.6

        z0r_teor2 = 0.0092 * (M / M_sun) ** (-7 / 20) * alpha ** (-1 / 10) * (r / 1e7) ** (1 / 20) * (mu / 0.6) ** (
                -2 / 5) * (Mdot / 1e17) ** (1 / 5) * 2.6

        sigma0_teor = 33 * (M / M_sun) ** (1 / 4) * alpha ** (-4 / 5) * (r / 1e10) ** (-3 / 4) * (mu / 0.6) ** (
                3 / 4) * (Mdot / 1e17) ** (7 / 10) * 1.03

        sigma0_teor2 = 5.1e3 * (M / M_sun) ** (1 / 5) * alpha ** (-4 / 5) * (r / 1e7) ** (-3 / 5) * (mu / 0.6) ** (
                4 / 5) * (Mdot / 1e17) ** (3 / 5) * 0.96

        rho_c_teor = 8e-8 * (M / M_sun) ** (5 / 8) * alpha ** (-7 / 10) * (r / 1e10) ** (-15 / 8) * (mu / 0.6) ** (
                9 / 8) * (Mdot / 1e17) ** (11 / 20) * 0.76

        rho_c_teor2 = 2.8e-2 * (M / M_sun) ** (11 / 20) * alpha ** (-7 / 10) * (r / 1e7) ** (-33 / 20) * (mu / 0.6) ** (
                6 / 5) * (Mdot / 1e17) ** (2 / 5) * 0.67

        t_c_teor = 4e4 * (M / M_sun) ** (1 / 4) * alpha ** (-1 / 5) * (r / 1e10) ** (-3 / 4) * (mu / 0.6) ** (
                1 / 4) * (Mdot / 1e17) ** (3 / 10) * 1.09

        t_c_teor2 = 8.2e6 * (M / M_sun) ** (3 / 10) * alpha ** (-1 / 5) * (r / 1e7) ** (-9 / 10) * (mu / 0.6) ** (
                1 / 5) * (Mdot / 1e17) ** (2 / 5) * 1.2

        func = 1 - np.sqrt(3 * rg / r)
        vs = mesa_vs.MesaVerticalStructureRadConv(M, alpha, r, Mdot * h * func)
        z0r, result = vs.fit()
        varkappa_C, rho_C, T_C, P_C, Sigma0 = vs.parameters_C()

        delta = (4 * sigmaSB) / (3 * c) * T_C ** 4 / P_C
        delta_plot.append(delta)
        varkappa_C_plot.append(varkappa_C)
        tau_plot.append(vs.tau())
        omegaK = np.sqrt(G * M / r ** 3)
        t_vis = 1 / (alpha * omegaK) * z0r ** (-2)
        t_vis_plot.append(t_vis)

        sigma0_plot.append(Sigma0)
        z0r_plot.append(z0r)
        rho_c_plot.append(rho_C)
        t_c_plot.append(T_C)
        sigma0_teor_plot.append(sigma0_teor)
        sigma0_teor2_plot.append(sigma0_teor2)
        z0r_teor_plot.append(z0r_teor)
        z0r_teor2_plot.append(z0r_teor2)
        rho_c_teor_plot.append(rho_c_teor)
        rho_c_teor2_plot.append(rho_c_teor2)
        t_c_teor_plot.append(t_c_teor)
        t_c_teor2_plot.append(t_c_teor2)
        r_plot.append(r)

    # np.savetxt('fig/z0r_plot_{:g}_Mdot{:g}.txt'.format(M / M_sun, Mdot / Mdot_edd), z0r_plot)
    # np.savetxt('fig/r_plot_{:g}_Mdot{:g}.txt'.format(M / M_sun, Mdot / Mdot_edd), r_plot)
    # np.savetxt('fig/delta_plot_{:g}_Mdot{:g}.txt'.format(M / M_sun, Mdot / Mdot_edd), delta_plot)
    # np.savetxt('fig/varkappa_C_plot_{:g}_Mdot{:g}.txt'.format(M / M_sun, Mdot / Mdot_edd), varkappa_C_plot)
    # np.savetxt('fig/t_vis_plot_{:g}_Mdot{:g}.txt'.format(M / M_sun, Mdot / Mdot_edd), t_vis_plot)
    # np.savetxt('fig/tau_plot_{:g}_Mdot{:g}.txt'.format(M / M_sun, Mdot / Mdot_edd), tau_plot)
    #
    # raise Exception

    plt.plot(r_plot, z0r_plot, label='Mesa')
    plt.plot(r_plot, z0r_teor_plot, '--')
    plt.plot(r_plot, z0r_teor2_plot, '-.')
    # plt.ylim(ymin=0, ymax=0.024)
    plt.grid()
    plt.xscale('log')
    plt.ylabel(r'$z_0 / r$')
    plt.xlabel(r'$r/rg$')
    plt.tight_layout()
    plt.savefig('fig/r-z0r1111.pdf')
    plt.close()

    plt.plot(r_plot, sigma0_plot)
    plt.plot(r_plot, sigma0_teor_plot, '--')
    plt.plot(r_plot, sigma0_teor2_plot, '-.')
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$\Sigma_0$')
    plt.xlabel(r'$r/rg$')
    plt.tight_layout()
    plt.savefig('fig/r-Sigma01111.pdf')
    plt.close()

    plt.plot(r_plot, rho_c_plot)
    plt.plot(r_plot, rho_c_teor_plot, '--')
    plt.plot(r_plot, rho_c_teor2_plot, '-.')
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$\rho_c$')
    plt.xlabel(r'$r/rg$')
    plt.tight_layout()
    plt.savefig('fig/r-Rho_c1111.pdf')
    plt.close()

    plt.plot(r_plot, t_c_plot)
    plt.plot(r_plot, t_c_teor_plot, '--')
    plt.plot(r_plot, t_c_teor2_plot, '-.')
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$T_c$')
    plt.xlabel(r'$r/rg$')
    plt.tight_layout()
    plt.savefig('fig/r-T_c1111.pdf')
    plt.close()


if __name__ == '__main__':
    mass = 1e6
    alpha = 0.01
    Mdot = np.loadtxt('fig/SigmaDotsAll/MdotForSigmaMin_{:g}Msun_{:g}alpha.txt'.format(mass, alpha))
    start = 29
    end = 30
    # index = start
    step = 10
    # for dotM in Mdot[start:end:step]:
    #     main(dotM, mass, alpha, index)
    #     index = index + step

    # for mass in [1e6, 1e7, 1e8, 1e9]:
    for mass in [1e8]:
        Mdot = np.loadtxt('fig/SigmaDotsAll/MdotForSigmaMin_{:g}Msun_{:g}alpha.txt'.format(mass, alpha))
        index = start
        for dotM in Mdot[start:end:step]:
            print('mass = {:.1e}, index = {}'.format(mass, index))
            main(dotM, mass, alpha, index)
            index = index + step

    # for mass in [1e6, 1e7, 1e8, 1e9]:
    #     for dotM in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
    #         for alpha in [0.01, 0.5]:
    #             main(dotM, mass, alpha, index=0)
