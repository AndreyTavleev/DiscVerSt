import numpy as np
from astropy import constants as const
from matplotlib import pyplot as plt
from matplotlib import rcParams
from plots import S_curve

rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['text.latex.preamble'] = [r'\usepackage[utf8]{inputenc}', r'\usepackage{amsfonts, amsmath, amsthm, amssymb}',
                                   r'\usepackage[english]{babel}']

G = const.G.cgs.value
M_sun = const.M_sun.cgs.value
R_sun = const.R_sun.cgs.value
c = const.c.cgs.value
pl_const = const.h.cgs.value
k_B = const.k_B.cgs.value


def main():
    M = 1 * M_sun
    GM = G * M
    rg = 2 * GM / c ** 2
    r = 1e10

    # for abun in [{'he4': 1.0}]:
    for abun in [{'h1': 1.0}, 'solar']:
        for struct in ['MesaRadConv', 'Mesa']:
            for alpha in [0.01, 0.5]:
                if abun != 'solar':
                    path = 'fig/Experiment/T_C/h1/' + struct
                else:
                    path = 'fig/Experiment/T_C/solar/' + struct
                conv_param_z0, conv_param_sigma = S_curve(
                    2e2, 3e4, M, alpha=alpha, r=r, structure=struct, abundance=abun, n=500, input='Teff', output='T_C',
                    savedots=True, path_Sigma=path + '/Sigma_0{:g}.txt'.format(alpha),
                    path_output=path + '/T_C{:g}.txt'.format(alpha),
                    save_indexes=True, path_indexes=path + '/indexes{:g}.txt'.format(alpha), make_pic=False,
                    save_plot=False, set_title=True, title=r'M = {:g} Msun, r = {:g} cm'.format(M / M_sun, r))
                np.savetxt(path + '/conv_z0{:g}.txt'.format(alpha), conv_param_z0)
                np.savetxt(path + '/conv_sigma{:g}.txt'.format(alpha), conv_param_sigma)
    # plt.savefig('fig/Very_important_S_curve.pdf')
    # plt.close()
    raise Exception

    # n = 57
    # m = 80
    # M = 1.5 * M_sun
    # alpha = 0.5
    # Mdot_plot = np.loadtxt('fig/Aql-X_1_Shakura/Mdot_plot.txt')
    # Mdot = Mdot_plot[n]
    # Sigma0_plot = np.loadtxt('fig/Aql-X_1_Shakura/Sigma0_plot{}.txt'.format(n))
    # r_plot = np.loadtxt('fig/Aql-X_1_Shakura/r.txt')
    # r = r_plot[m]
    # Sigma0 = Sigma0_plot[m]
    #
    # S_curve(0.1 * Mdot, 2 * Mdot, M, alpha=alpha, r=r, structure='MesaRadConvIrrZero', n=50,
    #         input='Mdot', output='Mdot', xscale='parlog', yscale='parlog', save=False, set_title=True,
    #         title=r'M = {:g} Msun, r = {:g} Rsun'.format(M / M_sun, r / R_sun), savedots=False)
    #
    # plt.scatter(np.log10(Sigma0), np.log10(Mdot), marker='H')
    # plt.savefig('fig/Aql-X_1_Shakura/S-curve{}.pdf'.format(n))
    # raise Exception

    M = 1.4 * M_sun
    Rtid = 1.86584883711 * R_sun
    r = 0.85 * Rtid

    S_curve(4e16, 6e19, M, alpha=0.5, r=r, structure='MesaRadConv', n=100,
            input='Mdot', output=end, xscale='log', yscale='log', save=False, set_title=True,
            title=r'M = {:g} Msun, r = {:g} Rsun'.format(M / M_sun, r / R_sun))
    # S_curve(4e16, 6e19, M, alpha=0.01, r=r, structure='MesaRadConv', n=200,
    #         input='Mdot', output=end, xscale='log', yscale='log', save=False, set_title=True,
    #         title=r'M = {:g} Msun, r = {:g} Rtid'.format(M / M_sun, r / Rtid))
    # plt.savefig('fig/Aql_Sigma_{}1.pdf'.format(end))
    plt.savefig('fig/Very_important_S_curve.pdf'.format(end))
    plt.close()

    return

    M = 3e9 * M_sun
    GM = G * M
    rg = 2 * GM / c ** 2
    r = 300 * rg
    alpha = 0.1
    Pi1 = 7
    Pi2 = 0.5
    Pi3 = 1.15
    Pi4 = 0.46
    Ledd = 1.25e38 * (M / M_sun)
    h = np.sqrt(GM * r)
    omegaK = np.sqrt(GM / r ** 3)

    S_curve(10 ** 2.6, 10 ** 3.6, M, alpha=alpha, r=r, structure='MesaRadConv', n=200,
            input='Teff', output='Teff', xscale='parlog', yscale='parlog', save=False, set_title=True,
            title=r'M = {:g} Msun, r = {:g} rg'.format(M / M_sun, r / rg))

    # S_curve(1e-6 * Mdot_edd, Mdot_edd, M, alpha=alpha, r=r, structure='MesaRadConv', n=300,
    #         input='Mdot', output='Mdot', xscale='parlog', yscale='parlog', save=False, set_title=True,
    #         title=r'M = {:g} Msun, r = {:g} cm'.format(M / M_sun, r))

    yy = np.geomspace(2.5, 6, 100)
    y = np.geomspace(2.5, 5.2, 100)
    ff = [(Pi3 * Pi4 ** 2 / (Pi1 * Pi2 ** 2) * 32 / (9 * np.pi * alpha * omegaK) *
           (Ledd / GM) ** 2 * 1 / (10 ** sigma)) for sigma in y]
    gg = [((2 * np.pi * alpha * h) / (Pi1 * Pi3) * (1 / 1) ** 2 * 10 ** sigma) for sigma in yy]

    # plt.plot(y, np.log10(ff), '--')
    # plt.plot(yy, np.log10(gg), '--')

    plt.savefig('fig/Janiuk2004_3.pdf')

    raise Exception

    M = 1e8 * M_sun
    GM = G * M
    rg = 2 * GM / c ** 2
    r = 300 * rg
    # r = 1.86 * R_sun
    # alpha = 0.02
    # S_curve(1e3, 1e4, M, alpha, r, structure='MesaRadConv', n=100, input='Teff', output='z0r', save=False, set_title=False)
    # alpha = 0.01
    # S_curve(2e3, 2e4, M, alpha=0.02, r=r, structure='MesaRadConvIrr', n=300, input='Teff', output='Teff',
    #         xscale='log', yscale='log', save=False, set_title=False)
    S_curve(1e18, 5e25, M, alpha=0.02, r=r, structure='MesaRadConv', n=500, input='Mdot', output='Mdot',
            xscale='parlog', yscale='parlog', save=False, set_title=True,
            title=r'M = {:g} Msun, r = {:g} cm'.format(M / M_sun, r))

    S_curve(1e18, 5e25, M, alpha=0.5, r=r, structure='MesaRadConv', n=500, input='Mdot', output='Mdot',
            xscale='parlog', yscale='parlog', save=False, set_title=True,
            title=r'M = {:g} Msun, r = {:g} rg'.format(M / M_sun, r / rg))

    Pi1 = 7
    Pi2 = 0.5
    Pi3 = 1.15
    Pi4 = 0.46
    Ledd = 1.25e38 * (M / M_sun)
    h = np.sqrt(GM * r)
    omegaK = np.sqrt(GM / r ** 3)

    alpha = 0.02
    xx = np.geomspace(4, 7, 100)
    x = np.geomspace(4, 6.4, 100)
    f = [(Pi3 * Pi4 ** 2 / (Pi1 * Pi2 ** 2) * 32 / (9 * np.pi * alpha * omegaK) *
          (Ledd / GM) ** 2 * 1 / (10 ** sigma)) for sigma in x]
    g = [((2 * np.pi * alpha * h) / (Pi1 * Pi3) * (1 / 1) ** 2 * 10 ** sigma) for sigma in xx]

    alpha = 0.5
    yy = np.geomspace(2.5, 6, 100)
    y = np.geomspace(2.5, 5.2, 100)
    ff = [(Pi3 * Pi4 ** 2 / (Pi1 * Pi2 ** 2) * 32 / (9 * np.pi * alpha * omegaK) *
           (Ledd / GM) ** 2 * 1 / (10 ** sigma)) for sigma in y]
    gg = [((2 * np.pi * alpha * h) / (Pi1 * Pi3) * (1 / 1) ** 2 * 10 ** sigma) for sigma in yy]

    plt.plot(x, np.log10(f), '--')
    plt.plot(y, np.log10(ff), '--')
    plt.plot(xx, np.log10(g), '--')
    plt.plot(yy, np.log10(gg), '--')

    plt.savefig('fig/S-curve_SMBH4.pdf')
    raise Exception

    sigma_down_sol = 74.6 * (alpha / 0.1) ** (-0.83) * (r / 1e10) ** 1.18 * (M / M_sun) ** (-0.4)
    sigma_up_sol = 39.9 * (alpha / 0.1) ** (-0.8) * (r / 1e10) ** 1.11 * (M / M_sun) ** (-0.37)

    mdot_down_sol = 2.64e15 * (alpha / 0.1) ** 0.01 * (r / 1e10) ** 2.58 * (M / M_sun) ** (-0.85)
    mdot_up_sol = 8.07e15 * (alpha / 0.1) ** (-0.01) * (r / 1e10) ** 2.64 * (M / M_sun) ** (-0.89)

    teff_up_sol = 6890 * (r / 1e10) ** (-0.09) * (M / M_sun) ** 0.03
    teff_down_sol = 5210 * (r / 1e10) ** (-0.10) * (M / M_sun) ** 0.04

    sigma_up_he = 589 * (alpha / 0.1) ** (-0.78) * (r / 1e10) ** 1.07 * (M / M_sun) ** (-0.36)
    teff_up_he = 13100 * (alpha / 0.1) ** (-0.01) * (r / 1e10) ** (-0.08) * (M / M_sun) ** 0.03
    mdot_up_he = 1.05e17 * (alpha / 0.1) ** (-0.05) * (r / 1e10) ** 2.69 * (M / M_sun) ** (-0.9)
    sigma_down_he = 1770 * (alpha / 0.1) ** (-0.83) * (r / 1e10) ** 1.2 * (M / M_sun) ** (-0.4)
    teff_down_he = 9700 * (r / 1e10) ** (-0.09) * (M / M_sun) ** 0.03
    mdot_down_he = 3.18e16 * (alpha / 0.1) ** (-0.01) * (r / 1e10) ** 2.65 * (M / M_sun) ** (-0.88)

    sigma_down_sol_ham = 13.4 * alpha ** (-0.83) * (r / 1e10) ** 1.14 * (M / M_sun) ** (-0.38)
    sigma_up_sol_ham = 8.3 * alpha ** (-0.77) * (r / 1e10) ** 1.11 * (M / M_sun) ** (-0.37)
    mdot_down_sol_ham = 4e15 * (alpha / 0.1) ** (-0.004) * (r / 1e10) ** 2.65 * (M / M_sun) ** (-0.88)
    mdot_up_sol_ham = 9.5e15 * (alpha / 0.1) ** 0.01 * (r / 1e10) ** 2.68 * (M / M_sun) ** (-0.89)

    # plt.scatter(sigma_down_sol, mdot_down_sol)
    # plt.scatter(sigma_up_sol, mdot_up_sol)

    # plt.savefig('fig/S-curve_with_dots.pdf')


if __name__ == '__main__':
    main()
