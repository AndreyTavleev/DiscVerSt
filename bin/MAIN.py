from plots import Structure_Plot, TempGrad_Plot, S_curve, Opacity_Plot
import numpy as np
from astropy import constants as const

G = const.G.cgs.value
M_sun = const.M_sun.cgs.value
c = const.c.cgs.value

def main():
    M = 1.5 * M_sun
    rg = 2 * G * M / c ** 2
    alpha = 0.1
    r = 1e9

    Structure_Plot(M, alpha, r, 1e4, structure='Mesa')

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
    S_curve(2e3, 1.2e4, M, alpha, r, structure='Mesa', n=300, input='Teff', output='Teff', save=False,
            path='fig/S-curve-Ab.pdf')

    sigma_down = 74.6 * (alpha / 0.1) ** (-0.83) * (r / 1e10) ** 1.18 * (M / M_sun) ** (-0.4)
    sigma_up = 39.9 * (alpha / 0.1) ** (-0.8) * (r / 1e10) ** 1.11 * (M / M_sun) ** (-0.37)

    mdot_down = 2.64e15 * (alpha / 0.1) ** 0.01 * (r / 1e10) ** 2.58 * (M / M_sun) ** (-0.85)
    mdot_up = 8.07e15 * (alpha / 0.1) ** (-0.01) * (r / 1e10) ** 2.64 * (M / M_sun) ** (-0.89)

    plt.scatter(sigma_down, mdot_down)
    plt.scatter(sigma_up, mdot_up)

    plt.savefig('fig/S-curve-Ab.pdf')

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

    # Opacity_Plot(2e3, 5e4, M, alpha, r, structure='Mesa', n=100, save=False)
    # Opacity_Plot(2e3, 5e4, M, alpha, r, structure='BellLin', n=100, save=False)
    # Opacity_Plot(1e4, 5e4, M, alpha, r, structure='Kramers', n=5, save=False)
    # plt.legend()
    # # plt.savefig('fig/Opacity.pdf')
    # plt.close()
    #
    # h = (G * M * r) ** (1 / 2)
    # F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * 1e4 ** 4
    # vs = MesaVerticalStructure(M, alpha, r, F)
    # TempGrad_Plot(vs)

    # A = (1.7e18 * 3 * c ** 6 / (64 * np.pi * G ** 2)) ** (1 / 3) * (M * M_sun * sigmaSB * 1e16) ** (-1 / 3)

    # B = 3 / (64 * np.pi) * 1.7e18 * c ** 6 * 1e-2 / (G ** 2 * sigmaSB * 1e16)
    # A = (B / M) ** (1 / 3)
    # print('{:g}'.format(A*rg))
    # print('{:g}'.format((1.7e18*c**6/G**2)**(1/3) * (M*M_sun)**(-1/3)))
    # print('{:g}'.format(A/r))

    # for M in [6 * M_sun, 1e2 * M_sun, 1e3 * M_sun, 1e4 * M_sun, 1e5 * M_sun, 1e7 * M_sx`un, 1e8 * M_sun]:
    # for M in [6*M_sun, 60*M_sun, 600 * M_sun]:
    #
    ##### for M in [2e5 * M_sun, 3e5 * M_sun, 5e5 * M_sun]:
    #     # M = 1e8 * M_sun
    #     # rg = 2 * G * M / c ** 2
    #     # for r in [30 * rg, 60 * rg, 100 * rg, 200 * rg, 300 * rg]:
    #     rg = 2 * G * M / c ** 2
    #     A = (1.7e18 * 3 * c ** 6 / (64 * np.pi * G ** 2)) ** (1 / 3) * (M * M_sun * sigmaSB * 1e16) ** (-1 / 3)
    #     r = A * rg
    #     print('{:g}'.format(r))
    #     print('{:g}'.format(M / M_sun))
    #     alpha = 0.5
    #     Mdot_edd = 1.7e18 * M / M_sun
    #     S_curve(Mdot_edd * 1e-4, 2 * Mdot_edd, M, alpha, r, n=1000, input='Mdot', output='Mdot/Mdot_edd', save=False)
    #
    # plt.savefig('fig/S-curve-big13.pdf')
    # plt.close()

    # for M in [1e6 * M_sun, 5e6 * M_sun, 1e7 * M_sun]:
    for M in [1e5 * M_sun, 1e6 * M_sun, 1e7 * M_sun]:
        rg = 2 * G * M / c ** 2
        A = (1.7e18 * 3 * c ** 6 / (64 * np.pi * G ** 2)) ** (1 / 3) * (M * M_sun * sigmaSB * 1e16) ** (-1 / 3)
        r = A * rg
        print('{:g}'.format(r))
        print('{:g}'.format(M / M_sun))
        alpha = 0.5
        Mdot_edd = 1.7e18 * M / M_sun
        S_curve(Mdot_edd * 1e-4, 2.3 * Mdot_edd, M, alpha, r, n=1000, input='Mdot', output='Teff', save=False)
        # dFdS_Plot(Mdot_edd * 1e-4, 2 * Mdot_edd, M, alpha, r, n=1000, input='Mdot', save=True)
    plt.savefig('fig/S-curve-big-ABD.pdf')
    plt.close()
    #
    #
    # for M in [6000*M_sun, 6e5*M_sun, 6e6 * M_sun]:
    #     rg = 2 * G * M / c ** 2
    #     A = (1.7e18 * 3 * c ** 6 / (64 * np.pi * G ** 2)) ** (1 / 3) * (M * M_sun * sigmaSB * 1e16) ** (-1 / 3)
    #     r = A * rg
    #     print('{:g}'.format(r))
    #     print('{:g}'.format(M))
    #     alpha = 0.5
    #     Mdot_edd = 1.7e18 * M / M_sun
    #     S_curve(Mdot_edd * 1e-4, 2 * Mdot_edd, M, alpha, r, input='Mdot', output='Mdot')
    #
    # plt.savefig('fig/S-curve-big2.pdf')
    # plt.close()
    #
    # for M in [6e7*M_sun, 6e8*M_sun, 6e9 * M_sun]:
    #     rg = 2 * G * M / c ** 2
    #     A = (1.7e18 * 3 * c ** 6 / (64 * np.pi * G ** 2)) ** (1 / 3) * (M * M_sun * sigmaSB * 1e16) ** (-1 / 3)
    #     r = A * rg
    #     print('{:g}'.format(r))
    #     print('{:g}'.format(M))
    #     alpha = 0.5
    #     Mdot_edd = 1.7e18 * M / M_sun
    #     S_curve(Mdot_edd * 1e-4, 2 * Mdot_edd, M, alpha, r, input='Mdot', output='Mdot')
    #
    # plt.savefig('fig/S-curve-big3.pdf')
    # plt.close()
    #
    #
    #
    #
    # for M in [6 * M_sun, 60 * M_sun, 600 * M_sun]:
    #     rg = 2 * G * M / c ** 2
    #     A = (1.7e18 * 3 * c ** 6 / (64 * np.pi * G ** 2)) ** (1 / 3) * (M * M_sun * sigmaSB * 1e16) ** (-1 / 3)
    #     r = A * rg
    #     print('{:g}'.format(r))
    #     print('{:g}'.format(M))
    #     alpha = 0.5
    #     Mdot_edd = 1.7e18 * M / M_sun
    #     S_curve(Mdot_edd * 1e-4, 2 * Mdot_edd, M, alpha, r, input='Mdot', output='Teff')
    #
    # plt.savefig('fig/S-curve-big4.pdf')
    # plt.close()
    #
    # for M in [6000 * M_sun, 6e5 * M_sun, 6e6 * M_sun]:
    #     rg = 2 * G * M / c ** 2
    #     A = (1.7e18 * 3 * c ** 6 / (64 * np.pi * G ** 2)) ** (1 / 3) * (M * M_sun * sigmaSB * 1e16) ** (-1 / 3)
    #     r = A * rg
    #     print('{:g}'.format(r))
    #     print('{:g}'.format(M))
    #     alpha = 0.5
    #     Mdot_edd = 1.7e18 * M / M_sun
    #     S_curve(Mdot_edd * 1e-4, 2 * Mdot_edd, M, alpha, r, input='Mdot', output='Teff')
    #
    # plt.savefig('fig/S-curve-big5.pdf')
    # plt.close()
    #
    # for M in [6e7 * M_sun, 6e8 * M_sun, 6e9 * M_sun]:
    #     rg = 2 * G * M / c ** 2
    #     A = (1.7e18 * 3 * c ** 6 / (64 * np.pi * G ** 2)) ** (1 / 3) * (M * M_sun * sigmaSB * 1e16) ** (-1 / 3)
    #     r = A * rg
    #     print('{:g}'.format(r))
    #     print('{:g}'.format(M))
    #     alpha = 0.5
    #     Mdot_edd = 1.7e18 * M / M_sun
    #     S_curve(Mdot_edd * 1e-4, 2 * Mdot_edd, M, alpha, r, input='Mdot', output='Teff')
    #
    # plt.savefig('fig/S-curve-big6.pdf')
    # plt.close()

    # Opacity_Plot(1e21, 1e27, M, alpha, r, input='Mdot')

    # plot = []
    # plot_t = []
    # for i, Mdot in enumerate(np.r_[1e21:1e27:200j]):
    #     print(i + 1)
    #     h = (G * M * r) ** (1 / 2)
    #     F = Mdot * h
    #     vs = MesaVerticalStructure(M, alpha, r, F)
    #     vs.fit()
    #     print(vs.parameters_C()[1])
    #     oplya = opacity.kappa(3e-7, vs.parameters_C()[2])
    #     plot.append(oplya)
    #     plot_t.append(vs.parameters_C()[2])
    #
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.plot(plot_t, plot)
    # plt.savefig('fig/oplya.pdf')

    # for Mdot in np.linspace(1e21, 1e27, 20):
    #     Structure_Plot(M, alpha, r, Mdot, input='Mdot')

    # Structure_Plot(M, alpha, r, 2e4, input='Teff')

    # for r in [5.5e10, 6e10, 6.75e10]:
    #     S_curve(2.3e3, 1e4, M, alpha, r, output='Mdot')
    #
    # plt.hlines(1e17, *plt.xlim(), linestyles='--')
    # plt.grid(True, which='both', ls='-')
    # plt.legend()
    # plt.title('S-curve')
    # plt.show()

    # for Teff in np.linspace(2e3, 5e4, 20):
    # for Mdot in np.linspace(1e18, 1e19, 20):
    #     h = (G * M * r) ** (1 / 2)
    #     # F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Teff ** 4
    #     F = Mdot * h
    #     vs = MesaVerticalStructure(M, alpha, r, F)
    #     TempGrad_Plot(vs)
    #     print('Teff = %d' % vs.Teff)
    #     print('tau = %d' % vs.tau0())

    # S_curve(1e17, 1e20, M, alpha, r, 1000, input='Mdot', output='Mdot', save=False)

    # Sigma_up = 589 * (alpha / 0.1) ** (-0.78) * (r / 1e10) ** 1.07 * (M / M_sun) ** (-0.36)
    # Teff_up = 13100 * (alpha / 0.1) ** (-0.01) * (r / 1e10) ** (-0.08) * (M / M_sun) ** 0.03
    # Mdot_up = 1.05e17 * (alpha / 0.1) ** (-0.05) * (r / 1e10) ** 2.69 * (M / M_sun) ** (-0.9)
    # Sigma_down = 1770 * (alpha / 0.1) ** (-0.83) * (r / 1e10) ** 1.2 * (M / M_sun) ** (-0.4)
    # Teff_down = 9700 * (r / 1e10) ** (-0.09) * (M / M_sun) ** 0.03
    # Mdot_down = 3.18e16 * (alpha / 0.1) ** (-0.01) * (r / 1e10) ** 2.65 * (M / M_sun) ** (-0.88)
    #
    # Mdot_down_another = 5.9e16 * (alpha / 0.1) ** (-0.41) * (r / 1e10) ** 2.62 * (M / M_sun) ** (-0.87)
    #
    # print(Sigma_up, Mdot_up)
    # print(Sigma_down, Mdot_down)
    #
    # plt.scatter(Sigma_down, Mdot_down)
    # plt.scatter(Sigma_up, Mdot_up)
    # plt.scatter(Sigma_down, Mdot_down_another, marker='*')
    #
    # plt.savefig('fig/Helium-S-curve.pdf')
    # plt.close()
    #
    # # Opacity_Plot(1e3, 3e4, M, alpha, r)
    #
    # # Teff = 11077.0
    # # h = (G * M * r) ** (1 / 2)
    # # F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Teff ** 4
    # # vs = MesaVerticalStructure(M, alpha, r, F)
    # # TempGrad_Plot(vs)
    # Structure_Plot(M, alpha, r, 1e4, input='Teff')


if __name__ == '__main__':
    main()
