from vs import Vars, IdealKramersVerticalStructure, IdealBellLin1994VerticalStructure
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simps
from astropy import constants as const

try:
    import mesa_vs
except ImportError:
    mesa_vs = np.nan

sigmaSB = const.sigma_sb.cgs.value
G = const.G.cgs.value
M_sun = const.M_sun.cgs.value
c = const.c.cgs.value

rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['text.latex.preamble'] = [r'\usepackage[utf8]{inputenc}', r'\usepackage{amsfonts, amsmath, amsthm, amssymb}',
                                   r'\usepackage[english]{babel}']


def Structure_Plot(M, alpha, r, Par, mu=0.6, input='Teff', structure='Kramers', abundance='solar', title=True):
    h = (G * M * r) ** (1 / 2)

    if input == 'F':
        F = Par
    elif input == 'Teff':
        F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Par ** 4
    elif input == 'Mdot':
        F = Par * h
    else:
        print('Incorrect input, try Teff, Mdot of F')
        return

    if structure == 'Kramers':
        vs = IdealKramersVerticalStructure(M, alpha, r, F, mu=mu)
    elif structure == 'BellLin':
        vs = IdealBellLin1994VerticalStructure(M, alpha, r, F, mu=mu)
    elif structure == 'Mesa':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructure(M, alpha, r, F, abundance=abundance)
    elif structure == 'MesaIdeal':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaIdealVerticalStructure(M, alpha, r, F, mu=mu, abundance=abundance)
    elif structure == 'MesaAd':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructureAdiabatic(M, alpha, r, F, abundance=abundance)
    elif structure == 'MesaFirst':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructureFirstAssumption(M, alpha, r, F, abundance=abundance)
    elif structure == 'MesaRadConv':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructureRadConv(M, alpha, r, F, abundance=abundance)
    else:
        print('Incorrect structure, try Kramers, BellLin, Mesa, MesaIdeal, MesaAd, MesaFirst or MesaRadConv')
        return

    vs.fit()
    print('Teff = ', vs.Teff)
    t = np.linspace(0, 1, 100)
    S, P, Q, T = vs.integrate(t)[0]
    plt.plot(1 - t, S, label=r'$\Sigma$')
    plt.plot(1 - t, P, label='$P$')
    plt.plot(1 - t, Q, label='$Q$')
    plt.plot(1 - t, T, label='$T$')
    plt.grid()
    plt.legend()
    plt.xlabel('$z / z_0$')
    if title:
        plt.title('Vertical structure, Teff = %d K' % vs.Teff)
    plt.savefig('fig/vs%d.pdf' % vs.Teff)
    plt.close()


def S_curve(Par_min, Par_max, M, alpha, r, mu=0.6, structure='Mesa', abundance='solar', n=100, input='Teff',
            output='Mdot', xscale='log', yscale='log', save=True, path='fig/S-curve.pdf', set_title=False,
            title='S-curve', savedots=True, path_dots='fig/'):

    if xscale not in ['linear', 'log', 'parlog']:
        print('Incorrect xscale, try linear, log or parlog')
        return
    if yscale not in ['linear', 'log', 'parlog']:
        print('Incorrect yscale, try linear, log or parlog')
        return
    Sigma_plot = []
    Plot = []
    Add_Plot = np.zeros(n)
    a = 0
    key = True

    h = (G * M * r) ** (1 / 2)

    for i, Par in enumerate(np.geomspace(Par_max, Par_min, n)):

        if input == 'Teff':
            Teff = Par
            F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Par ** 4
            Mdot = F / h
        elif input == 'Mdot':
            Mdot = Par
            F = Par * h
            Teff = (3 / (8 * np.pi) * (G * M) ** 4 * F / (sigmaSB * h ** 7)) ** (1 / 4)
        elif input == 'F':
            F = Par
            Mdot = Par / h
            Teff = (3 / (8 * np.pi) * (G * M) ** 4 * Par / (sigmaSB * h ** 7)) ** (1 / 4)
        else:
            print('Incorrect input, try Teff, Mdot of F')
            return

        if structure == 'Kramers':
            vs = IdealKramersVerticalStructure(M, alpha, r, F, mu=mu)
        elif structure == 'BellLin':
            vs = IdealBellLin1994VerticalStructure(M, alpha, r, F, mu=mu)
        elif structure == 'Mesa':
            try:
                if np.isnan(mesa_vs):
                    raise ModuleNotFoundError('Mesa2py is not installed')
            except TypeError:
                vs = mesa_vs.MesaVerticalStructure(M, alpha, r, F, abundance=abundance)
        elif structure == 'MesaIdeal':
            try:
                if np.isnan(mesa_vs):
                    raise ModuleNotFoundError('Mesa2py is not installed')
            except TypeError:
                vs = mesa_vs.MesaIdealVerticalStructure(M, alpha, r, F, mu=mu, abundance=abundance)
        elif structure == 'MesaAd':
            try:
                if np.isnan(mesa_vs):
                    raise ModuleNotFoundError('Mesa2py is not installed')
            except TypeError:
                vs = mesa_vs.MesaVerticalStructureAdiabatic(M, alpha, r, F, abundance=abundance)
        elif structure == 'MesaFirst':
            try:
                if np.isnan(mesa_vs):
                    raise ModuleNotFoundError('Mesa2py is not installed')
            except TypeError:
                vs = mesa_vs.MesaVerticalStructureFirstAssumption(M, alpha, r, F, abundance=abundance)
        elif structure == 'MesaRadConv':
            try:
                if np.isnan(mesa_vs):
                    raise ModuleNotFoundError('Mesa2py is not installed')
            except TypeError:
                vs = mesa_vs.MesaVerticalStructureRadConv(M, alpha, r, F, abundance=abundance)
        elif structure == 'MesaRadConvIrr':
            try:
                if np.isnan(mesa_vs):
                    raise ModuleNotFoundError('Mesa2py is not installed')
            except TypeError:
                vs = mesa_vs.MesaVerticalStructureRadConvExternalIrradiation(M, alpha, r, F, abundance=abundance)
        else:
            print('Incorrect structure, try Kramers, BellLin, Mesa, MesaIdeal, MesaAd, MesaFirst, MesaRadConv or '
                  'MesaRadConvIrr')
            return

        z0r, result = vs.fit()
        print('Teff = {:g}, tau = {:g}, z0r = {:g}'.format(Teff, vs.tau(), z0r))

        y = vs.parameters_C()
        Sigma_plot.append(y[4])

        if output == 'Teff':
            Plot.append(Teff)
        elif output == 'Mdot':
            Plot.append(Mdot)
        elif output == 'Mdot/Mdot_edd':
            Mdot_edd = 1.7e18 * M / M_sun
            Plot.append(Mdot / Mdot_edd)
        elif output == 'F':
            Plot.append(F)
        elif output == 'z0r':
            Plot.append(z0r)
        else:
            print('Incorrect output, try Teff, Mdot, Mdot/Mdot_edd, F or z0r')
            return

        delta = y[3] - 4 * sigmaSB / (3 * c) * y[2] ** 4
        delta = delta / y[3]
        if delta > 0.0 and key:
            a = i
            key = False
        Add_Plot[i] = delta
        print(delta)

        print(i + 1)

    print('(P_C - P_rad_C) / P_C =', Add_Plot.min())

    label = r'M = {:g} Msun, r = {:g} cm, alpha = {:g}'.format(M / M_sun, r, alpha)

    xlabel = r'$\Sigma_0, \, g/cm^2$'
    if xscale == 'parlog' and yscale == 'parlog':
        plt.plot(np.log10(Sigma_plot[:a+1]), np.log10(Plot[:a+1]), label='$Pgas < Prad$',  color='b')
        plt.plot(np.log10(Sigma_plot[a:]), np.log10(Plot[a:]), label='$Pgas > Prad$', color='g')
        xlabel = r'$log \,$' + xlabel
    elif xscale == 'parlog':
        plt.plot(np.log10(Sigma_plot), Plot, label=label)
        xlabel = r'$log \,$' + xlabel
        plt.yscale(yscale)
    elif yscale == 'parlog':
        plt.plot(Sigma_plot, np.log10(Plot), label=label)
        plt.xscale(xscale)
    else:
        plt.plot(Sigma_plot, Plot, label=label)
        plt.xscale(xscale)
        plt.yscale(yscale)

    plt.xlabel(xlabel)

    if savedots:
        np.savetxt(path_dots + 'Sigma_0.txt', Sigma_plot)
        np.savetxt(path_dots + output + '.txt', Plot)

    # rg = 2 * G * M / c ** 2
    # plt.plot(Sigma_plot[Add_Plot.argmin():], Plot[Add_Plot.argmin():],
    #          label=r'M = {:g} Msun, r = {:g} rg'.format(M / M_sun, r / rg))

    # plt.text(2e4, 3.7e3, 'BC', rotation=-11, wrap=True)

    ### plt.plot(Sigma_plot[:a], Plot[:a], 'g-', label='BC')
    ### plt.plot(Sigma_plot[a:b], Plot[a:b], 'c--', label='Partial ionization, instability')
    ### plt.plot(Sigma_plot[b:], Plot[b:], 'b-', label='Cold disc')

    # if structure == 'MesaIdeal':
    #      structure = r'Mesa + ideal gas, $\mu$ = %g' % mu
    # elif structure == 'Kramers':
    #     structure = r'Kramers + ideal gas, $\mu$ = %g' % mu
    # plt.plot(Sigma_plot, Plot, label=structure)
    #
    # Mdot_up_A = 2545454545454.5454 * 1.0 * r / Mdot_edd
    # Sigma_up_A = 1.4e19 * (M / M_sun) * alpha ** (-1) * (r / rg) ** (3 / 2) / (Mdot_up_A * Mdot_edd)
    # Teff_up_A = (3 / (8 * np.pi ) * (G * M) ** 4 * Mdot_up_A * Mdot_edd / (sigmaSB * h ** 6)) ** (1 / 4)

    # plt.plot([Sigma_plot[Add_Plot.argmin()], Sigma_up_A], [Plot[Add_Plot.argmin()], Mdot_up_A], '--')

    # Mdot_up_AA = 2 * np.pi * alpha * h * 100 * Sigma_up_A / Mdot_edd
    # Teff_up_AA = (3 / (8 * np.pi) * (G * M) ** 4 * Mdot_up_AA * Mdot_edd / (sigmaSB * h ** 6)) ** (1 / 4)

    # plt.plot([Sigma_up_A, 100 * Sigma_up_A], [Mdot_up_A, Mdot_up_AA], '--')
    #
    # plt.scatter(Sigma_plot[Add_Plot.argmin()], Plot[Add_Plot.argmin()])

    if output == 'Teff':
        ylabel = r'$T_{\rm eff}, \, K$'
    elif output == 'Mdot':
        ylabel = r'$\dot{M}, \, g/s$'
    elif output == 'Mdot/Mdot_edd':
        ylabel = r'$\dot{M}/\dot{M}_{edd} $'
    elif output == 'F':
        ylabel = r'$F, \, g~cm^2$'
    elif output == 'z0r':
        ylabel = r'$z_0/r$'
    if yscale == 'parlog':
        ylabel = r'$log \,$' + ylabel
    plt.ylabel(ylabel)
    plt.grid(True, which='both', ls='-')
    plt.legend()
    if set_title:
        plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(path)
        plt.close()


def TempGrad_Plot(vs, abundance='solar', title=True):
    vs.fit()
    n = 1000
    t = np.linspace(0, 1, n)
    y = vs.integrate(t)[0]
    S, P, Q, T = y
    grad_plot = InterpolatedUnivariateSpline(np.log(P), np.log(T)).derivative()
    try:
        from opacity import Opac
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError('Mesa2py is not installed') from e
    mesaop = Opac(abundance, mesa_dir='/mesa')
    rho, eos = mesaop.rho(P * vs.P_norm, T * vs.T_norm, True)
    ion = np.exp(eos.lnfree_e)
    kappa = vs.opacity(y)
    plt.plot(1 - t, grad_plot(np.log(P)), label=r'$\nabla_{rad}$')
    plt.plot(1 - t, eos.grad_ad, label=r'$\nabla_{ad}$')
    plt.plot(1 - t, T * vs.T_norm / 2e5, label='T / 2e5K')
    plt.plot(1 - t, ion, label='free e')
    plt.plot(1 - t, kappa / (3 * kappa[-1]), label=r'$\varkappa / 3\varkappa_C$')
    plt.legend()
    plt.xlabel('$z / z_0$')
    if title:
        plt.title(r'$\frac{d(lnT)}{d(lnP)}, T_{\rm eff} = %d K$' % vs.Teff)
    # plt.hlines(0.4, *plt.xlim(), linestyles='--')
    plt.grid()
    plt.savefig('fig/TempGrad%d.pdf' % vs.Teff)
    plt.close()
    conv_param_sigma = simps(2 * rho * (grad_plot(np.log(P)) > eos.grad_ad), t * vs.z0) / (S[-1] * vs.sigma_norm)
    conv_param_z0 = simps(grad_plot(np.log(P)) > eos.grad_ad, t * vs.z0) / vs.z0
    print('Convective parameter (sigma) = ', conv_param_sigma)
    print('Convective parameter (z_0) = ', conv_param_z0)
    return conv_param_z0


def Opacity_Plot(Par_min, Par_max, M, alpha, r, mu=0.6, structure='Mesa', abundance='solar', n=100, input='Teff',
                 save=True,
                 path='fig/Opacity.pdf', title=True):
    T_C_plot = []
    Opacity_Plot = []
    k = 1
    kk = 1
    a = 0
    b = 0

    h = (G * M * r) ** (1 / 2)

    for i, Par in enumerate(np.geomspace(Par_max, Par_min, n)):

        if input == 'Teff':
            Teff = Par
            F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Par ** 4
            Mdot = F / h
        elif input == 'Mdot':
            Mdot = Par
            F = Par * h
            Teff = (3 / (8 * np.pi) * (G * M) ** 4 * F / (sigmaSB * h ** 7)) ** (1 / 4)
        elif input == 'F':
            F = Par
            Mdot = Par / h
            Teff = (3 / (8 * np.pi) * (G * M) ** 4 * Par / (sigmaSB * h ** 7)) ** (1 / 4)
        else:
            print('Incorrect input, try Teff, Mdot of F')
            return

        if structure == 'Kramers':
            vs = IdealKramersVerticalStructure(M, alpha, r, F, mu=mu)
        elif structure == 'BellLin':
            vs = IdealBellLin1994VerticalStructure(M, alpha, r, F, mu=mu)
        elif structure == 'Mesa':
            try:
                if np.isnan(mesa_vs):
                    raise ModuleNotFoundError('Mesa2py is not installed')
            except TypeError:
                vs = mesa_vs.MesaVerticalStructure(M, alpha, r, F, abundance=abundance)
        elif structure == 'MesaIdeal':
            try:
                if np.isnan(mesa_vs):
                    raise ModuleNotFoundError('Mesa2py is not installed')
            except TypeError:
                vs = mesa_vs.MesaIdealVerticalStructure(M, alpha, r, F, mu=mu, abundance=abundance)
        elif structure == 'MesaAd':
            try:
                if np.isnan(mesa_vs):
                    raise ModuleNotFoundError('Mesa2py is not installed')
            except TypeError:
                vs = mesa_vs.MesaVerticalStructureAdiabatic(M, alpha, r, F, abundance=abundance)
        elif structure == 'MesaFirst':
            try:
                if np.isnan(mesa_vs):
                    raise ModuleNotFoundError('Mesa2py is not installed')
            except TypeError:
                vs = mesa_vs.MesaVerticalStructureFirstAssumption(M, alpha, r, F, abundance=abundance)
        elif structure == 'MesaRadConv':
            try:
                if np.isnan(mesa_vs):
                    raise ModuleNotFoundError('Mesa2py is not installed')
            except TypeError:
                vs = mesa_vs.MesaVerticalStructureRadConv(M, alpha, r, F, abundance=abundance)
        else:
            print('Incorrect structure, try Kramers, BellLin, Mesa, MesaIdeal, MesaAd, MesaFirst or MesaRadConv')
            return

        vs.fit()
        # y = vs.parameters_C()
        # T_C_plot.append(y[2])
        # Opacity_Plot.append(y[0])

        T_C_plot.append(Par)
        rho = 1e-8
        Opacity_Plot.append(vs.law_of_opacity(rho, Par))

        if Teff < 5.6e3 and k == 1:
            a = i
            k = 0
        if Teff < 4.5e3 and kk == 1:
            b = i
            kk = 0

        print(i + 1)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$T, K$')
    plt.ylabel(r'$\varkappa_C, cm^2/g$')
    plt.plot(T_C_plot, Opacity_Plot, label=structure)

    # plt.plot(T_C_plot[:a], Opacity_Plot[:a], 'g-')
    # plt.plot(T_C_plot[a:b], Opacity_Plot[a:b], 'c--')
    # plt.plot(T_C_plot[b:], Opacity_Plot[b:], 'b-')

    plt.grid(True, which='both', ls='-')
    if title:
        plt.title('Opacity')
    plt.tight_layout()
    if save:
        plt.savefig(path)
        plt.close()


def main():
    M = 1.5 * M_sun
    alpha = 0.2
    r = 1e10

    # Structure_Plot(M, alpha, r, 1e4, structure='Mesa')

    h = (G * M * r) ** (1 / 2)

    # for Teff in [2000, 3000, 4000, 5000, 7000, 10000, 12000]:
    #     F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Teff ** 4
    #
    #     vs = MesaVerticalStructure(M, alpha, r, F)
    #     print(Teff, 'K')
    #     print(vs.tau0())
    #     TempGrad_Plot(vs)
    #
    # raise Exception

    S_curve(2e3, 2e4, M, alpha, r, structure='Mesa', n=300, input='Teff', output='Mdot', save=False)
    S_curve(2e3, 2e4, M, alpha, r, structure='MesaAd', n=300, input='Teff', output='Mdot', save=False)
    S_curve(2e3, 2e4, M, alpha, r, structure='MesaFirst', n=300, input='Teff', output='Mdot', save=False)
    S_curve(2e3, 2e4, M, alpha, r, structure='MesaRadConv', n=300, input='Teff', output='Mdot', save=False)

    sigma_down = 74.6 * (alpha / 0.1) ** (-0.83) * (r / 1e10) ** 1.18 * (M / M_sun) ** (-0.4)
    sigma_up = 39.9 * (alpha / 0.1) ** (-0.8) * (r / 1e10) ** 1.11 * (M / M_sun) ** (-0.37)

    mdot_down = 2.64e15 * (alpha / 0.1) ** 0.01 * (r / 1e10) ** 2.58 * (M / M_sun) ** (-0.85)
    mdot_up = 8.07e15 * (alpha / 0.1) ** (-0.01) * (r / 1e10) ** 2.64 * (M / M_sun) ** (-0.89)

    teff_up = 6890 * (r / 1e10) ** (-0.09) * (M / M_sun) ** 0.03
    teff_down = 5210 * (r / 1e10) ** (-0.10) * (M / M_sun) ** 0.04

    plt.scatter(sigma_down, mdot_down)
    plt.scatter(sigma_up, mdot_up)

    plt.savefig('fig/S-curve-all.pdf')


if __name__ == '__main__':
    main()
