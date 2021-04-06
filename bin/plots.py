import numpy as np
from astropy import constants as const
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.integrate import simps
from scipy.interpolate import InterpolatedUnivariateSpline

import vs as vert

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
rcParams['text.latex.preamble'] = r'\usepackage[utf8]{inputenc} \usepackage{amsfonts, amsmath, amsthm, amssymb} ' \
                                  r'\usepackage[english]{babel} '


def StructureChoice(M, alpha, r, Par, input, structure, mu=0.6, abundance='solar'):
    h = np.sqrt(G * M * r)
    rg = 2 * G * M / c ** 2
    r_in = 3 * rg
    func = 1 - np.sqrt(r_in / r)
    if input == 'Teff':
        Teff = Par
        F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Par ** 4
        Mdot = F / (h * func)
    elif input == 'Mdot':
        Mdot = Par
        F = Par * h * func
        Teff = (3 / (8 * np.pi) * (G * M) ** 4 * F / (sigmaSB * h ** 7)) ** (1 / 4)
    elif input == 'F':
        F = Par
        Mdot = Par / (h * func)
        Teff = (3 / (8 * np.pi) * (G * M) ** 4 * Par / (sigmaSB * h ** 7)) ** (1 / 4)
    elif input == 'Mdot_Mdot_edd':
        Mdot = Par * 1.39e18 * M / M_sun
        F = Mdot * h * func
        Teff = (3 / (8 * np.pi) * (G * M) ** 4 * F / (sigmaSB * h ** 7)) ** (1 / 4)
    else:
        print('Incorrect input, try Teff, Mdot, F of Mdot_Mdot_edd')
        raise Exception

    if structure == 'Kramers':
        vs = vert.IdealKramersVerticalStructure(M, alpha, r, F, mu=mu)
    elif structure == 'BellLin':
        vs = vert.IdealBellLin1994VerticalStructure(M, alpha, r, F, mu=mu)
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
    elif structure == 'Prad':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructureRadConvPrad(M, alpha, r, F, abundance=abundance, mu=mu)
    elif structure == 'Prad_BellLin':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = vert.IdealBellLin1994VerticalStructurePrad(M, alpha, r, F, mu=mu)
    else:
        print('Incorrect structure, try Kramers, BellLin, Mesa, MesaIdeal, MesaAd, MesaFirst, MesaRadConv, '
              'Prad or Prad_BellLin')
        raise Exception

    return vs, F, Teff, Mdot


def Convective_parameter(vs):
    n = 1000
    t = np.linspace(0, 1, n)
    y = vs.integrate(t)[0]
    S, P, Q, T = y
    grad_plot = InterpolatedUnivariateSpline(np.log(P), np.log(T)).derivative()
    try:
        rho, eos = vs.mesaop.rho(P * vs.P_norm, T * vs.T_norm, True)
    except AttributeError:
        print('Incorrect vertical structure. Use vertical structure with Mesa EOS.')
        raise Exception
    conv_param_sigma = simps(2 * rho * (grad_plot(np.log(P)) > eos.grad_ad), t * vs.z0) / (
            S[-1] * vs.sigma_norm)
    conv_param_z0 = simps(grad_plot(np.log(P)) > eos.grad_ad, t * vs.z0) / vs.z0
    return conv_param_z0, conv_param_sigma


def Structure_Plot(M, alpha, r, Par, mu=0.6, input='Teff', structure='Kramers', abundance='solar', n=100, savedots=True,
                   path_output='fig/vs.dat', make_pic=True, save_plot=True, path_plot='fig/vs.pdf', set_title=True,
                   title='Vertical structure'):
    vs, F, Teff, Mdot = StructureChoice(M, alpha, r, Par, input, structure, mu, abundance)
    z0r, result = vs.fit()
    rg = 2 * G * M / c ** 2
    print('Teff = ', vs.Teff)
    t = np.linspace(0, 1, n)
    S, P, Q, T = vs.integrate(t)[0]
    varkappa_C, rho_C, T_C, P_C, Sigma0 = vs.parameters_C()
    delta = (4 * sigmaSB) / (3 * c) * T_C ** 4 / P_C
    grad_plot = InterpolatedUnivariateSpline(np.log(P), np.log(T)).derivative()
    rho, eos = vs.law_of_rho(P * vs.P_norm, T * vs.T_norm, True)
    varkappa = vs.law_of_opacity(rho, T * vs.T_norm, lnfree_e=eos.lnfree_e)
    output_arr = np.column_stack((t, S, P, Q, T, rho / rho_C, varkappa / varkappa_C, grad_plot(np.log(P))))
    try:
        output_arr = np.column_stack((output_arr, eos.grad_ad, eos.lnfree_e))
        header = 't\t S\t P\t Q\t T\t rho\t varkappa\t grad\t grad_ad\t lnfree_e'
    except AttributeError:
        header = 't\t S\t P\t Q\t T\t rho\t varkappa\t grad'
    header_input = '\nM = {:e} Msun, alpha = {}, r = {:e} cm, r = {} rg, Teff = {} K, Mdot = {:e} g/s, ' \
                   'F = {:e} dyn*cm, abundance = {}, structure = {}'.format(
                    M / M_sun, alpha, r, r / rg, Teff, Mdot, F, abundance, structure)
    header_C = '\nvarkappa_C = {:e}, rho_C = {:e}, T_C = {:e}, P_C = {:e}, Sigma0 = {:e}, ' \
               'PradPgas = {:e}, z0r = {:e}'.format(varkappa_C, rho_C, T_C, P_C, Sigma0, delta, z0r)
    header_norm = '\nSigma_norm = {:e}, P_norm = {:e}, T_norm = {:e}, Q_norm = {:e}'.format(
                   vs.sigma_norm, vs.P_norm, vs.T_norm, vs.Q_norm)
    header = header + header_input + header_C + header_norm

    if savedots:
        np.savetxt(path_output, output_arr, header=header)
    if not make_pic:
        return
    plt.plot(1 - t, S, label=r'$\hat{\Sigma}$')
    plt.plot(1 - t, P, label=r'$\hat{P}$')
    plt.plot(1 - t, Q, label=r'$\hat{Q}$')
    plt.plot(1 - t, T, label=r'$\hat{T}$')
    plt.grid()
    plt.legend()
    plt.xlabel('$z / z_0$')
    if set_title:
        plt.title(title)
    plt.tight_layout()
    if save_plot:
        plt.savefig(path_plot)
        plt.close()


def S_curve(Par_min, Par_max, M, alpha, r, mu=0.6, structure='Mesa', abundance='solar', n=100, input='Teff',
            output='Mdot', savedots=True, save_indexes=True, path_output='fig/S_curve.dat',
            path_indexes='fig/indexes.txt', make_Pi_table=True, Pi_table_path='fig/Pi.dat',
            make_pic=True, xscale='log', yscale='log', save_plot=True, path_plot='fig/S-curve.pdf',
            set_title=True, title='S-curve'):
    if xscale not in ['linear', 'log', 'parlog']:
        print('Incorrect xscale, try linear, log or parlog')
        raise Exception
    if yscale not in ['linear', 'log', 'parlog']:
        print('Incorrect yscale, try linear, log or parlog')
        raise Exception
    Sigma_plot, Mdot_plot, Teff_plot, F_plot = [], [], [], []
    Output_Plot = []
    varkappa_c_plot, T_c_plot, P_c_plot, rho_c_plot = [], [], [], []
    z0r_plot, tau_plot, PradPgas_Plot = [], [], []
    conv_param_z0_plot, conv_param_sigma_plot = [], []

    PradPgas10_index = 0  # where Prad = Pgas
    PradPgas08_index = 0  # where Prad = 0.8 Pgas
    PradPgas05_index = 0  # where Prad = 0.5 Pgas
    PradPgas04_index = 0  # where Prad = 0.4 Pgas
    tau_index = n  # where tau < 1
    Sigma_minus_index = 0  # where free_e < 0.5, Sigma_minus
    key = True  # for Prad = Pgas
    key_1 = True  # for Prad = 0.8 Pgas
    key_2 = True  # for Prad = 0.5 Pgas
    key_3 = True  # for Prad = 0.4 Pgas
    tau_key = True  # for tau < 1
    Sigma_minus_key = True  # for free_e < 0.5, Sigma_minus
    SigmaKey = True  # for SigmaAB

    Sigma_plus_key = True  # for Sigma_plus
    Sigma_plus_index = 0  # for Sigma_plus
    delta_Sigma_plus = -1

    if make_Pi_table:
        nomer = -1
        with open(Pi_table_path, 'w') as g:
            g.write('#pi1 pi2 pi3 pi4 Prad/Pgas r/Rin setNo Mx alpha\n')

    for i, Par in enumerate(np.geomspace(Par_max, Par_min, n)):
        vs, F, Teff, Mdot = StructureChoice(M, alpha, r, Par, input, structure, mu, abundance)
        z0r, result = vs.fit()

        print('Teff = {:g}, tau = {:g}, z0r = {:g}'.format(Teff, vs.tau(), z0r))

        y = vs.parameters_C()
        Sigma_plot.append(y[4])
        varkappa_c_plot.append(y[0])
        T_c_plot.append(y[2])
        rho_c_plot.append(y[1])
        z0r_plot.append(z0r)
        tau_plot.append(vs.tau())
        P_c_plot.append(y[3])
        Mdot_plot.append(Mdot)
        Teff_plot.append(Teff)
        F_plot.append(F)

        if i == 0:
            sigma_temp = y[4]
        else:
            delta_Sigma_plus = y[4] - sigma_temp
            sigma_temp = y[4]

        if make_pic:
            if output == 'Teff':
                Output_Plot.append(Teff)
            elif output == 'Mdot':
                Output_Plot.append(Mdot)
            elif output == 'Mdot_Mdot_edd':
                Mdot_edd = 1.39e18 * M / M_sun
                Output_Plot.append(Mdot / Mdot_edd)
            elif output == 'F':
                Output_Plot.append(F)
            elif output == 'z0r':
                Output_Plot.append(z0r)
            elif output == 'T_C':
                Output_Plot.append(y[2])
            else:
                print('Incorrect output, try Teff, Mdot, Mdot_Mdot_edd, F or z0r')
                raise Exception

        delta = (4 * sigmaSB) / (3 * c) * y[2] ** 4 / y[3]
        PradPgas_Plot.append(delta)

        Ledd = 1.25e38 * (M / M_sun)
        Pi1 = 7
        Pi2 = 0.5
        Pi3 = 1.15
        Pi4 = 0.46
        omegaK = np.sqrt(G * M / r ** 3)

        SigmaAB_teor = Pi3 * Pi4 ** 2 / (Pi1 * Pi2 ** 2) * 32 / (9 * np.pi * alpha * omegaK) * (
                Ledd / (G * M)) ** 2 * 1 / Mdot

        delta_SigmaAB = (y[4] - SigmaAB_teor) / y[4]
        # print(delta_SigmaAB)
        if abs(delta_SigmaAB) < 0.1 and SigmaKey:
            SigmaAB = y[4]
            # print('Yeah!!!')
            SigmaKey = False

        if make_Pi_table:
            pi1, pi2, pi3, pi4 = vs.Pi_finder()
            rg = 2 * G * M / c ** 2
            string_for_Pi = str(pi1) + ' ' + str(pi2) + ' ' + str(pi3) + ' ' + str(pi4) + ' ' + str(delta) + ' ' + str(
                r / (3 * rg)) + ' ' + str(nomer) + ' ' + str(M / M_sun) + ' ' + str(alpha) + '\n'
            nomer -= 1
            with open(Pi_table_path, 'a') as g:
                g.write(string_for_Pi)

        if delta < 1.0 and key:
            PradPgas10_index = i
            key = False
            Pi = vs.Pi_finder()
            print('P_rad_C / P_gas_C =', delta, '\n', Pi)
            print('index =', i)
            print(y)
            string_for_Pi_1 = str(Pi) + ' ' + str(i) + ' ' + str(delta)
        if delta < 0.8 and key_1:
            PradPgas08_index = i
            key_1 = False
            Pi = vs.Pi_finder()
            print('P_rad_C / P_gas_C =', delta, '\n', Pi)
            print('index =', i)
            print(y)
            string_for_Pi_2 = str(Pi) + ' ' + str(i) + ' ' + str(delta)
        if delta < 0.5 and key_2:
            PradPgas05_index = i
            key_2 = False
            Pi = vs.Pi_finder()
            print('P_rad_C / P_gas_C =', delta, '\n', Pi)
            print('index =', i)
            print(y)
            string_for_Pi_3 = str(Pi) + ' ' + str(i) + ' ' + str(delta)
            # return string_for_Pi_1, string_for_Pi_2, string_for_Pi_3
        if delta < 0.4 and key_3:
            PradPgas04_index = i
            key_3 = False
            Pi = vs.Pi_finder()
            print('P_rad_C / P_gas_C =', delta, '\n', Pi)
            print('index =', i)
            print(y)
        if delta_Sigma_plus > 0.0 and Sigma_plus_key:  # search Sigma_plus
            Sigma_plus_index = i
            Sigma_plus_key = False
            # Sigma_plus = y[4]
            # return Sigma_plus, Mdot, z0r
        print('delta_Sigma_plus = ', delta_Sigma_plus)
        if vs.tau() < 1 and tau_key:
            tau_index = i
            tau_key = False
            break
        if structure not in ['Kramers', 'BellLin', 'Prad_BellLin'] and Sigma_minus_key:
            rho, eos = vs.mesaop.rho(y[3], y[2], full_output=True)
            free_e = np.exp(eos.lnfree_e)
            if free_e < (1 + vs.mesaop.X) / 4 and Sigma_minus_key:
                Sigma_minus_index = i
                Sigma_minus_key = False
                # try:
                #     return SigmaAB, Sigma_plus, y[4], Mdot, string_for_Pi_1, string_for_Pi_2, string_for_Pi_3
                # except UnboundLocalError:
                #     SigmaAB = np.nan
                #     print('Ouch!!!')
                #     return SigmaAB, Sigma_plus, y[4], Mdot, string_for_Pi_1, string_for_Pi_2, string_for_Pi_3
            # conv_param_z0, conv_param_sigma = Convective_parameter(vs)
            # conv_param_z0_plot.append(conv_param_z0)
            # conv_param_sigma_plot.append(conv_param_sigma)
        print(i + 1)

    # if Sigma_plus_index == 0:  # search Sigma_plus
    #     return np.nan

    if save_indexes:
        indexes = [PradPgas10_index, PradPgas08_index, PradPgas05_index, PradPgas04_index,
                   tau_index, Sigma_plus_index, Sigma_minus_index]
        np.savetxt(path_indexes, indexes, fmt='%d',
                   header='PradPgas10_index \nPradPgas08_index \nPradPgas05_index \nPradPgas04_index \ntau_index '
                          '\nSigma_plus_index \nSigma_minus_index')

    if savedots:
        rg = 2 * G * M / c ** 2
        header = 'Sigma0 \tTeff \tMdot \tF \tz0r \trho_c \tT_c \tP_c \ttau \tPradPgas \tvarkappa_c' + \
                 '\nM = {:e} Msun, alpha = {}, r = {:e} cm, r = {} rg, abundance = {}, structure = {}'.format(
                     M / M_sun, alpha, r, r / rg, abundance, structure)
        np.savetxt(path_output, np.column_stack([Sigma_plot, Teff_plot, Mdot_plot, F_plot,
                                                 z0r_plot, rho_c_plot, T_c_plot, P_c_plot,
                                                 tau_plot, PradPgas_Plot, varkappa_c_plot]), header=header)

    if not make_pic:
        if structure not in ['Kramers', 'BellLin', 'Prad_BellLin']:
            return conv_param_z0_plot, conv_param_sigma_plot
        else:
            return

    xlabel = r'$\Sigma_0, \, g/cm^2$'

    if xscale == 'parlog':
        Sigma_plot = np.log10(Sigma_plot)
        xlabel = r'$log \,$' + xlabel
    if yscale == 'parlog':
        Output_Plot = np.log10(Output_Plot)

    pl, = plt.plot(Sigma_plot[:tau_index + 1], Output_Plot[:tau_index + 1],
                   label=r'$P_{{\rm gas}} > P_{{\rm rad}}, \alpha = {:g}$'.format(alpha))
    plt.plot(Sigma_plot[tau_index:], Output_Plot[tau_index:], color=pl.get_c(), alpha=0.5)
    if PradPgas10_index != 0:
        plt.plot(Sigma_plot[:PradPgas10_index + 1], Output_Plot[:PradPgas10_index + 1],
                 label=r'$P_{\rm gas} < P_{\rm rad}$')
    plt.scatter(Sigma_plot[Sigma_minus_index], Output_Plot[Sigma_minus_index], s=20, color=pl.get_c())
    plt.scatter(Sigma_plot[Sigma_plus_index], Output_Plot[Sigma_plus_index], s=20, color=pl.get_c())

    if xscale != 'parlog':
        plt.xscale(xscale)
    if yscale != 'parlog':
        plt.yscale(yscale)

    if output == 'Teff':
        ylabel = r'$T_{\rm eff}, \, K$'
    elif output == 'Mdot':
        ylabel = r'$\dot{M}, \, g/s$'
    elif output == 'Mdot_Mdot_edd':
        ylabel = r'$\dot{M}/\dot{M}_{edd} $'
    elif output == 'F':
        ylabel = r'$F, \, g~cm^2$'
    elif output == 'z0r':
        ylabel = r'$z_0/r$'
    elif output == 'T_C':
        ylabel = r'$T_{\rm c}, K$'
    if yscale == 'parlog':
        ylabel = r'$log \,$' + ylabel
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True, which='both', ls='-')
    plt.legend()
    if set_title:
        plt.title(title)
    plt.tight_layout()
    if save_plot:
        plt.savefig(path_plot)
        plt.close()


def TempGrad_Plot(M, alpha, r, Par, mu=0.6, input='Teff', structure='Mesa', abundance='solar', path='fig/TempGrad.pdf',
                  set_title=True, title=r'$\frac{d(lnT)}{d(lnP)}$'):
    if structure not in ['Kramers', 'BellLin', 'Mesa', 'MesaIdeal', 'Prad_BellLin']:
        print('Incorrect structure, try Kramers, BellLin, Mesa, MesaIdeal, Prad_BellLin')
        raise Exception
    vs, F, Teff, Mdot = StructureChoice(M, alpha, r, Par, input, structure, mu, abundance)
    vs.fit()

    conv_param_z0, conv_param_sigma = Convective_parameter(vs)

    n = 1000
    t = np.linspace(0, 1, n)
    y = vs.integrate(t)[0]
    S, P, Q, T = y
    grad_plot = InterpolatedUnivariateSpline(np.log(P), np.log(T)).derivative()
    rho, eos = vs.mesaop.rho(P * vs.P_norm, T * vs.T_norm, True)
    ion = np.exp(eos.lnfree_e)
    varkappa = vs.opacity(y)
    plt.plot(1 - t, grad_plot(np.log(P)), label=r'$\nabla_{rad}$')
    plt.plot(1 - t, eos.grad_ad, label=r'$\nabla_{ad}$')
    plt.plot(1 - t, T * vs.T_norm / 2e5, label='T / 2e5K')
    plt.plot(1 - t, ion, label='free e')
    plt.plot(1 - t, varkappa / (3 * varkappa[-1]), label=r'$\varkappa / 3\varkappa_C$')
    plt.legend()
    plt.xlabel('$z / z_0$')
    if set_title:
        plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print('Convective parameter (sigma) = ', conv_param_sigma)
    print('Convective parameter (z_0) = ', conv_param_z0)
    return conv_param_z0, conv_param_sigma


def main():
    M = 1.5 * M_sun
    alpha = 0.2
    r = 1e10

    # Structure_Plot(M, alpha, r, 1e4, structure='Mesa')

    # h = np.sqrt(G * M * r)

    # for Teff in [2000, 3000, 4000, 5000, 7000, 10000, 12000]:
    #     F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Teff ** 4
    #
    #     vs = MesaVerticalStructure(M, alpha, r, F)
    #     print(Teff, 'K')
    #     print(vs.tau0())
    #     TempGrad_Plot(vs)
    #
    # raise Exception

    # S_curve(2e3, 2e4, M, alpha, r, structure='Mesa', n=300, input='Teff', output='Mdot', make_pic=True,
    #             save_plot=False)
    # S_curve(2e3, 2e4, M, alpha, r, structure='MesaAd', n=300, input='Teff', output='Mdot', make_pic=True,
    #             save_plot=False)
    # S_curve(2e3, 2e4, M, alpha, r, structure='MesaFirst', n=300, input='Teff', output='Mdot', make_pic=True,
    #             save_plot=False)
    S_curve(9400, 2e4, M, alpha, r, structure='MesaRadConv', abundance={'he4': 1.0}, n=300, input='Teff', output='Mdot',
            savedots=False, save_indexes=False, make_Pi_table=False, make_pic=True, save_plot=False)

    # sigma_down = 74.6 * (alpha / 0.1) ** (-0.83) * (r / 1e10) ** 1.18 * (M / M_sun) ** (-0.4)
    # sigma_up = 39.9 * (alpha / 0.1) ** (-0.8) * (r / 1e10) ** 1.11 * (M / M_sun) ** (-0.37)
    #
    # mdot_down = 2.64e15 * (alpha / 0.1) ** 0.01 * (r / 1e10) ** 2.58 * (M / M_sun) ** (-0.85)
    # mdot_up = 8.07e15 * (alpha / 0.1) ** (-0.01) * (r / 1e10) ** 2.64 * (M / M_sun) ** (-0.89)
    #
    # teff_up = 6890 * (r / 1e10) ** (-0.09) * (M / M_sun) ** 0.03
    # teff_down = 5210 * (r / 1e10) ** (-0.10) * (M / M_sun) ** 0.04

    sigma_down = 1770 * (alpha / 0.1) ** (-0.83) * (r / 1e10) ** 1.20 * (M / M_sun) ** (-0.4)
    sigma_up = 589 * (alpha / 0.1) ** (-0.78) * (r / 1e10) ** 1.07 * (M / M_sun) ** (-0.36)

    mdot_down = 3.18e16 * (alpha / 0.1) ** (-0.01) * (r / 1e10) ** 2.65 * (M / M_sun) ** (-0.88)
    mdot_up = 1.05e17 * (alpha / 0.1) ** (-0.05) * (r / 1e10) ** 2.69 * (M / M_sun) ** (-0.90)

    plt.scatter(sigma_down, mdot_down)
    plt.scatter(sigma_up, mdot_up)

    plt.savefig('fig/S-curve-all.pdf')


if __name__ == '__main__':
    main()
