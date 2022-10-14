#!/usr/bin/env python3
"""
Module contains functions that calculate vertical structure and S-curve. Functions return tables with calculated data
and make plots of structure or S-curve.

Structure_Plot -- calculates vertical structure and makes table with disc parameters as functions of vertical
    coordinate. Table also contains input parameters of structure, parameters in the symmetry plane and
    parameter normalisations. Also makes a plot of structure (if 'make_pic' parameter is True).
S_curve -- Calculates S-curve and makes table with disc parameters on the S-curve.
    Table contains input parameters of system, surface density Sigma0, viscous torque F,
    accretion rate Mdot, effective temperature Teff, geometrical half-thickness of the disc z0r,
    parameters in the symmetry plane of disc on the S-curve.
    Also makes a plot of S-curve (if 'make_pic' parameter is True).
Radial_Plot -- Calculates radial structure of disc. Return table, which contains input parameters of the system,
    surface density Sigma0, viscous torque F, accretion rate Mdot, effective temperature Teff,
    geometrical half-thickness of the disc z0r and parameters in the symmetry plane of disc
    as functions of radius.

"""
import os

import numpy as np
from astropy import constants as const
from matplotlib import pyplot as plt
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


def StructureChoice(M, alpha, r, Par, input, structure, mu=0.6, abundance='solar', nu_irr=None, L_X_irr=None,
                    spectrum_irr=None, spectrum_irr_par=None, args_spectrum_irr=(), kwargs_spectrum_irr={},
                    C_irr=None, T_irr=None, cos_theta_irr=None, cos_theta_irr_exp=1 / 12, P_ph_0=None):
    """
    Initialize the chosen vertical structure class.

    Parameters
    ----------
    M, alpha, r, Par, input
        Base parameters of structure.
    mu, abundance
        Parameters that describe the chemical composition (ideal gas, tabular values).
    nu_irr, L_X_irr, spectrum_irr, spectrum_irr_par
        Additional parameters in case of advanced external irradiation scheme from (Mescheryakov et al. 2011).
    args_spectrum_irr, kwargs_spectrum_irr, cos_theta_irr, cos_theta_irr_exp
        Additional parameters in case of advanced external irradiation scheme from (Mescheryakov et al. 2011).
    C_irr, T_irr
        Additional parameters in case of simple external irradiation scheme.
    P_ph_0
        Additional parameter in case of external irradiation.
    structure : str
        Type of vertical structure. Possible options are:
        'Kramers', 'BellLin' -- ideal gas EoS, analytical opacities and radiative temperature gradient;
        'Mesa', 'MesaAd', 'MesaRadAd', 'MesaRadConv' -- MESA EoS and opacities and
                                                        radiative, adiabatic, rad+ad
                                                        and rad+conv temperature gradient;
        'MesaIdealGas' -- ideal gas EoS, MESA opacities and radiative temperature gradient;
        'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero' -- simple external irradiation scheme
                                                                   via T_irr or C_irr
        'MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr' -- advanced external irradiation scheme
                                                       from (Mescheryakov et al. 2011).

    Returns
    -------
    vs : vertical structure
        Chosen vertical structure.
    F : double
        Viscous torque in g*cm^2/s^2.
    Teff : double
        Effective temperature in Kelvins.
    Mdot : double
        Accretion rate in g/s.

    """
    h = np.sqrt(G * M * r)
    rg = 2 * G * M / c ** 2
    r_in = 3 * rg
    if r <= r_in:
        raise Exception('Radius r should be greater than inner radius r_in = 3*rg. '
                        'Actual radius r = {:g} rg'.format(r / rg))
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
    elif input == 'Mdot_Msun_yr':
        Mdot = Par * M_sun / 31557600.0
        F = Mdot * h * func
        Teff = (3 / (8 * np.pi) * (G * M) ** 4 * F / (sigmaSB * h ** 7)) ** (1 / 4)
    else:
        raise Exception('Incorrect input, try Teff, Mdot, F of Mdot_Mdot_edd')

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
    elif structure == 'MesaIdealGas':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaIdealGasVerticalStructure(M, alpha, r, F, mu=mu, abundance=abundance)
    elif structure == 'MesaAd':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructureAd(M, alpha, r, F, abundance=abundance)
    elif structure == 'MesaRadAd':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructureRadAd(M, alpha, r, F, abundance=abundance)
    elif structure == 'MesaRadConv':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructureRadConv(M, alpha, r, F, abundance=abundance)
    elif structure == 'MesaIrrZero':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructureExternalIrradiationZeroAssumption(M, alpha, r, F, C_irr=C_irr,
                                                                                T_irr=T_irr, abundance=abundance,
                                                                                P_ph_0=P_ph_0)
    elif structure == 'MesaRadAdIrrZero':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructureRadAdExternalIrradiationZeroAssumption(M, alpha, r, F, C_irr=C_irr,
                                                                                     T_irr=T_irr, abundance=abundance,
                                                                                     P_ph_0=P_ph_0)
    elif structure == 'MesaRadConvIrrZero':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructureRadConvExternalIrradiationZeroAssumption(M, alpha, r, F, C_irr=C_irr,
                                                                                       T_irr=T_irr, abundance=abundance,
                                                                                       P_ph_0=P_ph_0)
    elif structure == 'MesaIrr':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            if L_X_irr is None:
                eta = 0.1
                L_X_irr = eta * Mdot * c ** 2
            vs = mesa_vs.MesaVerticalStructureExternalIrradiation(M, alpha, r, F, nu_irr=nu_irr,
                                                                  L_X_irr=L_X_irr, spectrum_irr=spectrum_irr,
                                                                  spectrum_irr_par=spectrum_irr_par,
                                                                  args_spectrum_irr=args_spectrum_irr,
                                                                  kwargs_spectrum_irr=kwargs_spectrum_irr,
                                                                  cos_theta_irr=cos_theta_irr,
                                                                  cos_theta_irr_exp=cos_theta_irr_exp,
                                                                  abundance=abundance, P_ph_0=P_ph_0)
    elif structure == 'MesaRadAdIrr':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            if L_X_irr is None:
                eta = 0.1
                L_X_irr = eta * Mdot * c ** 2
            vs = mesa_vs.MesaVerticalStructureRadAdExternalIrradiation(M, alpha, r, F, nu_irr=nu_irr,
                                                                       L_X_irr=L_X_irr, spectrum_irr=spectrum_irr,
                                                                       spectrum_irr_par=spectrum_irr_par,
                                                                       args_spectrum_irr=args_spectrum_irr,
                                                                       kwargs_spectrum_irr=kwargs_spectrum_irr,
                                                                       cos_theta_irr=cos_theta_irr,
                                                                       cos_theta_irr_exp=cos_theta_irr_exp,
                                                                       abundance=abundance, P_ph_0=P_ph_0)
    elif structure == 'MesaRadConvIrr':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            if L_X_irr is None:
                eta = 0.1
                L_X_irr = eta * Mdot * c ** 2
            vs = mesa_vs.MesaVerticalStructureRadConvExternalIrradiation(M, alpha, r, F, nu_irr=nu_irr,
                                                                         L_X_irr=L_X_irr, spectrum_irr=spectrum_irr,
                                                                         spectrum_irr_par=spectrum_irr_par,
                                                                         args_spectrum_irr=args_spectrum_irr,
                                                                         kwargs_spectrum_irr=kwargs_spectrum_irr,
                                                                         cos_theta_irr=cos_theta_irr,
                                                                         cos_theta_irr_exp=cos_theta_irr_exp,
                                                                         abundance=abundance, P_ph_0=P_ph_0)
    else:
        raise Exception("Incorrect structure. Possible options are: 'Kramers', 'BellLin',\n"
                        "'Mesa', 'MesaAd', 'MesaRadAd', 'MesaRadConv', 'MesaIdealGas',\n"
                        "'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero',\n"
                        "'MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'.")

    return vs, F, Teff, Mdot


def Convective_parameter(vs):
    """
    Calculates convective parameter of structure. This parameter shows what part of disc is convective.

    Parameters
    ----------
    vs : vertical structure
        Fitted vertical structure, for which convective parameter is calculated.

    Returns
    -------
    conv_param_z : double
        z_convective / z0, z-fraction of convective region, from 0 to 1.
    conv_param_sigma : double
        sigma_convective / sigma0, mass fraction of convective region, from 0 to 1.

    """
    n = 1000
    t = np.linspace(0, 1, n)
    y = vs.integrate(t)[0]
    S, P, Q, T = y
    grad_plot = InterpolatedUnivariateSpline(np.log(P), np.log(T)).derivative()
    rho, eos = vs.law_of_rho(P * vs.P_norm, T * vs.T_norm, True)
    try:
        _ = eos.c_p
    except AttributeError:
        raise Exception('Incorrect vertical structure. Use vertical structure with MESA EoS.')
    conv_param_sigma = simps(2 * rho * (grad_plot(np.log(P)) > eos.grad_ad), t * vs.z0) / (
            S[-1] * vs.sigma_norm)
    conv_param_z = simps(grad_plot(np.log(P)) > eos.grad_ad, t * vs.z0) / vs.z0
    return conv_param_z, conv_param_sigma


def Structure_Plot(M, alpha, r, Par, input='Teff', mu=0.6, structure='BellLin', abundance='solar', nu_irr=None,
                   L_X_irr=None, spectrum_irr=None, spectrum_irr_par=None, args_spectrum_irr=(), kwargs_spectrum_irr={},
                   cos_theta_irr=None, cos_theta_irr_exp=1 / 12, C_irr=None, T_irr=None,
                   z0r_estimation=None, Sigma0_estimation=None, P_ph_0=None,
                   n=100, add_Pi_values=True, path_dots=None,
                   make_pic=True, path_plot=None, set_title=True, title='Vertical structure'):
    """
    Calculates vertical structure and makes table with disc parameters as functions of vertical coordinate.
    Table also contains input parameters of structure, parameters in the symmetry plane and parameter normalisations.
    Also makes a plot of structure (if 'make_pic' parameter is True).

    Parameters
    ----------
    M : double
        Mass of central object in grams.
    alpha : double
        Alpha parameter for alpha-prescription of viscosity.
    r : double
        Distance from central star (radius in cylindrical coordinate system) in cm.
    Par : double
        Can be viscous torque in g*cm^2/s^2, effective temperature in K, accretion rate in g/s or in eddington limits.
        Choice depends on 'input' parameter.
    input : str
        Define the choice of 'Par' parameter. Can be 'F' (viscous torque), 'Teff' (effective temperature),
        'Mdot' (accretion rate), 'Mdot_Mdot_edd' (Mdot in eddington limits) or 'Mdot_Msun_yr' (Mdot in Msun/yr).
    mu : double
        Mean molecular weight. Use in case of ideal gas EoS.
    structure : str
        Type of vertical structure. Possible options are:
        'Kramers', 'BellLin' -- ideal gas EoS, analytical opacities and radiative temperature gradient;
        'Mesa', 'MesaAd', 'MesaRadAd', 'MesaRadConv' -- MESA EoS and opacities and
                                                        radiative, adiabatic, rad+ad
                                                        and rad+conv temperature gradient;
        'MesaIdealGas' -- ideal gas EoS, MESA opacities and radiative temperature gradient;
        'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero' -- simple external irradiation scheme
                                                                   via T_irr or C_irr
        'MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr' -- advanced external irradiation scheme
                                                       from (Mescheryakov et al. 2011).
    abundance : dict or str
        Chemical composition of disc. Use in case of MESA EoS.
        Format: {'isotope_name': abundance}. For example: {'h1': 0.7, 'he4': 0.3}.
        Use 'solar' str in case of solar composition.
    nu_irr : array-like
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'], nu_irr is the irradiation spectral flux argument.
        Can be (X-ray) frequency (in Hz) or energy (in keV) array for spectral external irradiation flux.
        Choose depends on the 'spectrum_irr_par'.
    spectrum_irr : array-like or callable
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        spectrum_irr is the spectrum of external irradiation flux, i.e.
        the spectral (X-ray) external irradiation flux F_nu_irr = F_irr * spectrum_irr.
        If spectrum_irr is array-like, then its size must be equal to nu_irr.size.
        If spectrum_irr is callable, then
        ``F_nu_irr = F_irr * spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``.
        The normalisation of the spectrum_irr in that case is performed automatically.
    spectrum_irr_par : str
        Defines the irradiation spectral flux argument. Can be 'nu' (frequency in Hz) and 'E_in_keV' (energy in keV).
    args_spectrum_irr, kwargs_spectrum_irr : tuple and dict
        Extra arguments and keyword arguments of spectrum_irr, if it's callable.
        The calling signature is ``spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``
    L_X_irr : double or None
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        L_X_irr is the (X-ray) bolometric luminosity of external irradiation source.
        If None, then ``L_X_irr = 0.1 * Mdot * c ** 2``.
        The irradiation flux ``F_irr = L_X_irr / (4 * pi * r ** 2)``.
    cos_theta_irr : double or None
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'], cos_theta_irr is the cosine of angle
        of incidence for external irradiation flux. If None, ``cos_theta_irr = cos_theta_irr_exp * (z0/r)``.
    cos_theta_irr_exp : double
        If cos_theta_irr is None, ``cos_theta_irr = cos_theta_irr_exp * (z0/r)``.
    C_irr : double
        If structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero'],
        C_irr is the irradiation parameter.
    T_irr : double
        If structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero'],
        T_irr is the irradiation temperature.
    n : int
        Number of dots to calculate.
    z0r_estimation : double
        Start estimation of z0r free parameter to fit the structure. Default is None,
        the estimation is calculated automatically.
    Sigma0_estimation : double
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        it's the start estimation of Sigma0 free parameter to fit the structure.
        Default is None, the estimation is calculated automatically.
    P_ph_0 : double
        If structure contains Irradiation (either irradiation scheme),
        it's the start estimation for pressure at the photosphere (pressure boundary condition).
        Default is None, the estimation is calculated automatically.
    add_Pi_values : bool
        Whether to write Pi-parameters (see Ketsaris & Shakura, 1998) to the output file header.
    path_dots : str
        Where to save data table.
    make_pic : bool
        Whether to make plot of structure.
    path_plot : str
        Where to save structure plot.
    set_title : bool
        Whether to make title of the plot.
    title : str
        The title of the plot.

    """
    if path_dots is None:
        print("ATTENTION: the data wil not be saved, since 'path_dots' is None")
    if make_pic and path_plot is None:
        print("ATTENTION: 'make_pic' == True. "
              "The plot will only be created, but not be saved, since 'path_plot' is None")
    vs, F, Teff, Mdot = StructureChoice(M, alpha, r, Par, input, structure, mu, abundance,
                                        nu_irr=nu_irr, L_X_irr=L_X_irr,
                                        spectrum_irr=spectrum_irr, spectrum_irr_par=spectrum_irr_par,
                                        args_spectrum_irr=args_spectrum_irr, kwargs_spectrum_irr=kwargs_spectrum_irr,
                                        cos_theta_irr=cos_theta_irr, cos_theta_irr_exp=cos_theta_irr_exp,
                                        C_irr=C_irr, T_irr=T_irr, P_ph_0=P_ph_0)
    if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
        result = vs.fit(start_estimation_z0r=z0r_estimation, start_estimation_Sigma0=Sigma0_estimation)
        z0r, sigma_par = result.x
    else:
        z0r, result = vs.fit(start_estimation_z0r=z0r_estimation)
    rg = 2 * G * M / c ** 2
    t = np.linspace(0, 1, n)
    S, P, Q, T = vs.integrate(t)[0]
    varkappa_C, rho_C, T_C, P_C, Sigma0 = vs.parameters_C()
    tau = vs.tau()
    delta = (4 * sigmaSB) / (3 * c) * T_C ** 4 / P_C
    grad_plot = InterpolatedUnivariateSpline(np.log(P), np.log(T)).derivative()
    rho, eos = vs.law_of_rho(P * vs.P_norm, T * vs.T_norm, True)
    varkappa = vs.law_of_opacity(rho, T * vs.T_norm, lnfree_e=eos.lnfree_e, return_grad=False)
    tau_arr = np.array([simps(rho[:i] * varkappa[:i], t[:i] * z0r * r) for i in range(2, n+1)]) + 2 / 3
    tau_arr = np.r_[2/3, tau_arr]
    dots_arr = np.c_[t, S, P, abs(Q), T, rho, varkappa, tau_arr, grad_plot(np.log(P))]
    header_input_irr = ''
    try:
        _ = eos.c_p
        dots_arr = np.c_[dots_arr, eos.grad_ad, np.exp(eos.lnfree_e)]
        header = 't\t S\t P\t Q\t T\t rho\t varkappa\t tau\t grad\t grad_ad\t free_e'
        conv_param_z, conv_param_sigma = Convective_parameter(vs)
        header_conv = '\nconv_param_z = {} \tconv_param_sigma = {}'.format(conv_param_z, conv_param_sigma)
    except AttributeError:
        header = 't\t S\t P\t Q\t T\t rho\t varkappa\t tau\t grad'
        header_conv = ''
    header_input = '\nS, P, Q, T -- normalized values, rho -- in g/cm^3, ' \
                   'varkappa -- in cm^2/g \nt = 1 - z/z0 ' \
                   '\nM = {:e} Msun, alpha = {}, r = {:e} cm, r = {} rg, Teff = {} K, Mdot = {:e} g/s, ' \
                   'F = {:e} g*cm^2/s^2, structure = {}'.format(M / M_sun, alpha, r, r / rg, Teff, Mdot, F, structure)
    if structure in ['Kramers', 'BellLin', 'MesaIdealGas']:
        header_input += ', mu = {}'.format(mu)
    else:
        header_input += ', abundance = {}'.format(abundance)
    if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr',
                     'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
        header_input_irr = '\nT_irr = {:g} K, C_irr = {:g}, QirrQvis = {:g}'.format(vs.T_irr, vs.C_irr,
                                                                                    vs.Q_irr / vs.Q0)
        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
            try:
                cost_func = result.cost * 2
            except AttributeError:
                cost_func = sum(result.fun ** 2)
            header_input_irr += ', cost = {:g}, converged = {}, Sigma_ph = {:g} g/cm^2'.format(
                cost_func, vs.converged, vs.Sigma_ph)
    header_C = '\nvarkappa_C = {:e} cm^2/g, rho_C = {:e} g/cm^3, T_C = {:e} K, P_C = {:e} dyn, Sigma0 = {:e} g/cm^2, ' \
               'PradPgas_C = {:e}, z0r = {:e}, tau = {:e}'.format(varkappa_C, rho_C, T_C, P_C, Sigma0, delta, z0r, tau)
    header_norm = '\nSigma_norm = {:e}, P_norm = {:e}, T_norm = {:e}, Q_norm = {:e}'.format(
        vs.sigma_norm, vs.P_norm, vs.T_norm, vs.Q_norm)
    header = header + header_input + header_input_irr + header_C + header_norm + header_conv
    if add_Pi_values:
        header += '\nPi1 = {:f}, Pi2 = {:f}, Pi3 = {:f}, Pi4 = {:f}'.format(*vs.Pi_finder())
    if path_dots is not None:
        np.savetxt(path_dots, dots_arr, header=header)
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
    if path_plot is not None:
        plt.savefig(path_plot)
        plt.close()


def S_curve(Par_min, Par_max, M, alpha, r, input='Teff', structure='BellLin', mu=0.6, abundance='solar', nu_irr=None,
            L_X_irr=None, spectrum_irr=None, spectrum_irr_par=None, args_spectrum_irr=(), kwargs_spectrum_irr={},
            cos_theta_irr=None, cos_theta_irr_exp=1 / 12, C_irr=None, T_irr=None,
            z0r_start_estimation=None, Sigma0_start_estimation=None,
            n=100, tau_break=True, path_dots=None, add_Pi_values=True,
            make_pic=True, output='Mdot', xscale='log', yscale='log',
            path_plot=None, set_title=True, title='S-curve'):
    """
    Calculates S-curve and makes table with disc parameters on the S-curve.
    Table contains input parameters of system, surface density Sigma0, viscous torque F,
    accretion rate Mdot, effective temperature Teff, geometrical half-thickness of the disc z0r,
    parameters in the symmetry plane of disc on the S-curve.
    Also makes a plot of S-curve (if 'make_pic' parameter is True).

    Parameters
    ----------
    Par_min : double
        The starting value of Par. Par_min and Par_max can be viscous torque in g*cm^2/s^2, effective temperature in K,
        accretion rate in g/s or in eddington limits. Choice depends on 'input' parameter.
    Par_max : double
        The end value of Par. Par_min and Par_max can be viscous torque in g*cm^2/s^2, effective temperature in K,
        accretion rate in g/s or in eddington limits. Choice depends on 'input' parameter.
    M : double
        Mass of central object in grams.
    alpha : double
        Alpha parameter for alpha-prescription of viscosity.
    r : double
        Distance from central star (radius in cylindrical coordinate system) in cm.
    input : str
        Define the choice of 'Par_min' and 'Par_max' parameters.
        Can be 'F' (viscous torque), 'Teff' (effective temperature),
        'Mdot' (accretion rate), 'Mdot_Mdot_edd' (Mdot in eddington limits) or 'Mdot_Msun_yr' (Mdot in Msun/yr).
    structure : str
        Type of vertical structure. Possible options are:
        'Kramers', 'BellLin' -- ideal gas EoS, analytical opacities and radiative temperature gradient;
        'Mesa', 'MesaAd', 'MesaRadAd', 'MesaRadConv' -- MESA EoS and opacities and
                                                        radiative, adiabatic, rad+ad
                                                        and rad+conv temperature gradient;
        'MesaIdealGas' -- ideal gas EoS, MESA opacities and radiative temperature gradient;
        'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero' -- simple external irradiation scheme
                                                                   via T_irr or C_irr
        'MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr' -- advanced external irradiation scheme
                                                       from (Mescheryakov et al. 2011).
    mu : double
        Mean molecular weight. Use in case of ideal gas EoS.
    abundance : dict or str
        Chemical composition of disc. Use in case of MESA EoS.
        Format: {'isotope_name': abundance}. For example: {'h1': 0.7, 'he4': 0.3}.
        Use 'solar' str in case of solar composition.
    nu_irr : array-like
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'], nu_irr is the irradiation spectral flux argument.
        Can be (X-ray) frequency (in Hz) or energy (in keV) array for spectral external irradiation flux.
        Choose depends on the 'spectrum_irr_par'.
    spectrum_irr : array-like or callable
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        spectrum_irr is the spectrum of external irradiation flux, i.e.
        the spectral (X-ray) external irradiation flux F_nu_irr = F_irr * spectrum_irr.
        If spectrum_irr is array-like, then its size must be equal to nu_irr.size.
        If spectrum_irr is callable, then
        ``F_nu_irr = F_irr * spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``.
        The normalisation of the spectrum_irr in that case is performed automatically.
    spectrum_irr_par : str
        Defines the irradiation spectral flux argument. Can be 'nu' (frequency in Hz) and 'E_in_keV' (energy in keV).
    args_spectrum_irr, kwargs_spectrum_irr : tuple and dict
        Extra arguments and keyword arguments of spectrum_irr, if it's callable.
        The calling signature is ``spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``
    L_X_irr : double or None
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        L_X_irr is the (X-ray) bolometric luminosity of external irradiation source.
        If None, then ``L_X_irr = 0.1 * Mdot * c ** 2``.
        The irradiation flux ``F_irr = L_X_irr / (4 * pi * r ** 2)``.
    cos_theta_irr : double or None
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'], cos_theta_irr is the cosine of angle
        of incidence for external irradiation flux. If None, ``cos_theta_irr = cos_theta_irr_exp * (z0/r)``.
    cos_theta_irr_exp : double
        If cos_theta_irr is None, ``cos_theta_irr = cos_theta_irr_exp * (z0/r)``.
    C_irr : double
        If structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero'],
        C_irr is the irradiation parameter.
    T_irr : double
        If structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero'],
        T_irr is the irradiation temperature.
    z0r_start_estimation : double
        Start estimation of z0r free parameter to fit the first point of S-curve.
        Further, z0r estimation of the next point is the z0r value of the previous point.
        Default is None, the start estimation is calculated automatically.
    Sigma0_start_estimation : double
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        it's the start estimation of Sigma0 free parameter to fit the first point of S-curve.
        Further, Sigma0 estimation of the next point is the 2*Sigma0 value of the previous point.
        Default is None, the start estimation is calculated automatically.
    n : int
        Number of dots to calculate.
    tau_break : bool
        Whether to end calculation, when tau<1.
    path_dots : str
        Where to save data table.
    add_Pi_values : bool
        Whether to write Pi-parameters (see Ketsaris & Shakura, 1998) to the output file.
    make_pic : bool
        Whether to make S-curve plot.
    output : str
        In which y-coordinate draw S-curve plot. Can be 'Teff', 'Mdot', 'Mdot_Mdot_edd', 'F', 'z0r' or 'T_C'.
    xscale : str
        Scale of x-axis. Can be 'linear', 'log' or 'parlog' (linear scale, but logarithmic values).
    yscale : str
        Scale of y-axis. Can be 'linear', 'log' or 'parlog' (linear scale, but logarithmic values).
    path_plot : str
        Where to save S-curve plot.
    set_title : bool
        Whether to make title of the plot.
    title : str
        The title of the plot.

    """
    if path_dots is None:
        print("ATTENTION: the data wil not be saved, since 'path_dots' is None")
    if make_pic and path_plot is None:
        print("ATTENTION: 'make_pic' == True. "
              "The plot will only be created, but not be saved, since 'path_plot' is None")
    if xscale not in ['linear', 'log', 'parlog']:
        raise Exception('Incorrect xscale, try linear, log or parlog')
    if yscale not in ['linear', 'log', 'parlog']:
        raise Exception('Incorrect yscale, try linear, log or parlog')
    Sigma_plot = []
    Output_Plot = []

    PradPgas10_index = 0  # where Prad = Pgas
    tau_index = n  # where tau < 1
    Sigma_minus_index = 0  # for Sigma_minus
    key = True  # for Prad = Pgas
    tau_key = True  # for tau < 1
    Sigma_minus_key = True  # for Sigma_minus

    Sigma_plus_key = True  # for Sigma_plus
    Sigma_plus_index = 0  # for Sigma_plus
    delta_Sigma_plus = -1
    z0r_estimation = z0r_start_estimation
    sigma_par_estimation = Sigma0_start_estimation

    sigma_temp = np.infty
    except_fits = 0
    P_ph_0 = None

    if path_dots is not None:
        rg = 2 * G * M / c ** 2
        header = 'Sigma0 \tTeff \tMdot \tF \tz0r \trho_c \tT_c \tP_c \ttau \tPradPgas_c \tvarkappa_c'
        header_end = '\nM = {:e} Msun, alpha = {}, r = {:e} cm, r = {} rg, structure = {}'.format(
            M / M_sun, alpha, r, r / rg, structure)
        if structure in ['Kramers', 'BellLin', 'MesaIdealGas']:
            header_end += ', mu = {}'.format(mu)
        else:
            header_end += ', abundance = {}'.format(abundance)
            header += ' \tfree_e_c \tconv_param_z \tconv_param_sigma'
        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr',
                         'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
            header += ' \tQirrQvis \tT_irr \tC_irr'
        if add_Pi_values:
            header += ' \tPi1 \tPi2 \tPi3 \tPi4'
        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
            header += ' \tcost \tconverged \tSigma_ph'
        header = header + '\nAll values are in CGS units.' + header_end
        np.savetxt(path_dots, [], header=header)

    for i, Par in enumerate(np.geomspace(Par_max, Par_min, n)):
        print(i)
        vs, F, Teff, Mdot = StructureChoice(M, alpha, r, Par, input, structure, mu, abundance,
                                            nu_irr=nu_irr, L_X_irr=L_X_irr,
                                            spectrum_irr=spectrum_irr, spectrum_irr_par=spectrum_irr_par,
                                            args_spectrum_irr=args_spectrum_irr,
                                            kwargs_spectrum_irr=kwargs_spectrum_irr,
                                            cos_theta_irr=cos_theta_irr, cos_theta_irr_exp=cos_theta_irr_exp,
                                            C_irr=C_irr, T_irr=T_irr, P_ph_0=P_ph_0)
        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
            try:
                result = vs.fit(start_estimation_z0r=z0r_estimation, start_estimation_Sigma0=sigma_par_estimation)
            except (mesa_vs.NotConvergeError, mesa_vs.PphNotConvergeError):
                print('Except fit')
                except_fits += 1
                continue
            z0r, sigma_par = result.x
            z0r_estimation, sigma_par_estimation = z0r, 2 * sigma_par
            P_ph_0 = vs.P_ph_0
        elif structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
            try:
                z0r, result = vs.fit(start_estimation_z0r=z0r_estimation)
            except mesa_vs.PphNotConvergeError:
                print('Except fit')
                except_fits += 1
                continue
            z0r_estimation = z0r
            P_ph_0 = vs.P_ph_0
        else:
            z0r, result = vs.fit(start_estimation_z0r=z0r_estimation)
            z0r_estimation = z0r

        tau = vs.tau()
        print('Mdot = {:1.3e} g/s, Teff = {:g} K, tau = {:g}, z0r = {:g}'.format(Mdot, Teff, tau, z0r))

        if tau < 1 and tau_key:
            tau_index = i - except_fits
            tau_key = False
            if tau_break:
                print('Note: tau<1, tau_break=True. Cycle ends, when tau<1.')
                break

        varkappa_C, rho_C, T_C, P_C, Sigma0 = vs.parameters_C()
        PradPgas_C = (4 * sigmaSB) / (3 * c) * T_C ** 4 / P_C

        output_string = [Sigma0, Teff, Mdot, F, z0r, rho_C, T_C, P_C, tau, PradPgas_C, varkappa_C]

        print('Sigma0 = {:g} g/cm^2'.format(Sigma0))

        rho, eos = vs.law_of_rho(P_C, T_C, full_output=True)
        try:
            _ = eos.c_p
            free_e = np.exp(eos.lnfree_e)
            conv_param_z, conv_param_sigma = Convective_parameter(vs)
            output_string.extend([free_e, conv_param_z, conv_param_sigma])
        except AttributeError:
            pass

        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr',
                         'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
            QirrQvis = vs.Q_irr / vs.Q0
            T_irr_, C_irr_ = vs.T_irr, vs.C_irr
            output_string.extend([QirrQvis, T_irr_, C_irr_])
            print('T_irr, C_irr = {:g}, {:g}'.format(T_irr_, C_irr_))

        if add_Pi_values:
            output_string.extend(vs.Pi_finder())

        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
            try:
                cost_func = result.cost * 2
            except AttributeError:
                cost_func = sum(result.fun ** 2)
            output_string.extend([cost_func, vs.converged, vs.Sigma_ph])

        Sigma_plot.append(Sigma0)

        if i == 0:
            sigma_temp = Sigma0
        else:
            delta_Sigma_plus = Sigma0 - sigma_temp
            sigma_temp = Sigma0

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
                Output_Plot.append(T_C)
            else:
                raise Exception('Incorrect output, try Teff, Mdot, Mdot_Mdot_edd, F, z0r or T_C')

        if PradPgas_C < 1.0 and key:
            PradPgas10_index = i - 1 - except_fits
            key = False
        if delta_Sigma_plus > 0.0 and Sigma_plus_key:
            Sigma_plus_index = i - 1 - except_fits
            Sigma_plus_key = False
        if delta_Sigma_plus < 0.0 and not Sigma_plus_key and Sigma_minus_key:
            Sigma_minus_index = i - 1 - except_fits
            Sigma_minus_key = False

        output_string = np.array(output_string)
        with open(path_dots, 'a') as file:
            np.savetxt(file, output_string, newline=' ')
            file.write('\n')
    with open(path_dots, 'a') as file:
        file.write('# Sigma_plus_index = {:d}  Sigma_minus_index = {:d}'.format(Sigma_plus_index, Sigma_minus_index))
        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr',
                         'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
            file.write('\n# Except_fits = {}'.format(except_fits))

    if not make_pic:
        return 0

    xlabel = r'$\Sigma_0, \, \rm g/cm^2$'

    if xscale == 'parlog':
        Sigma_plot = np.log10(Sigma_plot)
        xlabel = r'$\log \,$' + xlabel
    if yscale == 'parlog':
        Output_Plot = np.log10(Output_Plot)

    pl, = plt.plot(Sigma_plot[:tau_index + 1], Output_Plot[:tau_index + 1],
                   label=r'$P_{{\rm gas}} > P_{{\rm rad}}, \alpha = {:g}$'.format(alpha))
    if tau_index != n:
        plt.plot(Sigma_plot[tau_index:], Output_Plot[tau_index:], color=pl.get_c(), alpha=0.5, label=r'$\tau<1$')
    if PradPgas10_index != 0:
        plt.plot(Sigma_plot[:PradPgas10_index + 1], Output_Plot[:PradPgas10_index + 1],
                 label=r'$P_{\rm gas} < P_{\rm rad}$')

    if xscale != 'parlog':
        plt.xscale(xscale)
    if yscale != 'parlog':
        plt.yscale(yscale)

    if output == 'Teff':
        ylabel = r'$T_{\rm eff}, \, \rm K$'
    elif output == 'Mdot':
        ylabel = r'$\dot{M}, \, \rm g/s$'
    elif output == 'Mdot_Mdot_edd':
        ylabel = r'$\dot{M}/\dot{M}_{\rm edd} $'
    elif output == 'F':
        ylabel = r'$F, \, \rm g~cm^2 / s^2$'
    elif output == 'z0r':
        ylabel = r'$z_0/r$'
    elif output == 'T_C':
        ylabel = r'$T_{\rm c}, \rm K$'
    if yscale == 'parlog':
        ylabel = r'$\log \,$' + ylabel
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True, which='both', ls='-')
    plt.legend()
    if set_title:
        plt.title(title)
    plt.tight_layout()
    if path_plot is not None:
        plt.savefig(path_plot)
        plt.close()


def Radial_Plot(M, alpha, r_start, r_end, Par, input='Mdot', structure='BellLin', mu=0.6, abundance='solar',
                nu_irr=None, L_X_irr=None, spectrum_irr=None, spectrum_irr_par=None, args_spectrum_irr=(),
                kwargs_spectrum_irr={}, cos_theta_irr=None, cos_theta_irr_exp=1 / 12, C_irr=None, T_irr=None,
                z0r_start_estimation=None, Sigma0_start_estimation=None,
                n=100, tau_break=True, path_dots=None, add_Pi_values=True):
    """
    Calculates radial structure of disc. Return table, which contains input parameters of the system,
    surface density Sigma0, viscous torque F, accretion rate Mdot, effective temperature Teff,
    geometrical half-thickness of the disc z0r and parameters in the symmetry plane of disc
    as functions of radius.

    Parameters
    ----------
    M : double
        Mass of central object in grams.
    alpha : double
        Alpha parameter for alpha-prescription of viscosity.
    r_start : double
        The starting value of radius. Radius (in cylindrical coordinate system, in cm)
        is the distance from central star.
    r_end : double
        The end value of radius. Radius (in cylindrical coordinate system, in cm)
        is the distance from central star.
    Par : double
        Par can be accretion rate in g/s, in eddington limits or in Msun/yr. Choice depends on 'input' parameter.
    input : str
        Define the choice of 'Par' parameter.
        Can be 'Mdot' (accretion rate), 'Mdot_Mdot_edd' (Mdot in eddington limits)
        or 'Mdot_Msun_yr' (Mdot in Msun/yr).
    structure : str
        Type of vertical structure. Possible options are:
        'Kramers', 'BellLin' -- ideal gas EoS, analytical opacities and radiative temperature gradient;
        'Mesa', 'MesaAd', 'MesaRadAd', 'MesaRadConv' -- MESA EoS and opacities and
                                                        radiative, adiabatic, rad+ad
                                                        and rad+conv temperature gradient;
        'MesaIdealGas' -- ideal gas EoS, MESA opacities and radiative temperature gradient;
        'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero' -- simple external irradiation scheme
                                                                   via T_irr or C_irr
        'MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr' -- advanced external irradiation scheme
                                                       from (Mescheryakov et al. 2011).
    mu : double
        Mean molecular weight. Use in case of ideal gas EoS.
    abundance : dict or str
        Chemical composition of disc. Use in case of MESA EoS.
        Format: {'isotope_name': abundance}. For example: {'h1': 0.7, 'he4': 0.3}.
        Use 'solar' str in case of solar composition.
    nu_irr : array-like
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'], nu_irr is the irradiation spectral flux argument.
        Can be (X-ray) frequency (in Hz) or energy (in keV) array for spectral external irradiation flux.
        Choose depends on the 'spectrum_irr_par'.
    spectrum_irr : array-like or callable
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        spectrum_irr is the spectrum of external irradiation flux, i.e.
        the spectral (X-ray) external irradiation flux F_nu_irr = F_irr * spectrum_irr.
        If spectrum_irr is array-like, then its size must be equal to nu_irr.size.
        If spectrum_irr is callable, then
        ``F_nu_irr = F_irr * spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``.
        The normalization of the spectrum_irr in that case is performed automatically.
    spectrum_irr_par : str
        Defines the irradiation spectral flux argument. Can be 'nu' (frequency in Hz) and 'E_in_keV' (energy in keV).
    args_spectrum_irr, kwargs_spectrum_irr : tuple and dict
        Extra arguments and keyword arguments of spectrum_irr, if it's callable.
        The calling signature is ``spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``
    L_X_irr : double or None
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        L_X_irr is the (X-ray) bolometric luminosity of external irradiation source.
        If None, then ``L_X_irr = 0.1 * Mdot * c ** 2``.
        The irradiation flux ``F_irr = L_X_irr / (4 * pi * r ** 2)``.
    cos_theta_irr : double or None
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'], cos_theta_irr is the cosine of angle
        of incidence for external irradiation flux. If None, ``cos_theta_irr = cos_theta_irr_exp * (z0/r)``.
    cos_theta_irr_exp : double
        If cos_theta_irr is None, ``cos_theta_irr = cos_theta_irr_exp * (z0/r)``.
    C_irr : double
        If structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero'],
        C_irr is the irradiation parameter.
    T_irr : double
        If structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero'],
        T_irr is the irradiation temperature.
    z0r_start_estimation : double
        Start estimation of z0r free parameter to fit the first point of radial structure.
        Further, z0r estimation of the next point is the z0r value of the previous point.
        Default is None, the start estimation is calculated automatically.
    Sigma0_start_estimation : double
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        it's the start estimation of Sigma0 free parameter to fit the first point of radial structure.
        Further, Sigma0 estimation of the next point is the 2*Sigma0 value of the previous point.
        Default is None, the start estimation is calculated automatically.
    n : int
        Number of dots to calculate.
    tau_break : bool
        Whether to end calculation, when tau<1.
    path_dots : str
        Where to save data table.
    add_Pi_values : bool
        Whether to write Pi-parameters (see Ketsaris & Shakura, 1998) to the output file.

    """
    if path_dots is None:
        print("ATTENTION: the data wil not be saved, since 'path_dots' is None")

    tau_key = True
    z0r_estimation = z0r_start_estimation
    sigma_par_estimation = Sigma0_start_estimation
    P_ph_0 = None
    except_fits = 0

    if input == 'Mdot':
        Mdot = Par
    elif input == 'Mdot_Mdot_edd':
        Mdot = Par * 1.39e18 * M / M_sun
    else:
        raise Exception("Incorrect input, try 'Mdot' or 'Mdot_Mdot_edd'.")

    if path_dots is not None:
        header = 'r \tr/rg \tSigma0 \tTeff \tF \tz0r \trho_c \tT_c \tP_c \ttau \tPradPgas_c \tvarkappa_c'
        header_end = '\nM = {:e} Msun, alpha = {}, Mdot = {} g/s, structure = {}'.format(
            M / M_sun, alpha, Mdot, structure)
        if structure in ['Kramers', 'BellLin', 'MesaIdealGas']:
            header_end += ', mu = {}'.format(mu)
        else:
            header_end += ', abundance = {}'.format(abundance)
            header += ' \tfree_e_c \tconv_param_z \tconv_param_sigma'
        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr',
                         'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
            header += ' \tQirrQvis \tT_irr \tC_irr'
        if add_Pi_values:
            header += ' \tPi1 \tPi2 \tPi3 \tPi4'
        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
            header += ' \tcost \tconverged \tSigma_ph'
        header = header + '\nAll values are in CGS units.' + header_end
        np.savetxt(path_dots, [], header=header)

    for i, r in enumerate(np.geomspace(r_start, r_end, n)):
        print(i)
        vs, F, Teff, Mdot = StructureChoice(M, alpha, r, Par, input, structure, mu, abundance,
                                            nu_irr=nu_irr, L_X_irr=L_X_irr,
                                            spectrum_irr=spectrum_irr, spectrum_irr_par=spectrum_irr_par,
                                            args_spectrum_irr=args_spectrum_irr,
                                            kwargs_spectrum_irr=kwargs_spectrum_irr,
                                            cos_theta_irr=cos_theta_irr, cos_theta_irr_exp=cos_theta_irr_exp,
                                            C_irr=C_irr, T_irr=T_irr, P_ph_0=P_ph_0)
        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
            try:
                result = vs.fit(start_estimation_z0r=z0r_estimation, start_estimation_Sigma0=sigma_par_estimation)
            except (mesa_vs.NotConvergeError, mesa_vs.PphNotConvergeError):
                print('Except fit')
                except_fits += 1
                continue
            z0r, sigma_par = result.x
            z0r_estimation, sigma_par_estimation = z0r, 2 * sigma_par
            P_ph_0 = vs.P_ph_0
        elif structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
            try:
                z0r, result = vs.fit(start_estimation_z0r=z0r_estimation)
            except mesa_vs.PphNotConvergeError:
                print('Except fit')
                except_fits += 1
                continue
            z0r_estimation = z0r
            P_ph_0 = vs.P_ph_0
        else:
            z0r, result = vs.fit(start_estimation_z0r=z0r_estimation)
            z0r_estimation = z0r

        tau = vs.tau()
        rg = 2 * G * M / c ** 2
        print('r = {:1.3e} cm = {:g} rg, Teff = {:g} K, tau = {:g}, z0r = {:g}'.format(r, r / rg, Teff, tau, z0r))

        if tau < 1 and tau_key:
            tau_key = False
            if tau_break:
                print('Note: tau<1, tau_break=True. Cycle ends, when tau<1.')
                break

        varkappa_C, rho_C, T_C, P_C, Sigma0 = vs.parameters_C()
        PradPgas_C = (4 * sigmaSB) / (3 * c) * T_C ** 4 / P_C

        output_string = [r, r / rg, Sigma0, Teff, F, z0r, rho_C, T_C, P_C, tau, PradPgas_C, varkappa_C]
        print('Sigma0 = {:g} g/cm^2'.format(Sigma0))

        rho, eos = vs.law_of_rho(P_C, T_C, full_output=True)
        try:
            _ = eos.c_p
            free_e = np.exp(eos.lnfree_e)
            conv_param_z, conv_param_sigma = Convective_parameter(vs)
            output_string.extend([free_e, conv_param_z, conv_param_sigma])
        except AttributeError:
            pass

        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr',
                         'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
            QirrQvis = vs.Q_irr / vs.Q0
            T_irr_, C_irr_ = vs.T_irr, vs.C_irr
            output_string.extend([QirrQvis, T_irr_, C_irr_])
            print('T_irr, C_irr = {:g}, {:g}'.format(T_irr_, C_irr_))

        if add_Pi_values:
            output_string.extend(vs.Pi_finder())

        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
            try:
                cost_func = result.cost * 2
            except AttributeError:
                cost_func = sum(result.fun ** 2)
            output_string.extend([cost_func, vs.converged, vs.Sigma_ph])

        output_string = np.array(output_string)
        with open(path_dots, 'a') as file:
            np.savetxt(file, output_string, newline=' ')
            file.write('\n')
    if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr',
                     'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
        with open(path_dots, 'a') as file:
            file.write('# Except_fits = {}'.format(except_fits))
    return 0


def main():
    M = 1.5 * M_sun
    alpha = 0.2
    r = 1e10
    Teff = 1e4
    os.makedirs('fig/', exist_ok=True)

    print('Calculation of vertical structure. Return structure table and plot.')
    print('M = {:g} M_sun \nr = {:g} cm \nalpha = {:g} \nTeff = {:g} K'.format(M / M_sun, r, alpha, Teff))

    Structure_Plot(M, alpha, r, Teff, input='Teff', mu=0.62, structure='BellLin', n=100, add_Pi_values=True,
                   path_dots='fig/vs.dat', make_pic=True, path_plot='fig/vs.pdf',
                   set_title=True,
                   title=r'$M = {:g} \, M_{{\odot}}, r = {:g} \, {{\rm cm}}, \alpha = {:g}, T_{{\rm eff}} = {:g} \, '
                         r'{{\rm K}}$'.format(M / M_sun, r, alpha, Teff))
    print('Structure is calculated successfully. Plot is saved to fig/vs.pdf, table is saved to fig/vs.dat. \n')

    print('Calculation of S-curve for Teff from 4e3 K to 1e4 K. Return S-curve table and Sigma0-Mdot plot.\n')

    S_curve(4e3, 1e4, M, alpha, r, input='Teff', structure='BellLin', mu=0.62, n=200, tau_break=False,
            path_dots='fig/S-curve.dat', add_Pi_values=True, make_pic=True, output='Mdot',
            xscale='parlog', yscale='parlog', path_plot='fig/S-curve.pdf', set_title=True,
            title=r'$M = {:g} \, M_{{\odot}}, r = {:g} \, {{\rm cm}}, \alpha = {:g}$'.format(M / M_sun, r, alpha))
    print('S-curve is calculated successfully. Plot is saved to fig/S-curve.pdf, table is saved to fig/S-curve.dat.')

    print('Calculation of radial structure of disc for radius from 3.1*rg to 1e3*rg and Mdot = Mdot_edd. '
          'Return radial structure table.\n')

    rg = 2 * G * M / c ** 2
    Radial_Plot(M, alpha, 3.1 * rg, 1e3 * rg, 1, input='Mdot_Mdot_edd', structure='BellLin', mu=0.62, n=200,
                tau_break=True, path_dots='fig/radial_struct.dat', add_Pi_values=True)
    print('Radial structure is calculated successfully. Table is saved to fig/radial_struct.dat.')

    return


if __name__ == '__main__':
    main()
