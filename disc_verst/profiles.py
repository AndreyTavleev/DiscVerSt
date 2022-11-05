#!/usr/bin/env python3
"""
Module contains functions that calculate vertical and radial structure and S-curve.
Functions return tables with calculated data of structure or S-curve.

StructureChoice -- Initialize the chosen vertical structure class.
    It serves as interface for creating the right structure class in a simpler way.
Vertical_Profile -- calculates vertical structure and makes table with disc parameters as functions of vertical
    coordinate. Table also contains input parameters of structure, parameters in the symmetry plane and
    parameter normalizations.
S_curve -- Calculates S-curve and makes table with disc parameters on the S-curve.
    Table contains input parameters of system, surface density Sigma0, viscous torque F,
    accretion rate Mdot, effective temperature Teff, geometrical half-thickness of the disc z0r,
    parameters in the symmetry plane of disc on the S-curve.
Radial_Profile -- Calculates radial structure of disc. Return table, which contains input parameters of the system,
    surface density Sigma0, viscous torque F, accretion rate Mdot, effective temperature Teff,
    geometrical half-thickness of the disc z0r and parameters in the symmetry plane of disc
    as functions of radius.

"""
import os

import numpy as np
from astropy import constants as const
from scipy.integrate import simps
from scipy.interpolate import InterpolatedUnivariateSpline

import disc_verst.vs as vert

try:
    import disc_verst.mesa_vs as mesa_vs
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
    M : double
        Mass of central object in grams.
    alpha : double
        Alpha parameter for alpha-prescription of viscosity.
    r : double
        Distance from central star (radius in cylindrical coordinate system) in cm.
    Par : double
        Can be viscous torque in g*cm^2/s^2, effective temperature in K,
        accretion rate in g/s, in eddington limits or in Msun/yr.
        Choice depends on 'input' parameter.
    input : str
        Define the choice of 'Par' parameter. Can be 'F' (viscous torque),
        'Teff' (effective temperature, or viscous temperature in case of irradiation),
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
        If spectrum_irr is array-like it must be in 1/Hz or in 1/keV depending on 'spectrum_irr_par',
        must be normalized to unity, and its size must be equal to nu_irr.size.
        If spectrum_irr is callable, then
        ``F_nu_irr = F_irr * spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``.
        The normalization of the spectrum_irr in that case is performed automatically.
    spectrum_irr_par : str
        Defines the irradiation spectral flux argument.
        Can be 'nu' (frequency in Hz) and 'E_in_keV' (energy in keV).
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
    P_ph_0 : double
        If structure contains Irradiation (either irradiation scheme),
        it's the start estimation for pressure at the photosphere (pressure boundary condition).
        Default is None, the estimation is calculated automatically.

    Returns
    -------
    vs : vertical structure
        Chosen NON-FITTED vertical structure.
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
                        'Actual radius r = {:g} rg.'.format(r / rg))
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
        raise Exception("Incorrect input, try 'Teff', 'F', 'Mdot', 'Mdot_Mdot_edd' or 'Mdot_Msun_yr'.")

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
        FITTED vertical structure, for which convective parameter is calculated.

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
        raise Exception('Incorrect vertical structure for convective parameter calculation. '
                        'Use vertical structure with MESA EoS.') from None
    conv_param_sigma = simps(2 * rho * (grad_plot(np.log(P)) > eos.grad_ad), t * vs.z0) / (
            S[-1] * vs.sigma_norm)
    conv_param_z = simps(grad_plot(np.log(P)) > eos.grad_ad, t * vs.z0) / vs.z0
    return conv_param_z, conv_param_sigma


def Vertical_Profile(M, alpha, r, Par, input='Teff', mu=0.6, structure='BellLin', abundance='solar', nu_irr=None,
                     L_X_irr=None, spectrum_irr=None, spectrum_irr_par=None,
                     args_spectrum_irr=(), kwargs_spectrum_irr={},
                     cos_theta_irr=None, cos_theta_irr_exp=1 / 12, C_irr=None, T_irr=None,
                     z0r_estimation=None, Sigma0_estimation=None, P_ph_0=None,
                     n=100, add_Pi_values=True, path_dots=None):
    """
    Calculates vertical structure and makes table with disc parameters as functions of vertical coordinate.
    Table also contains input parameters of structure, parameters in the symmetry plane and parameter normalizations.

    Parameters
    ----------
    M : double
        Mass of central object in grams.
    alpha : double
        Alpha parameter for alpha-prescription of viscosity.
    r : double
        Distance from central star (radius in cylindrical coordinate system) in cm.
    Par : double
        Can be viscous torque in g*cm^2/s^2, effective temperature in K,
        accretion rate in g/s, in eddington limits or in Msun/yr.
        Choice depends on 'input' parameter.
    input : str
        Define the choice of 'Par' parameter. Can be 'F' (viscous torque),
        'Teff' (effective temperature, or viscous temperature in case of irradiation),
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
        If spectrum_irr is array-like it must be in 1/Hz or in 1/keV depending on 'spectrum_irr_par',
        must be normalized to unity, and its size must be equal to nu_irr.size.
        If spectrum_irr is callable, then
        ``F_nu_irr = F_irr * spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``.
        The normalization of the spectrum_irr in that case is performed automatically.
    spectrum_irr_par : str
        Defines the irradiation spectral flux argument.
        Can be 'nu' (frequency in Hz) and 'E_in_keV' (energy in keV).
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

    """
    if path_dots is None:
        raise Exception("ATTENTION: the data wil not be saved, since 'path_dots' is None.")
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
    tau_arr = np.array([simps(rho[:i] * varkappa[:i], t[:i] * z0r * r) for i in range(2, n + 1)]) + 2 / 3
    tau_arr = np.r_[2 / 3, tau_arr]
    dots_arr = np.c_[t, S, P, abs(Q), T, rho, varkappa, tau_arr, grad_plot(np.log(P))]
    header_input_irr = ''
    if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr',
                     'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
        Teff_string = 'Tvis'
    else:
        Teff_string = 'Teff'
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
                   '\nM = {:e} Msun, alpha = {}, r = {:e} cm, r = {} rg, {} = {} K, Mdot = {:e} g/s, ' \
                   'F = {:e} g*cm^2/s^2, structure = {}'.format(M / M_sun, alpha, r, r / rg, Teff_string,
                                                                Teff, Mdot, F, structure)
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
    return


def S_curve(Par_min, Par_max, M, alpha, r, input='Teff', structure='BellLin', mu=0.6, abundance='solar', nu_irr=None,
            L_X_irr=None, spectrum_irr=None, spectrum_irr_par=None, args_spectrum_irr=(), kwargs_spectrum_irr={},
            cos_theta_irr=None, cos_theta_irr_exp=1 / 12, C_irr=None, T_irr=None,
            z0r_start_estimation=None, Sigma0_start_estimation=None,
            n=100, tau_break=True, add_Pi_values=True, path_dots=None):
    """
    Calculates S-curve and makes table with disc parameters on the S-curve.
    Table contains input parameters of system, surface density Sigma0, viscous torque F,
    accretion rate Mdot, effective temperature Teff, geometrical half-thickness of the disc z0r,
    parameters in the symmetry plane of disc on the S-curve.

    Parameters
    ----------
    Par_min : double
        The starting value of Par. Par_min and Par_max can be viscous torque in g*cm^2/s^2, effective temperature in K,
        accretion rate in g/s, in eddington limits or in Msun/yr. Choice depends on 'input' parameter.
    Par_max : double
        The end value of Par. Par_min and Par_max can be viscous torque in g*cm^2/s^2, effective temperature in K,
        accretion rate in g/s, in eddington limits or in Msun/yr. Choice depends on 'input' parameter.
    M : double
        Mass of central object in grams.
    alpha : double
        Alpha parameter for alpha-prescription of viscosity.
    r : double
        Distance from central star (radius in cylindrical coordinate system) in cm.
    input : str
        Define the choice of 'Par_min' and 'Par_max' parameters.
        Can be 'F' (viscous torque), 'Teff' (effective temperature, or viscous temperature in case of irradiation),
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
        If spectrum_irr is array-like it must be in 1/Hz or in 1/keV depending on 'spectrum_irr_par',
        must be normalized to unity, and its size must be equal to nu_irr.size.
        If spectrum_irr is callable, then
        ``F_nu_irr = F_irr * spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``.
        The normalization of the spectrum_irr in that case is performed automatically.
    spectrum_irr_par : str
        Defines the irradiation spectral flux argument.
        Can be 'nu' (frequency in Hz) and 'E_in_keV' (energy in keV).
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
    add_Pi_values : bool
        Whether to write Pi-parameters (see Ketsaris & Shakura, 1998) to the output file.
    path_dots : str
        Where to save data table.

    """
    if path_dots is None:
        raise Exception("ATTENTION: the data wil not be saved, since 'path_dots' is None.")

    Sigma_minus_index = 0
    Sigma_plus_index = 0
    Sigma_minus_key = True
    Sigma_plus_key = True
    delta_Sigma_plus = -1
    z0r_estimation = z0r_start_estimation
    sigma_par_estimation = Sigma0_start_estimation

    sigma_temp = np.infty
    except_fits = 0
    P_ph_0 = None
    if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr',
                     'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
        Teff_string = 'Tvis'
    else:
        Teff_string = 'Teff'

    if path_dots is not None:
        rg = 2 * G * M / c ** 2
        header = 'Sigma0 \t{} \tMdot \tF \tz0r \trho_c \tT_c \tP_c \ttau \tPradPgas_c \tvarkappa_c'.format(Teff_string)
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
        print('Mdot = {:1.3e} g/s, {} = {:g} K, tau = {:g}, z0r = {:g}'.format(Mdot, Teff_string, Teff, tau, z0r))

        if tau < 1 and tau_break:
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
            print('T_irr, C_irr = {:g} K, {:g}'.format(T_irr_, C_irr_))

        if add_Pi_values:
            output_string.extend(vs.Pi_finder())

        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
            try:
                cost_func = result.cost * 2
            except AttributeError:
                cost_func = sum(result.fun ** 2)
            output_string.extend([cost_func, vs.converged, vs.Sigma_ph])

        if i == 0:
            sigma_temp = Sigma0
        else:
            delta_Sigma_plus = Sigma0 - sigma_temp
            sigma_temp = Sigma0

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
    return


def Radial_Profile(M, alpha, r_start, r_end, Par, input='Mdot', structure='BellLin', mu=0.6, abundance='solar',
                   nu_irr=None, L_X_irr=None, spectrum_irr=None, spectrum_irr_par=None, args_spectrum_irr=(),
                   kwargs_spectrum_irr={}, cos_theta_irr=None, cos_theta_irr_exp=1 / 12, C_irr=None, T_irr=None,
                   z0r_start_estimation=None, Sigma0_start_estimation=None,
                   n=100, tau_break=True, add_Pi_values=True, path_dots=None):
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
    Par : double or array-like
        Can be viscous torque in g*cm^2/s^2, effective temperature in K,
        accretion rate in g/s, in eddington limits or in Msun/yr.
        Choice depends on 'input' parameter.
        If Par is array-like, its size must be equal to n (number of dots to calculate).
    input : str
        Define the choice of 'Par' parameter.
        Can be 'F' (viscous torque), 'Teff' (effective temperature, or viscous temperature in case of irradiation),
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
        If spectrum_irr is array-like it must be in 1/Hz or in 1/keV depending on 'spectrum_irr_par',
        must be normalized to unity, and its size must be equal to nu_irr.size.
        If spectrum_irr is callable, then
        ``F_nu_irr = F_irr * spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``.
        The normalization of the spectrum_irr in that case is performed automatically.
    spectrum_irr_par : str
        Defines the irradiation spectral flux argument.
        Can be 'nu' (frequency in Hz) and 'E_in_keV' (energy in keV).
    args_spectrum_irr, kwargs_spectrum_irr : tuple and dict
        Extra arguments and keyword arguments of spectrum_irr, if it's callable.
        The calling signature is ``spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``
    L_X_irr : double or None
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        L_X_irr is the (X-ray) bolometric luminosity of external irradiation source.
        If None, then ``L_X_irr = 0.1 * Mdot * c ** 2``.
        The irradiation flux ``F_irr = L_X_irr / (4 * pi * r ** 2)``.
    cos_theta_irr : double or array-like or None
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'], cos_theta_irr is the cosine of angle
        of incidence for external irradiation flux. If None, ``cos_theta_irr = cos_theta_irr_exp * (z0/r)``.
        If cos_theta_irr is array-like, its size must be equal to n (number of dots to calculate).
    cos_theta_irr_exp : double or array-like
        If cos_theta_irr is None, ``cos_theta_irr = cos_theta_irr_exp * (z0/r)``.
        If cos_theta_irr_exp is array-like, its size must be equal to n (number of dots to calculate).
    C_irr : double or array-like
        If structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero'],
        C_irr is the irradiation parameter.
        If C_irr is array-like, its size must be equal to n (number of dots to calculate).
    T_irr : double or array-like
        If structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero'],
        T_irr is the irradiation temperature.
        If T_irr is array-like, its size must be equal to n (number of dots to calculate).
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
    add_Pi_values : bool
        Whether to write Pi-parameters (see Ketsaris & Shakura, 1998) to the output file.
    path_dots : str
        Where to save data table.

    """
    if path_dots is None:
        raise Exception("ATTENTION: the data wil not be saved, since 'path_dots' is None.")

    z0r_estimation = z0r_start_estimation
    sigma_par_estimation = Sigma0_start_estimation
    P_ph_0 = None
    except_fits = 0

    if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr',
                     'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
        Teff_string = 'Tvis'
    else:
        Teff_string = 'Teff'

    if path_dots is not None:
        header = 'r \tr/rg \tSigma0 \tMdot \t{} \tF \tz0r \trho_c \tT_c \tP_c ' \
                 '\ttau \tPradPgas_c \tvarkappa_c'.format(Teff_string)
        header_end = '\nM = {:e} Msun, alpha = {}, structure = {}'.format(M / M_sun, alpha, structure)
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

    try:
        input_broadcast = np.broadcast(Par, cos_theta_irr, cos_theta_irr_exp, C_irr, T_irr)
    except ValueError:
        raise ValueError("All array-like input parameters must have the same size n = {}.".format(n)) from None

    if input_broadcast.size not in [1, n]:
        raise ValueError("All array-like input parameters must have the same size n = {}.".format(n))

    input_arr = list(input_broadcast)

    for i, r in enumerate(np.geomspace(r_start, r_end, n)):
        print(i)
        if len(input_arr) == n:
            input_pars = input_arr[i]
        else:
            input_pars = input_arr[0]
        vs, F, Teff, Mdot = StructureChoice(M=M, alpha=alpha, r=r, Par=input_pars[0], input=input,
                                            structure=structure, mu=mu, abundance=abundance,
                                            nu_irr=nu_irr, L_X_irr=L_X_irr,
                                            spectrum_irr=spectrum_irr, spectrum_irr_par=spectrum_irr_par,
                                            args_spectrum_irr=args_spectrum_irr,
                                            kwargs_spectrum_irr=kwargs_spectrum_irr,
                                            cos_theta_irr=input_pars[1], cos_theta_irr_exp=input_pars[2],
                                            C_irr=input_pars[3], T_irr=input_pars[4], P_ph_0=P_ph_0)
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
        print('r = {:1.3e} cm = {:g} rg, Mdot = {:1.3e} g/s, {} = {:g} K, tau = {:g}, z0r = {:g}'.format(
            r, r / rg, Mdot, Teff_string, Teff, tau, z0r))

        if tau < 1 and tau_break:
            print('Note: tau<1, tau_break=True. Cycle ends, when tau<1.')
            break

        varkappa_C, rho_C, T_C, P_C, Sigma0 = vs.parameters_C()
        PradPgas_C = (4 * sigmaSB) / (3 * c) * T_C ** 4 / P_C

        output_string = [r, r / rg, Sigma0, Mdot, Teff, F, z0r, rho_C, T_C, P_C, tau, PradPgas_C, varkappa_C]
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
            print('T_irr, C_irr = {:g} K, {:g}'.format(T_irr_, C_irr_))

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
    from matplotlib import pyplot as plt
    from astropy.io import ascii
    M = 1.5 * M_sun
    alpha = 0.2
    r = 1e10
    Teff = 1e4
    os.makedirs('fig/', exist_ok=True)

    print('Calculation of vertical structure. Return structure table.')
    print('M = {:g} M_sun \nr = {:g} cm \nalpha = {:g} \nTeff = {:g} K\n'.format(M / M_sun, r, alpha, Teff))

    Vertical_Profile(M, alpha, r, Teff, input='Teff', mu=0.62, structure='BellLin',
                     n=100, add_Pi_values=True, path_dots='fig/vs.dat')
    print('Structure is calculated successfully. Table is saved to fig/vs.dat.')
    vs_data = ascii.read('fig/vs.dat')
    print('Making the structure plot.')
    plt.plot(1 - vs_data['t'], vs_data['S'], label=r'$\hat{\Sigma}$')
    plt.plot(1 - vs_data['t'], vs_data['P'], label=r'$\hat{P}$')
    plt.plot(1 - vs_data['t'], vs_data['Q'], label=r'$\hat{Q}$')
    plt.plot(1 - vs_data['t'], vs_data['T'], label=r'$\hat{T}$')
    plt.grid()
    plt.legend()
    plt.xlabel('$z / z_0$')
    plt.title(r'$M = {:g} \, M_{{\odot}}, r = {:g} \, {{\rm cm}}, \alpha = {:g}, '
              r'T_{{\rm eff}} = {:g} \, {{\rm K}}$'.format(M / M_sun, r, alpha, Teff))
    plt.tight_layout()
    plt.savefig('fig/vs.pdf')
    plt.close()
    print('Plot of structure is successfully saved to fig/vs.pdf.\n')

    print('Calculation of S-curve for Teff from 4e3 K to 1e4 K. Return S-curve table.')
    S_curve(4e3, 1e4, M, alpha, r, input='Teff', structure='BellLin', mu=0.62, n=200, tau_break=False,
            path_dots='fig/S-curve.dat', add_Pi_values=True)
    print('S-curve is calculated successfully. Table is saved to fig/S-curve.dat.')
    s_curve_data = ascii.read('fig/S-curve.dat')
    tau = s_curve_data['tau']
    print('Making the S-curve plot.')
    plt.plot(s_curve_data['Sigma0'][tau > 1], s_curve_data['Teff'][tau > 1])
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$T_{\rm eff}, \, \rm K$')
    plt.xlabel(r'$\Sigma_0, \, \rm g/cm^2$')
    plt.grid(True, which='both', ls='-')
    plt.title(r'$M = {:g} \, M_{{\odot}}, r = {:g} \, {{\rm cm}}, '
              r'\alpha = {:g}$'.format(M / M_sun, r, alpha))
    plt.tight_layout()
    plt.savefig('fig/S-curve.pdf')
    plt.close()
    print('Plot of S-curve is successfully saved to fig/S-curve.pdf.\n')

    print('Calculation of radial structure of disc for radius from 1e9 cm to 1e12 cm and Mdot = Mdot_edd. '
          'Return radial structure table.')
    Radial_Profile(M, alpha, 1e9, 1e12, 1, input='Mdot_Mdot_edd', structure='BellLin', mu=0.62, n=200,
                   tau_break=True, path_dots='fig/radial_struct.dat', add_Pi_values=True)
    print('Radial structure is calculated successfully. Table is saved to fig/radial_struct.dat.')
    rad_struct_data = ascii.read('fig/radial_struct.dat')
    print('Making the radial structure plot.')
    plt.plot(rad_struct_data['r'], rad_struct_data['z0r'])
    plt.xscale('log')
    plt.ylabel(r'$z_0 / r$')
    plt.xlabel(r'$r, \,\rm cm$')
    plt.grid(True, which='both', ls='-')
    plt.title(r'$M = {:g} \, M_{{\odot}}, \dot{{M}} = 1\,\dot{{M}}_{{\rm edd}}, '
              r'\alpha = {:g}$'.format(M / M_sun, alpha))
    plt.tight_layout()
    plt.savefig('fig/radial_struct.pdf')
    plt.close()
    print('Plot of radial structure is successfully saved to fig/radial_struct.pdf.')

    return


if __name__ == '__main__':
    main()
