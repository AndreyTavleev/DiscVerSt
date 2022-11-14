# DiscVerSt — accretion disc vertical structure calculation

This code calculates vertical structure of accretion discs around neutron stars and black holes.

## Contents

   * [Installation](#Installation)
      * [Only analytical opacities and EoS](#Only-analytical-opacities-and-EoS)
      * [Tabular opacities and EoS](#Tabular-opacities-and-EoS)
   * [Calculate structure](#Calculate-structure)
   * [Irradiated discs](#Irradiated-discs)
   * [Structure Choice](#Structure-Choice)
   * [Vertical and radial profile calculation, S-curves](#Vertical-and-radial-profile-calculation-S-curves)

## Installation

### Only analytical opacities and EoS

If you want to use only analytical formulas for opacity and EoS to calculate structures, choose this installation way.

1. Create and activate virtual environment

``` shell
$ python3 -m venv ~/.venv/vs
$ source ~/.venv/vs/bin/activate
```

2. Update pip and setuptools

``` shell
$ pip3 install -U pip setuptools
```

3. Install 'disc_verst' package

``` shell
$ pip3 install .
```

4. Run python and try to calculate simple structure

``` shell
$ python3 -m disc_verst.vs
```

	Finding Pi parameters of structure and making a structure plot. 
	Structure with Kramers opacity and ideal gas EoS.
	M = 1.98841e+34 grams
	r = 1.1813e+09 cm = 400 rg
	alpha = 0.01
	Mdot = 1e+17 g/s
	The vertical structure has been calculated successfully.
	Pi parameters = [7.10271534 0.4859551  1.13097882 0.3985615 ]
	z0/r =  0.028869666211114635
	Plot of structure is successfully saved to fig/vs.pdf.

### Tabular opacities and EoS

['mesa2py'](https://github.com/hombit/mesa2py) is used to bind tabular opacities and EoS 
from [MESA code](http://docs.mesastar.org) with Python3.
If you want to use tabular values of opacity and EoS to calculate the structure, you should use Docker.

You can use the latest pre-build Docker image:

``` shell
$ docker pull ghcr.io/andreytavleev/discverst:latest
$ docker tag ghcr.io/andreytavleev/discverst discverst
```

Or build a Docker image by yourself

``` shell
$ git clone https://github.com/AndreyTavleev/DiscVerSt.git
$ cd DiscVerSt
$ docker build -t discverst .
```

Then run 'discverst' image as a container and try mesa_vs.main()

``` shell
$ docker run -v$(pwd)/fig:/app/fig --rm -ti discverst python3 -m disc_verst.mesa_vs
```

	Calculating structure and making a structure plot.
	Structure with tabular MESA opacity and EoS.
    Chemical composition is solar.
	M = 1.98841e+34 grams
	r = 1.1813e+09 cm = 400 rg
	alpha = 0.01
	Mdot = 1e+17 g/s
	The vertical structure has been calculated successfully.
	z0/r =  0.029812574021917705
	Plot of structure is successfully saved to fig/vs_mesa.pdf.

As the result, the plot `vs_mesa.pdf` is saved to the `/app/fig/` (`/app/` is a Docker WORKDIR) 
in the container and to the `$(pwd)/fig/` in the host machine. 

## Calculate structure

Module `vs` contains several classes that represent the vertical 
structure of accretion discs in X-ray binaries for different assumptions 
of opacity law. For given parameters the vertical structure of 
disc is calculated and can be used for research of disc stability.

Module `mesa_vs` contains some additional classes, that represent 
the vertical structure for tabular opacities and convective energy transport.

Both `vs` and `mesa_vs` modules have help
``` python3
help(disc_verst.vs)
help(disc_verst.mesa_vs)
```

### Usage with analytical opacities and EoS::

``` python3
from disc_verst import vs

# Input parameters:
M = 2e33  # Mass of central object in grams
alpha = 0.01  # alpha parameter
r = 2e9  # radius in cm
F = 2e34  # viscous torque
mu = 0.6  # molecular weight

# Structure with (Bell & Lin, 1994) power-law opacities and ideal gas EoS
# with molecular weight mu and radiative temperature gradient.

vertstr = vs.IdealBellLin1994VerticalStructure(Mx=M, alpha=alpha, r=r, F=F, mu=mu)  # create the structure object
# the estimation of z0r free parameter is calculated automatically (default)
z0r, result = vertstr.fit(z0r_estimation=None)  # calculate structure
varkappa_C, rho_C, T_C, P_C, Sigma0 = vertstr.parameters_C()
print(z0r)  # semi-thickness z0/r of disc
print(varkappa_C, rho_C, T_C, P_C)  # Opacity, bulk density, temperature and gas pressure in the symmetry plane of disc
print(Sigma0)  # Surface density of disc
print(vertstr.tau())  # optical thickness of disc
```

### Usage with tabular opacities and EoS:

``` shell
$ docker run --rm -ti discverst python3
```

``` python3
from disc_verst import mesa_vs

# Input parameters:
M = 2e33  # Mass of central object in grams
alpha = 0.01  # alpha parameter
r = 2e9  # radius in cm
F = 2e34  # viscous torque

# Structure with tabular MESA opacities and EoS 
# and both radiative and convective energy transport,
# default chemical composition is solar.

vertstr = mesa_vs.MesaVerticalStructureRadConv(Mx=M, alpha=alpha, r=r, F=F)  # create the structure object
z0r, result = vertstr.fit()  # calculate structure
varkappa_C, rho_C, T_C, P_C, Sigma0 = vertstr.parameters_C()
print(z0r)  # semi-thickness z0/r of disc
print(varkappa_C, rho_C, T_C, P_C)  # Opacity, bulk density, temperature and gas pressure in the symmetry plane of disc
print(Sigma0)  # Surface density of disc
print(vertstr.tau())  # optical thickness of disc
```

You can run your own files, that use this code, inside Docker

``` shell
$ docker run -v/path_to/your_file/file.py:/app/file.py --rm -ti discverst python3 file.py
```

## Irradiated discs

Module `mesa_vs` also contains classes that represent 
the vertical structure of self-irradiated discs.

Irradiation can be taken into account in two ways:

1. Via either `T_irr` or `C_irr` parameters, that is, the irradiation temperature and irradiation parameter. 
   It is a simple approach for irradiation, when the external flux doesn't penetrate into the disc and only heats the 
   disc surface.

2. In the second approximation the external flux is penetrated into the disc and affect the energy flux
   and disc temperature. In this case there are more additional parameters are required, that describe
   the incident spectral flux. Such parameters are: frequency range `nu_irr`, units of frequency range 
   `spectrum_irr_par` (see below), spectrum `spectrum_irr`, luminosity of irradiation source `L_X_irr` 
   and the cosine of incident angle `cos_theta_irr`. The spectral incident flux then will be 
   `F_nu_irr = L_X_irr / (4 * pi * r ** 2) * spectrum_irr`.

   1. Frequency range `nu_irr` is array-like and can be either in Hz or in energy units (keV), this is determined by 
      `spectrum_irr_par` in `['nu', 'E_in_keV']`.
   2. Spectrum `spectrum_irr` can be either an array-like or a Python function. 
      1. If `spectrum_irr` is array-like, it must be in 1/Hz or in 1/keV depending on 'spectrum_irr_par',
         must be normalised to unity, and its size must be equal to `nu_irr.size`.
      2. If `spectrum_irr` is a Python function, the spectrum is calculated 
         for the frequency range `nu_irr` and automatically normalised to unity over `nu_irr`. 
         Note, that units of `nu_irr` and units of `spectrum_irr` arguments must be consistent. 
         There are two optional parameters `args_spectrum_irr` and `kwargs_spectrum_irr` 
         for arguments (keyword arguments) of spectrum function.
   3. Cosine of incident angle `cos_theta_irr` can be either exact value or `None`. In the latter case
      cosine is calculated self-consistently as `cos_theta_irr_exp * z0 / r`, where `cos_theta_irr_exp` is
      additional parameter, namely the `dln(z0)/dln(r) - 1` derivative.

### Usage of simple irradiation scheme:

``` python3
from disc_verst import mesa_vs

# Input parameters:
M = 2e33  # Mass of central object in grams
alpha = 0.01  # alpha parameter
r = 2e9  # radius in cm
F = 2e34  # viscous torque
Tirr = 1.2e4  # irradiation temperature

# Structure with tabular MESA opacities and EoS,
# both radiative and convective energy transport,
# default chemical composition is solar, 
# external irradiation is taken into account
# through the simple scheme.

vertstr = mesa_vs.MesaVerticalStructureRadConvExternalIrradiationZeroAssumption(
                     Mx=M, alpha=alpha, r=r, F=F, T_irr=Tirr)  # create the structure object
z0r, result = vertstr.fit()  # calculate structure
varkappa_C, rho_C, T_C, P_C, Sigma0 = vertstr.parameters_C()
print(z0r)  # semi-thickness z0/r of disc
print(varkappa_C, rho_C, T_C, P_C)  # Opacity, bulk density, temperature and gas pressure in the symmetry plane of disc
print(Sigma0)  # Surface density of disc
print(vertstr.tau())  # optical thickness of disc
print(vertstr.C_irr)  # corresponding irradiation parameter
```

### Usage of advanced irradiation scheme:

Define the incident X-ray spectrum as a function of energy in keV:

``` python3
def power_law_exp_spectrum(E, n, scale):
   return (E / scale) ** n * np.exp(-E / scale)
```

Then calculate structure with this spectrum:

``` python3
from disc_verst import mesa_vs
import numpy as np

# Input parameters:
M = 2e33  # Mass of central object in grams
alpha = 0.01  # alpha parameter
r = 4e9  # radius in cm
F = 2e34  # viscous torque
nu_irr = np.geomspace(1, 10, 30)  # energy range from 1 to 10 keV
spectrum_par = 'E_in_keV'  # units of nu_irr if energy in keV
kwargs={'n': -0.4, 'scale': 8}  # spectrum function parameters
cos_theta_irr = None  # incident angle is calculated self-consistently
cos_theta_irr_exp = 1 / 12  # as cos_theta_irr_exp * z0 / r
L_X_irr = 1.0 * 1.25e38 * M / 2e33  #  luminosity of X-ray source = 1.0 * L_eddington

# Structure with tabular MESA opacities and EoS,
# both radiative and convective energy transport,
# default chemical composition is solar, 
# external irradiation is taken into account
# through the advanced scheme.

vertstr = mesa_vs.MesaVerticalStructureRadConvExternalIrradiation(
                     Mx=M, alpha=alpha, r=r, F=F, nu_irr=nu_irr, 
                     spectrum_irr=power_law_exp_spectrum, L_X_irr=L_X_irr,
                     spectrum_irr_par=spectrum_par, 
                     kwargs_spectrum_irr=kwargs,
                     cos_theta_irr=cos_theta_irr, 
                     cos_theta_irr_exp=cos_theta_irr_exp)  # create the structure object
# let us set the free parameters estimation
result = vertstr.fit(z0r_estimation=0.068, Sigma0_estimation=1032)  # calculate structure
# if structure is fitted successfully, cost function must be less than 1e-16
print(result.cost)  # cost function
z0r, Sigma0 = result.x  # Sigma0 is additional free parameter to find
varkappa_C, rho_C, T_C, P_C, Sigma0 = vertstr.parameters_C()
print(z0r)  # semi-thickness z0/r of disc
print(varkappa_C, rho_C, T_C, P_C)  # Opacity, bulk density, temperature and gas pressure in the symmetry plane of disc
print(Sigma0)  # Surface density of disc
print(vertstr.tau())  # optical thickness of disc
print(vertstr.C_irr, vertstr.T_irr)  # irradiation parameter and temperature
```

## Structure Choice
Module `profiles` contains `StructureChoice()` function, serves as interface for creating 
the right structure object in a simpler way. One can use other input parameters instead viscous torque `F`
(such as effective temperature and accretion rate) using `input` parameter,
and choose the structure type using `structure` parameter.
``` python3
from disc_verst.profiles import StructureChoice

# Input parameters:
M = 2e33  # Mass of central object in grams
alpha = 0.01  # alpha parameter
r = 4e9  # radius in cm
input = 'Teff'  # the input parameter will be effective temperature
Par = 1e4  # instead of viscous torque F
structure = 'BellLin'  # type of structure
mu = 0.6  # mean molecular weight for this type

# Returns the instance of chosen structure type.
# Also returns viscous torque F, effective temperature Teff and accretion rate Mdot.
vertstr, F, Teff, Mdot = StructureChoice(M=M, alpha=alpha, r=r, Par=Par, 
                                         input=input, structure=structure, mu=mu)
z0r, result = vertstr.fit()  # calculate the structure
```
The `StructureChoice` function has the detailed documentation
``` python3
from disc_verst import profiles

help(profiles.StructureChoice)
```

## Vertical and radial profile calculation, S-curves

Module `profiles` contains functions, that calculate vertical and radial disc profiles and S-curves, and return tables 
with disc parameters. With profiles.main() the vertical structure, S-curve and radial structure 
can be calculated for default parameters, stored as tables and plots:
``` shell
$ python3 -m disc_verst.profiles
```

`profiles` contains three functions: `Vertical_Profile`, `S_curve` and `Radial_Profile`. 

`Vertical_Profile` returns table with parameters of disc as functions of vertical coordinate at specific radius.

`S_curve` calculates S-curve and return table with disc parameters on the curve.

`Radial_Profile` returns table with parameters of disc as functions of radius for a given radius range.

### Usage:
``` python3
from disc_verst import profiles

# Input parameters:
M = 1.5 * 2e33  # 1.5 * M_sun
alpha = 0.2
r = 1e10
Teff = 1e4

profiles.Vertical_Profile(M, alpha, r, Teff, input='Teff', structure='BellLin', mu=0.62,
                          n=100, add_Pi_values=True, path_dots='vs.dat')

profiles.S_curve(4e3, 1e4, M, alpha, r, input='Teff', structure='BellLin', mu=0.62, 
                 n=200, tau_break=False, path_dots='S-curve.dat', add_Pi_values=True)

rg = 3e5 * (M / 2e33)  # Schwarzschild radius
profiles.Radial_Profile(M, alpha, 3.1 * rg, 1e3 * rg, 1, input='Mdot_Mdot_edd', structure='BellLin', mu=0.62, 
                        n=200, tau_break=True, path_dots='radial_struct.dat', add_Pi_values=True)
```
Both `profiles` module and functions in it have help
``` python3
from disc_verst import profiles

help(profiles)
help(profiles.Vertical_Profile)
help(profiles.S_curve)
help(profiles.Radial_Profile)
```
