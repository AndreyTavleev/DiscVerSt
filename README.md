# Vertical-structure-of-accretion-discs

This code can calculate vertical structure of accretion discs around neutron stars and black holes.

## Contents

   * [Installation](#Installation)
      * [Only analitical opacities and EOS](#Only-analitical-opacities-and-EOS)
      * [Tabular opacities and EOS](#Tabular-opacities-and-EOS)
   * [Calculate structure](#Calculate-structure)
   * [Irradiated discs](#Irradiated-discs)
   * [Make plots and tables with disc parameters](#Make-plots-and-tables-with-disc-parameters)

## Installation

### Only analytical opacities and EOS

If you want to use only analytical formulas for opacity and EOS to calculate structures, choose this installation way.

1. Create and activate virtual environment

``` shell
$ python3 -m venv ~/.venv/vs
$ source ~/.venv/vs/bin/activate
```

2. Update pip and install all requirements from
   [`requirements.txt`](https://github.com/Andrey890/Vertical-structure-of-accretion-discs/blob/master/requirements.txt)

``` shell
$ pip3 install -U pip
$ pip3 install -r requirements.txt
```

3. Install 'Vertical-structure-of-accretion-discs' package

``` shell
$ python3 setup.py install
```

4. Run python and try vs.main()

``` shell
$ python3 -m vs
```

	Finding Pi parameters of structure and making a structure plot. 
	Structure with Kramers opacity and ideal gas EOS.
	M = 1.98841e+34 grams
	r = 1.1813e+09 cm = 400 rg
	alpha = 0.01
	Mdot = 1e+17 g/s
	The vertical structure has been calculated successfully.
	Pi parameters = [7.10271534 0.4859551  1.13097882 0.3985615 ]
	z0/r =  0.028869666211114635
	Plot of structure is successfully saved to fig/vs.pdf.

### Tabular opacities and EOS

['mesa2py'](https://github.com/hombit/mesa2py) is used to bind tabular opacities and EOS 
from [MESA code](http://mesa.sourceforge.net) with Python3.
If you want to use tabular values of opacity and EOS to calculate the structure, you should use Docker.

You can use the latest pre-build Docker image:

``` shell
$ docker pull ghcr.io/andrey890/vertical-structure-of-accretion-discs:latest
$ docker tag ghcr.io/andrey890/vertical-structure-of-accretion-discs vertstr
```

Or build a docker image by yourself

``` shell
$ git clone https://github.com/Andrey890/Vertical-structure-of-accretion-discs.git
$ cd Vertical-structure-of-accretion-discs
$ docker build -t vertstr .
```

Then run 'vertstr' image as a container and try mesa_vs.main()

``` shell
$ docker run -v$(pwd)/fig:/app/fig --rm -ti vertstr python3 -m mesa_vs
```

	Calculating structure and making a structure plot.
	Structure with tabular MESA opacity and EOS.
	M = 1.98841e+34 grams
	r = 1.1813e+09 cm = 400 rg
	alpha = 0.01
	Mdot = 1e+17 g/s
	The vertical structure has been calculated successfully.
	z0/r =  0.02976707374263605
	Plot of structure is successfully saved to fig/vs_mesa.pdf.

As the result, the plot `vs_mesa.pdf` is saved to the `/app/fig/` (`/app/` is a WORKDIR) 
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
help(vs)
help(mesa_vs)
```

### Usage:
You can use `vs` module with different output parameters: mass of central object, alpha, radius and viscous torque

``` python3
import vs

M = 2e33  # Mass of central object in grams
alpha = 0.01  # alpha parameter
r = 2e9  # radius in cm
F = 2e34  # viscous torque
mu = 0.6  # molecular weight

# Structure with (Bell & Lin, 1994) power-law opacities and ideal gas EOS
# with molecular weight mu and radiative temperature gradient.

vertstr = vs.IdealBellLin1994VerticalStructure(Mx=M, alpha=alpha, r=r, F=F, mu=mu)  # create the structure
z0r, result = vertstr.fit()  # calculate structure
varkappa_C, rho_C, T_C, P_C, Sigma0 = vertstr.parameters_C()
print(z0r)  # semi-thickness z0/r of disc
print(varkappa_C, rho_C, T_C, P_C)  # Opacity, bulk density, temperature and gas pressure in the symmetry plane of disc
print(Sigma0)  # Surface density of disc
print(vertstr.tau())  # optical thickness of disc
```

### Usage with tabular opacities and EOS:

You can use `mesa_vs` module inside the Docker with different output parameters: 
mass of central object, alpha, radius and viscous torque

``` shell
$ docker run --rm -ti vertstr python3
```

``` python3
import mesa_vs

M = 2e33  # Mass of central object in grams
alpha = 0.01  # alpha parameter
r = 2e9  # radius in cm
F = 2e34  # viscous torque

# Structure with tabular MESA opacities and EOS 
# and both radiative and convective energy transport,
# default chemical composition is solar.

vertstr = mesa_vs.MesaVerticalStructureRadConv(Mx=M, alpha=alpha, r=r, F=F)  # create the structure
z0r, result = vertstr.fit()  # calculate structure
varkappa_C, rho_C, T_C, P_C, Sigma0 = vertstr.parameters_C()
print(z0r)  # semi-thickness z0/r of disc
print(varkappa_C, rho_C, T_C, P_C)  # Opacity, bulk density, temperature and gas pressure in the symmetry plane of disc
print(Sigma0)  # Surface density of disc
print(vertstr.tau())  # optical thickness of disc
```

You can run your own files, that use this code, inside Docker

``` shell
$ docker run -v/path_to/your_file/file.py:/app/your_code.py --rm -ti vertstr python3 file.py
```

## Irradiated discs

Module `mesa_vs` also contains classes, that represent 
the vertical structure of self-irradiated discs.

Irradiation can be taken into account in two ways:

1. Via either `T_irr` or `C_irr` parameters, that is, the irradiation temperature and irradiation constant. 
   It is a simple approach for irradiation, when the external flux doesn't penetrate into the disc and only heats the 
   disc surface.

2. In the second approximation the external flux is penetrated into the disc and affect the energy flux
   and disc temperature. In this case there are more additional parameters are required, that describe
   the incident spectral flux. Such parameters are: frequency range `nu_irr`, type of frequency range 
   `spectrum_irr_par` (see below), spectrum `spectrum_irr`, luminosity of irradiation source `L_X_irr` 
   and the cosine of incident angle `cos_theta_irr`. The spectral incident flux then will be 
   `F_nu_irr = L_X_irr / (4 * pi * r ** 2) * spectrum_irr`.

   1. Frequency range `nu_irr` is array-like and can be either in Hz or in energy units (keV), this is detemined by 
      `spectrum_irr_par` in `['nu', 'E_in_keV']`.
   2. Spectrum `spectrum_irr` can be either an array-like or a Python function. 
      In latter case the spectrum is calculated for the frequency range `nu_irr` and automatically normalized to unity 
      over `nu_irr`. Note, that units of `nu_irr` and units of `spectrum_irr` arguments must be consistent. 
      There are two optional parameters `args_spectrum_irr` and `kwargs_spectrum_irr` for arguments (keyword arguments) 
      of spectrum function.
   3. Cosine of incident angle `cos_theta_irr` can be either exact value or `None`. In the latter case
      cosine is calculated self-consistently as `1/8 * z0 / r`.

### Usage of simple irradiation scheme:

``` python3
import mesa_vs

M = 2e33  # Mass of central object in grams
alpha = 0.01  # alpha parameter
r = 2e9  # radius in cm
F = 2e34  # viscous torque
Tirr = 1.2e4  # irradiation temperature

# Structure with tabular MESA opacities and EOS,
# both radiative and convective energy transport,
# default chemical composition is solar, 
# external irradiation is taken into account
# through the simple scheme.

vertstr = mesa_vs.MesaVerticalStructureRadConvExternalIrradiationZeroAssumption(
                     Mx=M, alpha=alpha, r=r, F=F, T_irr=Tirr)  # create the structure
z0r, result = vertstr.fit()  # calculate structure
varkappa_C, rho_C, T_C, P_C, Sigma0 = vertstr.parameters_C()
print(z0r)  # semi-thickness z0/r of disc
print(varkappa_C, rho_C, T_C, P_C)  # Opacity, bulk density, temperature and gas pressure in the symmetry plane of disc
print(Sigma0)  # Surface density of disc
print(vertstr.tau())  # optical thickness of disc
print(vertstr.C_irr)  # corresponding irradiation constant
```

### Usage of advanced irradiation scheme:

Define the incident X-ray spectrum as a function of energy in keV:

``` python3
def power_law_exp_spectrum(E, n, scale):
   return (E / scale) ** n * np.exp(-E / scale)
```

Then calculate structure with this spectrum:

``` python3
import mesa_vs
import numpy as np

M = 2e33  # Mass of central object in grams
alpha = 0.01  # alpha parameter
r = 2e9  # radius in cm
F = 2e34  # viscous torque
nu_irr = np.geomspace(1, 10, 30)  # energy range from 1 to 10 keV
spectrum_par = 'E_in_keV'  # units of nu_irr if energy in keV
kwargs={'n': -0.4, 'scale': 8}  # spectrum function parameters
cos_theta_irr = None  # incident angle is calculated self-consistently as 1/8 * z0 / r
L_X_irr = 1.0 * 1.25e38 * M / 2e33  #  luminosity of X-ray source = 1.0 * L_eddington

# Structure with tabular MESA opacities and EOS,
# both radiative and convective energy transport,
# default chemical composition is solar, 
# external irradiation is taken into account
# through the advanced scheme.

vertstr = mesa_vs.MesaVerticalStructureRadConvExternalIrradiation(
                     Mx=M, alpha=alpha, r=r, F=F, nu_irr=nu_irr, 
                     spectrum_irr=power_law_exp_spectrum, L_X_irr=L_X_irr,
                     spectrum_irr_par=spectrum_par, 
                     kwargs_spectrum_irr=kwargs,
                     cos_theta_irr=cos_theta_irr)  # create the structure
result = vertstr.fit()  # calculate structure
z0r, Sigma0 = result.x  # Sigma0 is additional free parameter to find
varkappa_C, rho_C, T_C, P_C, Sigma0 = vertstr.parameters_C()
print(z0r)  # semi-thickness z0/r of disc
print(varkappa_C, rho_C, T_C, P_C)  # Opacity, bulk density, temperature and gas pressure in the symmetry plane of disc
print(Sigma0)  # Surface density of disc
print(vertstr.tau())  # optical thickness of disc
print(vertstr.C_irr, vertstr.T_irr)  # irradiation constant and temperature
```

## Make plots and tables with disc parameters

Module `plots_vs` contains functions, that calculate vertical and radial structure and S-curve and return tables 
with disc parameters and make plots. With plots_vs.main() the vertical structure, S-curve and radial structure 
can be calculated for default parameters, stored as a plot and tables.  
Try it
``` shell
$ python3 -m plots_vs
```

`plots_vs` contains three functions: `Structure_Plot`, `S_curve` and `Radial_Plot`. 

`Structure_Plot` returns table with parameters of disc as functions of vertical coordinate at specific radius. 
Also makes plot of structure.

`S_curve` calculates S-curve and return table with disc parameters on the curve. Also makes plot of S-curve.

`Radial_Plot` returns table with parameters of disc as functions of radius for a given radius range.

### Usage:
``` python3
M = 1.5 * 2e33  # 1.5 * M_sun
alpha = 0.2
r = 1e10
Teff = 1e4

plots_vs.Structure_Plot(M, alpha, r, Teff, input='Teff', mu=0.62, structure='BellLin', n=100, add_Pi_values=True,
                    	path_dots='vs.dat', make_pic=True, path_plot='vs.pdf',
                    	set_title=True,
                    	title=r'$M = {:g} \, M_{{\odot}}, r = {:g} \, {{\rm cm}}, \alpha = {:g}, T_{{\rm eff}} = {:g} \, '
                          	r'{{\rm K}}$'.format(M / 2e33, r, alpha, Teff))

plots_vs.S_curve(4e3, 1e4, M, alpha, r, input='Teff', structure='BellLin', mu=0.62, n=200, tau_break=False,
                 path_dots='S-curve.dat', add_Pi_values=True, make_pic=True, output='Mdot',
                 xscale='parlog', yscale='parlog', path_plot='S-curve.pdf', set_title=True,
                 title=r'$M = {:g} \, M_{{\odot}}, r = {:g} \, {{\rm cm}}, \alpha = {:g}$'.format(M / 2e33, r, alpha))

rg = 3e5 * (M / 2e33)  # Schwarzschild radius
plots_vs.Radial_Plot(M, alpha, 3.1 * rg, 1e3 * rg, 1, input='Mdot_Mdot_edd', structure='BellLin', mu=0.62, n=200, 
		     tau_break=True, path_dots='radial_struct.dat', add_Pi_values=True)
```
Both `plots_vs` module and functions in it have help
``` python3
help(plots_vs)
help(plots_vs.Structure_Plot)
help(plots_vs.S_curve)
help(plots_vs.Radial_Plot)
```
