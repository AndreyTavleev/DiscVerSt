# Vertical-structure-of-accretion-discs

This code can calculate vertical structure of accretion discs around neutron stars and black holes.

### Installation

1. Create and activate virtual environment

``` shell
$ python3 -m venv ~/.venv/vs
$ source ~/.venv/vs/bin/activate
```

2. Update pip and install all requirements from [`requirements.txt`](https://github.com/Andrey890/Vertical-structure-of-accretion-discs/blob/master/requirements.txt)

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
	Plot of structure is successfully saved.
The plot of structure 'vs.pdf' is created in the same directory after that.

## Calculate structure

Module 'vs' contains several classes that represent the vertical 
structure of accretion discs in X-ray binaries for different assumptions 
of opacity law. For given parameters the vertical structure of 
disc is calculated and can be used for research of disc stability.

Module 'mesa_vs' contains some additional classes, that represent 
the vertical structure for tabular opacities and convective energy transport.

'mesa2py' (https://github.com/hombit/mesa2py) is required for 'mesa_vs' structure for work.

### Usage:
You can use 'vs' module with different output parameters: mass of central object, alpha, radius and viscous torque

``` python3
import vs

M = 2e33  # Mass of central object in grams
alpha = 0.01  # alpha parameter
r = 2e9  # radius in cm
F = 2e34  # viscous torque

vertstr = vs.IdealBellLin1994VerticalStructure(M, alpha, r, F)  # create structure with BellLin opacities and ideal gas EOS
z0r, result = vertstr.fit()  # calculate structure
varkappa_C, rho_C, T_C, P_C, Sigma0 = vertstr.parameters_C()
print(z0r)  # semi-thickness z0/r of disc
print(varkappa_C, rho_C, T_C, P_C)  # Opacity, bulk density, temperature and gas pressure in the symmetry plane of disc
print(Sigma0)  # Surface density of disc
print(vertstr.tau())  # optical thickness of disc
```
Both 'vs' and 'mesa_vs' modules have help
``` python3
help(vs)
help(mesa_vs)
```

## Make plots and tables with disc parameters

Module 'plots_vs' contains functions, that calculate structure and S-curve and return tables with disc parameters and make plots.
With plots_vs.main() the structure and S-curves can be calculated for default parameters, stored as a plot and tables.  
Try it
``` shell
$ python3 -m plots_vs
```

'plots_vs' contains two functions: 'Structure_Plot' and 'S_curve'. 
'Structure_Plot' returns table with parameters of disc as functions of vertical coordinate at specific radius. Also makes plot of structure.
'S_curve' calculates S-curve and return table with disc parameters on the curve. Also makes plot of S-curve.

### Usage:
``` python3
M = 1.5 * 2e33  # 1.5 * M_sun
alpha = 0.2
r = 1e10
Teff = 1e4

plots_vs.Structure_Plot(M, alpha, r, Teff, input='Teff', mu=0.62, structure='BellLin', n=100, add_Pi_values=True,
                    savedots=True, path_dots='vs.dat', make_pic=True, save_plot=True, path_plot='vs.pdf',
                    set_title=True,
                    title=r'$M = {:g} \, M_{{\odot}}, r = {:g} \, {{\rm cm}}, \alpha = {:g}, T_{{\rm eff}} = {:g} \, '
                          r'{{\rm K}}$'.format(M / 2e33, r, alpha, Teff))

plots_vs.S_curve(4e3, 1e4, M, alpha, r, input='Teff', structure='BellLin', mu=0.62, n=200, tau_break=False, savedots=True,
                 path_dots='S-curve.dat', add_Pi_values=True, make_pic=True, output='Mdot',
                 xscale='parlog', yscale='parlog', save_plot=True, path_plot='S-curve.pdf', set_title=True,
                 title=r'$M = {:g} \, M_{{\odot}}, r = {:g} \, {{\rm cm}}, \alpha = {:g}$'.format(M / 2e33, r, alpha))
```
Both 'plots_vs' module and functions in it have help
``` python3
help(plots_vs)
help(plots_vs.Structure_Plot)
help(plots_vs.S_curve)
```
