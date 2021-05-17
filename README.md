# Vertical-structure-of-accretion-discs

Module 'vs' contains several classes that represent the vertical 
structure of accretion discs in X-ray binaries for different assumptions 
of opacity law. For given parameters the vertical structure of 
disc is calculated and can be used for research of disc stability.

Module 'mesa_vs' contains some additional classes, that represent 
the vertical structure for tabular opacities and convective energy transport.

'mesa2py' (https://github.com/hombit/mesa2py) is required for 'mesa_vs' structure for work.

## Usage:
1) Create and activate virtual environment

``` shell
$ python3 -m venv ~/.venv/vs
$ source ~/.venv/vs/bin/activate
```

2) Update pip and install all requirements from [`requirements.txt`](https://github.com/Andrey890/Vertical-structure-of-accretion-discs/blob/master/requirements.txt)

``` shell
$ pip3 install -U pip
$ pip3 install -r requirements.txt
```

2) Install 'Vertical-structure-of-accretion-discs' package

``` shell
$ python3 setup.py install
```

3) Run python and try vs.main()

```
$ python3
>>> import vs
>>> vs.main()
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

4) You can use 'vs' module with different output parameters: mass of central object, alpha, radius and viscous torque

``` python3
import vs

M = 2e33  # Mass of central object in grams
alpha = 0.01  # alpha parameter
r = 2e9  # radius in cm
F = 2e34  # viscous torque

vertstr = vs.IdealBellLin1994VerticalStructure(M, alpha, r, F)  # create structure with BellLin opacities and ideal gas EOS
z0r, result = vertstr.fit()  # calculate structure
print(z0r)  # semi-thickness of disc
```
