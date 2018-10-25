from astropy import units as u
from astropy import constants as q
import math


A = 1e17*u.g/u.s*1e5*u.cm**(1/2)*(q.G.cgs*q.M_sun.cgs)**(1/2)

#print(1/A)


B = 31.68*(1/A)**(7/10)
#print(B**(3/14)*0.0248)


QWE = ((q.G*q.M_sun/u.cm**3)**(3/2)).cgs  # omegaK**3
C = u.erg/(u.s*u.cm**2)  # Flux


print(QWE)
print(C)
print(C/QWE/u.cm**2)


print(q.G.cgs.value)


# My code changes.

# Second comment
