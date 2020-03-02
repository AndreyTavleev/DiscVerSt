import unittest
from numpy.testing import assert_allclose
from vertstr import FindPi
import numpy as np
from vs import IdealKramersVerticalStructure
from astropy import constants as const

sigmaSB = const.sigma_sb.cgs.value
G = const.G.cgs.value
c = const.c.cgs.value
M_sun = const.M_sun.cgs.value


class PiTestCase(unittest.TestCase):
    def test_kramers_Pi_rg(self):
        M = 10 * M_sun
        alpha = 0.3
        Mdot = 33.6e17
        rg = 2 * G * M / c ** 2
        mu = 0.62
        r = 3e4 * rg
        h = (G * M * r) ** (1 / 2)

        for r in np.linspace(2 * rg, 2.5e4 * rg, 50):
            with self.subTest(r=r):
                print('r/rg = ', r / rg)
                F = Mdot * h

                vs = IdealKramersVerticalStructure(M, alpha, r, F, mu=mu)
                vs.fit()
                tau0 = vs.tau0()
                fp = FindPi(tau0)

                actual = vs.Pi_finder()
                desired = fp.getPi()

                print(actual)
                print(desired)
                print(tau0)

                assert_allclose(actual[0], desired[0], atol=1e-2, rtol=1e-2,
                                err_msg='Pi1, tau0={}'.format(tau0))
                assert_allclose(actual[1:], desired[1:], atol=1e-2, rtol=1e-2,
                                err_msg='Pi2..4, tau0={}'.format(tau0))

    def test_kramers_Pi_Teff(self):
        M = 10 * M_sun
        alpha = 0.3
        rg = 2 * G * M / c ** 2
        mu = 0.62
        r = 3e4 * rg
        h = (G * M * r) ** (1 / 2)

        for Teff in np.linspace(2e3, 1e4, 50):
            with self.subTest(Teff=Teff):
                print('Teff = ', Teff)
                F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Teff ** 4

                vs = IdealKramersVerticalStructure(M, alpha, r, F, mu=mu)
                vs.fit()
                tau0 = vs.tau0()
                fp = FindPi(tau0)

                actual = vs.Pi_finder()
                desired = fp.getPi()

                print(actual)
                print(desired)
                print(tau0)

                assert_allclose(actual[0], desired[0], atol=1e-2, rtol=1e-2,
                                err_msg='Pi1, tau0={}'.format(tau0))
                assert_allclose(actual[1:], desired[1:], atol=1e-2, rtol=1e-2,
                                err_msg='Pi2..4, tau0={}'.format(tau0))


    def test_kramers_formulas_rg(self):
        M = 10 * M_sun
        alpha = 0.3
        Mdot = 0.336e17
        rg = 2 * G * M / c ** 2
        mu = 0.62
        r = 3e4 * rg
        h = (G * M * r) ** (1 / 2)

        for r in np.linspace(2 * rg, 2.5e4 * rg, 50):
            with self.subTest(r=r):
                print('r/rg = ', r / rg)
                F = Mdot * h

                vs = IdealKramersVerticalStructure(M, alpha, r, F, mu=mu)
                z0r, result = vs.fit()

                z0r_teor = 0.020 * (M / M_sun) ** (-3 / 8) * alpha ** (-1 / 10) * (r / 1e10) ** (1 / 8) * (
                            mu / 0.6) ** (-3 / 8) * (Mdot / 1e17) ** (3 / 20) * 2.6

                print(z0r, z0r_teor)

                assert_allclose(z0r, z0r_teor, atol=1e-3, rtol=1e-2)


if __name__ == '__main__':
    unittest.main()
