import unittest
from numpy.testing import assert_allclose
from vertstr import FindPi
import numpy as np
from vs import IdealKramersVerticalStructure
from astropy import constants as const

sigmaSB = const.sigma_sb.cgs.value
G = const.G.cgs.value


class PiTestCase(unittest.TestCase):
    def test_kramers(self):
        M = 6 * 2e33
        r = 1e11
        alpha = 0.5
        h = (G * M * r) ** (1 / 2)

        for Teff in np.linspace(2e3, 1e4, 50):
            with self.subTest(Teff=Teff):
                print('Teff = ', Teff)
                F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Teff ** 4

                vs = IdealKramersVerticalStructure(M, alpha, r, F)
                vs.fit()
                tau0 = vs.tau0()
                fp = FindPi(tau0)

                actual = vs.Pi_finder()
                desired = fp.getPi()

                assert_allclose(actual[0], desired[0], atol=1e-3, rtol=1e-3,
                                err_msg='Pi1, tau0={}'.format(tau0))
                assert_allclose(actual[1:], desired[1:], atol=1e-3, rtol=1e-3,
                                err_msg='Pi2..4, tau0={}'.format(tau0))


if __name__ == '__main__':
    unittest.main()
