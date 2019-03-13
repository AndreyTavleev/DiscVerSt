import unittest

from numpy.testing import assert_allclose
from vertstr import FindPi
import numpy as np
from astropy import constants as cnst

from vs import IdealKramersVerticalStructure


class PiTestCase(unittest.TestCase):
    def test_kramers(self):
        M = 6 * 2e33
        r = 1e11
        alpha = 0.5
        # F = 3e33

        sigmaSB = cnst.sigma_sb.cgs.value
        G = cnst.G.cgs.value
        Teff = 2.3e3
        h = (G * M * r) ** (1 / 2)
        F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Teff ** 4
        print(F)

        vs = IdealKramersVerticalStructure(M, alpha, r, F)
        vs.fit()
        tau0 = vs.tau0()
        fp = FindPi(tau0)
        print(vs.Pi_finder())
        print(fp.getPi())
        print(np.array(vs.Pi_finder())-np.array(fp.getPi()))

        assert_allclose(vs.Pi_finder(), fp.getPi(), rtol=1e-3, atol=1e-3,
                        err_msg='tau0={}'.format(tau0))
