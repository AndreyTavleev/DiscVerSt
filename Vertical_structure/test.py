import unittest

from numpy.testing import assert_allclose
from vertstr import FindPi

from vs import IdealKramersVerticalStructure


class PiTestCase(unittest.TestCase):
    def test_kramers(self):
        M = 2e33
        alpha = 0.5
        r = 1e11
        F = 3e33

        vs = IdealKramersVerticalStructure(M, alpha, r, F)
        vs.fit()
        tau0 = vs.tau0()
        fp = FindPi(tau0)

        assert_allclose(vs.Pi_finder(), fp.getPi(), rtol=1e-3, atol=1e-3,
                        err_msg='tau0={}'.format(tau0))
