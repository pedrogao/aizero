import numpy as np
import unittest
from grad.utils import get_dot_graph
from grad import Variable


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


class DotTest(unittest.TestCase):
    def test_dot(self):
        x = Variable(np.array(2))
        y = Variable(np.array(3))
        z = goldstein(x, y)
        z.backward()

        x.name = 'x'
        y.name = 'y'
        z.name = 'z'

        dot = get_dot_graph(z, verbose=False)
        print(dot)
