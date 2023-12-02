import unittest
import numpy as np
from grad import Variable


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


class NTest(unittest.TestCase):
    def test_grad(self):
        x = Variable(np.array(2.0))
        y = f(x)
        y.backward(create_graph=True)
        self.assertEqual(x.grad.data, 24.0)
        gx = x.grad
        x.cleargrad()
        gx.backward()
        self.assertEqual(x.grad.data, 44.0)
