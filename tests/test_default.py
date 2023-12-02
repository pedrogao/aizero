
import unittest
import numpy as np
from grad import *


def numerical_diff(f: Function, x: Variable, eps=1e-4):
    """
    微分法求导
    """
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class VariablePropertyTest(unittest.TestCase):
    def test_property(self):
        x = Variable(np.array(3.0))
        self.assertEqual(x.shape, ())
        self.assertEqual(x.ndim, 0)
        self.assertEqual(x.size, 1)
        self.assertEqual(x.dtype, np.float64)

    def test_property2(self):
        x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        self.assertEqual(x.shape, (2, 3))
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.size, 6)
        self.assertEqual(x.dtype, np.float64)


class OperatorTest(unittest.TestCase):
    def test_add(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        y = x0 + x1
        self.assertEqual(y.data, np.array(5))

    def test_mul(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        y = x0 * x1
        self.assertEqual(y.data, np.array(6))
