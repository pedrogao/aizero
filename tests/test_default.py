
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


def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


class ComplexFunctionTest(unittest.TestCase):
    def test_sphere(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = sphere(x, y)
        z.backward()
        self.assertEqual(x.grad.data, 2)
        self.assertEqual(y.grad.data, 2)

    def test_matyas(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = matyas(x, y)
        z.backward()
        self.assertEqual(x.grad.data, 0.040000000000000036)
        self.assertEqual(y.grad.data, 0.040000000000000036)

    def test_goldstein(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = goldstein(x, y)
        z.backward()
        self.assertEqual(x.grad.data, -5376.0)
        self.assertEqual(y.grad.data, 8064.0)
