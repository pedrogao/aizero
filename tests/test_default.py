
import unittest
from grad.lib import *


def numerical_diff(f: Function, x: Variable, eps=1e-4):
    """
    微分法求导
    """
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

    def test_add(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        y = add(x0, x1)
        self.assertEqual(y.data, np.array(5))


class GradTest(unittest.TestCase):
    def test_grad_same_variable_add(self):
        x = Variable(np.array(3.0))
        y = add(x, x)
        y.backward()
        self.assertEqual(x.grad, np.array(2.0))

    def test_grad_same_variable_add2(self):
        x = Variable(np.array(3.0))
        y = add(x, add(x, x))
        y.backward()
        self.assertEqual(x.grad, np.array(3.0))


class NoGradTest(unittest.TestCase):
    def test_no_grad(self):
        with no_grad():
            x = Variable(np.array(3.0))
            y = square(x)
        self.assertIsNone(x.grad)
        self.assertIsNone(y.grad)

    def test_no_grad2(self):
        x = Variable(np.array(3.0))
        with no_grad():
            y = square(x)
        self.assertIsNone(x.grad)
        self.assertIsNone(y.grad)

    def test_no_grad3(self):
        x = Variable(np.array(3.0))
        y = square(x)
        with no_grad():
            y.backward()
        self.assertEqual(x.grad, np.array(6.0))


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
