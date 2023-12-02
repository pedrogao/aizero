"""
__init__.py

`grad` is a simple autograd engine.
"""

is_simple_core = False

if is_simple_core:
    from grad.core_simple import Function, Variable, as_array, as_variable, no_grad, setup_variable, using_config
else:
    from grad.core import Function, Variable, as_array, as_variable, no_grad, setup_variable, using_config

setup_variable()
