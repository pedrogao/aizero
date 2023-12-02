is_simple_core = True

if is_simple_core:
    from .core_simple import Variable
    from .core_simple import Function
    from .core_simple import using_config
    from .core_simple import no_grad
    from .core_simple import as_array
    from .core_simple import as_variable
    from .core_simple import setup_variable
else:
    from .core import *

setup_variable()
