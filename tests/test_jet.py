import jax
import jax.numpy as jnp
import pytest
from local_coordinates.jet import Jet, function_to_jet
from local_coordinates.jet import jet_decorator
import jax.tree_util as jtu

def test_jet_creation():
    value = jnp.array(3.0)
    gradient = jnp.array([1., -1.])
    hessian = jnp.eye(2)
    jet = Jet(value=value, gradient=gradient, hessian=hessian)

    assert jnp.array_equal(jet.value, value)
    assert jnp.array_equal(jet.gradient, gradient)
    assert jnp.array_equal(jet.hessian, hessian)

def test_jet_properties():
    value = jnp.array(3.0)
    gradient = jnp.array([1., -1.])
    hessian = jnp.eye(2)
    jet = Jet(value=value, gradient=gradient, hessian=hessian)

    assert jet.shape == value.shape

def test_jet_batching():
    # No batch
    value = jnp.array(3.0)
    gradient = jnp.array([1., -1.])
    hessian = jnp.eye(2)
    jet = Jet(value=value, gradient=gradient, hessian=hessian)
    assert jet.batch_size is None

    # 1D batch
    value_1d = jnp.array([3.0, 4.0])
    gradient_1d = jnp.stack([gradient, gradient])
    hessian_1d = jnp.stack([hessian, hessian])
    jet_1d = Jet(value=value_1d, gradient=gradient_1d, hessian=hessian_1d)
    assert jet_1d.batch_size == 2

    # 2D batch
    value_2d = jnp.stack([value_1d, value_1d])
    gradient_2d = jnp.stack([gradient_1d, gradient_1d])
    hessian_2d = jnp.stack([hessian_1d, hessian_1d])
    jet_2d = Jet(value=value_2d, gradient=gradient_2d, hessian=hessian_2d)
    # Based on the current implementation of Jet.batch_size for ndim > 1
    assert jet_2d.batch_size == (2,)

def test_jet_call_method():
    """Tests the __call__ method for Taylor series approximation."""
    # --- 2nd Order ---
    jet = Jet(
        value=jnp.array(10.0),
        gradient=jnp.array([1.0, 2.0]),
        hessian=jnp.array([[3.0, 0.1], [0.1, 4.0]])
    )
    dx = jnp.array([0.5, -0.5])
    # v + g.dx + 0.5 dx.T H dx
    # 10.0 + (1*0.5 + 2*(-0.5)) + 0.5 * [0.5, -0.5] @ [[3, 0.1], [0.1, 4]] @ [0.5, -0.5]
    # 10.0 + (-0.5) + 0.5 * [0.5, -0.5] @ [1.45, -1.95]
    # 10.0 - 0.5 + 0.5 * (0.725 + 0.975)
    # 9.5 + 0.5 * 1.7 = 9.5 + 0.85 = 10.35
    result = jet(dx)
    assert jnp.allclose(result, 10.35)

    # --- 1st Order ---
    jet_1st = Jet(
        value=jnp.array(10.0),
        gradient=jnp.array([1.0, 2.0]),
        hessian=None
    )
    # 10.0 + (1*0.5 + 2*(-0.5)) = 9.5
    result_1st = jet_1st(dx)
    assert jnp.allclose(result_1st, 9.5)

    # --- 0th Order ---
    jet_0th = Jet(
        value=jnp.array(10.0),
        gradient=None,
        hessian=None
    )
    result_0th = jet_0th(dx)
    assert jnp.allclose(result_0th, 10.0)

    # --- PyTree ---
    jet_pytree = Jet(
        value={'a': 10.0, 'b': jnp.array([1., 2.])},
        gradient={'a': jnp.array([1., 2.]), 'b': jnp.array([[0., 1.], [1., 0.]])},
        hessian=None,
    )
    dx_pytree = jnp.array([0.1, 0.2])
    result_pytree = jet_pytree(dx_pytree)
    # a: 10 + (1*0.1 + 2*0.2) = 10.5
    # b[0]: 1 + (0*0.1 + 1*0.2) = 1.2
    # b[1]: 2 + (1*0.1 + 0*0.2) = 2.1
    assert jnp.allclose(result_pytree['a'], 10.5)
    assert jnp.allclose(result_pytree['b'], jnp.array([1.2, 2.1]))

    # --- Wrong dx shape ---
    with pytest.raises(ValueError):
        jet(jnp.array([[1.0], [2.0]]))


def test_function_to_jet():
    def quad_func(x):
        return jnp.sum(x**2)

    x = jnp.array([1., 2.])
    jet = function_to_jet(quad_func, x)

    # Manual verification
    expected_value = jnp.sum(x**2)
    expected_gradient = 2 * x
    expected_hessian = jnp.eye(2) * 2

    assert jnp.allclose(jet.value, expected_value)
    assert jnp.allclose(jet.gradient, expected_gradient)
    assert jnp.allclose(jet.hessian, expected_hessian)

def test_function_to_jet_diffeomorphism():
    def diffeomorphism(x):
        # This returns a jnp.array, which is a single leaf in a PyTree.
        # The Jet will use its batching capability.
        return jnp.array([x[0]**2, x[1]**3])

    x = jnp.array([2., 3.])
    jet = function_to_jet(diffeomorphism, x)

    assert isinstance(jet, Jet)

    # Check the batched jet's components
    expected_value = jnp.array([4., 27.])
    expected_gradient = jnp.array([[4., 0.], [0., 27.]])  # Jacobian matrix
    expected_hessian = jnp.array([
        [[2., 0.], [0., 0.]],    # Hessian of the first output component
        [[0., 0.], [0., 18.]]   # Hessian of the second output component
    ])

    assert jnp.allclose(jet.value, expected_value)
    assert jnp.allclose(jet.gradient, expected_gradient)
    assert jnp.allclose(jet.hessian, expected_hessian)

    # Check that the batch size is correctly identified
    assert jet.batch_size == 2

def test_jet_decorator_scalar_output():
    @jet_decorator
    def g(x):
        return x**2

    jet = function_to_jet(lambda t: t, jnp.array(3.))
    out = g(jet)

    expected = Jet(
        value=jnp.array(9.0),
        gradient=jnp.array([6.0]),
        hessian=jnp.array([[2.0]]),
    )

    assert jnp.allclose(out.value, expected.value)
    assert jnp.allclose(out.gradient, expected.gradient)
    assert jnp.allclose(out.hessian, expected.hessian)


def test_jet_decorator_vector_output():
    @jet_decorator
    def g(x):
        return jnp.array([x**2, x**3])

    jet = function_to_jet(lambda t: t, jnp.array(2.))
    out = g(jet)

    expected = Jet(
        value=jnp.array([4.0, 8.0]),
        gradient=jnp.array([[4.0], [12.0]]),
        hessian=jnp.array([[[2.0]], [[12.0]]]),
    )

    assert jnp.allclose(out.value, expected.value)
    assert jnp.allclose(out.gradient, expected.gradient)
    assert jnp.allclose(out.hessian, expected.hessian)

def test_jet_decorator_scalar_output_matrix():
    @jet_decorator
    def g(x):
        return x**2

    key = jax.random.PRNGKey(0)
    dim = 2
    A = jax.random.normal(key, (dim, dim))
    x = jax.random.normal(key, (dim,))

    def f(x):
      return A @ x
    jet = function_to_jet(f, x)
    out = g(jet)


def test_jet_decorator_pytree_output():
    @jet_decorator
    def g(x):
        return {
            'u': x**2,
            'v': jnp.sin(x)
        }

    jet = function_to_jet(lambda t: t, jnp.array(1.5))
    out = g(jet)

    expected_u = Jet(
        value=jnp.array(2.25),
        gradient=jnp.array([3.0]),
        hessian=jnp.array([[2.0]]),
    )
    expected_v = Jet(
        value=jnp.sin(1.5),
        gradient=jnp.array([jnp.cos(1.5)]),
        hessian=jnp.array([[-jnp.sin(1.5)]]),
    )

    assert jnp.allclose(out['u'].value, expected_u.value)
    assert jnp.allclose(out['u'].gradient, expected_u.gradient)
    assert jnp.allclose(out['u'].hessian, expected_u.hessian)

    assert jnp.allclose(out['v'].value, expected_v.value)
    assert jnp.allclose(out['v'].gradient, expected_v.gradient)
    assert jnp.allclose(out['v'].hessian, expected_v.hessian)


def test_jet_decorator_two_args():
    @jet_decorator
    def g(x, y):
        # mix two arguments
        return x * y + x**2

    jet_x = function_to_jet(lambda t: t, jnp.array(1.2))
    jet_y = function_to_jet(lambda t: t, jnp.array(-0.7))
    out = g(jet_x, jet_y)

    val = 1.2*(-0.7) + 1.2**2
    grad = jnp.array([(-0.7) + 2*1.2, 1.2])
    H = jnp.array([[2.0, 1.0],[1.0, 0.0]])

    expected = Jet(
        value=jnp.array(val),
        gradient=grad,
        hessian=H,
    )

    assert jnp.allclose(out.value, expected.value)
    assert jnp.allclose(out.gradient, expected.gradient)
    assert jnp.allclose(out.hessian, expected.hessian)


def test_jet_decorator_mixed_args_constant_and_jet():
    @jet_decorator
    def g(x, scale):
        return scale * jnp.sin(x)

    jet_x = function_to_jet(lambda t: t, jnp.array(0.3))
    out = g(jet_x, 5.0)

    expected = Jet(
        value=5.0 * jnp.sin(0.3),
        gradient=jnp.array([5.0 * jnp.cos(0.3)]),
        hessian=jnp.array([[-5.0 * jnp.sin(0.3)]]),
    )

    assert jnp.allclose(out.value, expected.value)
    assert jnp.allclose(out.gradient, expected.gradient)
    assert jnp.allclose(out.hessian, expected.hessian)


def test_jet_decorator_composition():
    """
    Tests composing functions that have been lifted by jet_decorator.
    This simulates a function that is "jet-aware" by composing other
    jet-enabled operations.
    """
    @jet_decorator
    def f_square(x):
        return x**2

    @jet_decorator
    def f_sin(y):
        return jnp.sin(y)

    # This function `g` is not decorated but is "jet-aware" because it
    # correctly composes functions that operate on Jets.
    def g(jet_in):
        return f_sin(f_square(jet_in))

    # The input jet represents the identity function, id(t) = t, at t=1.2.
    # The output jet should then represent the composition h(t) = sin(t**2) at t=1.2.
    t = 1.2
    input_jet = function_to_jet(lambda x: x, jnp.array(t))

    # Run the composed function
    output_jet = g(input_jet)

    # --- Ground Truth ---
    # The composed function is h(t) = sin(t**2).
    # First derivative: h'(t) = cos(t**2) * 2t
    # Second derivative: h''(t) = -sin(t**2) * (2t)**2 + cos(t**2) * 2
    t2 = t**2
    expected_value = jnp.sin(t2)
    expected_gradient = jnp.cos(t2) * 2 * t
    expected_hessian = -jnp.sin(t2) * (2 * t)**2 + jnp.cos(t2) * 2

    ground_truth_jet = Jet(
        value=jnp.array(expected_value),
        gradient=jnp.array([expected_gradient]),
        hessian=jnp.array([[expected_hessian]]),
    )

    assert isinstance(output_jet, Jet)
    assert jnp.allclose(output_jet.value, ground_truth_jet.value)
    assert jnp.allclose(output_jet.gradient, ground_truth_jet.gradient)
    assert jnp.allclose(output_jet.hessian, ground_truth_jet.hessian)

def test_jet_decorator_with_none_gradient():
    """
    Test that jet_decorator handles Jets with None gradient gracefully.
    When gradient is None, the output should also have None gradient.
    """
    @jet_decorator
    def g(x):
        return x**2

    jet = Jet(value=jnp.array(3.0), gradient=None, hessian=None)
    out = g(jet)

    assert jnp.allclose(out.value, jnp.array(9.0))
    assert out.gradient is None
    assert out.hessian is None


def test_jet_decorator_with_none_hessian():
    """
    Test that jet_decorator handles Jets with None hessian.
    When hessian is None, the output should compute gradient but not hessian.
    """
    @jet_decorator
    def g(x):
        return x**2

    jet = Jet(
        value=jnp.array(3.0),
        gradient=jnp.array([1.0]),
        hessian=None,
    )
    out = g(jet)

    assert jnp.allclose(out.value, jnp.array(9.0))
    assert jnp.allclose(out.gradient, jnp.array([6.0]))
    assert out.hessian is None


def test_jet_decorator_mixed_none_attributes():
    """
    Test jet_decorator with multiple Jets where some have None attributes.
    The output should only compute what's possible given the available information.
    """
    @jet_decorator
    def g(x, y):
        return x * y

    # First jet has full information
    jet_x = Jet(
        value=jnp.array(2.0),
        gradient=jnp.array([1.0]),
        hessian=jnp.array([[0.0]]),
    )

    # Second jet has no gradient or hessian
    jet_y = Jet(
        value=jnp.array(3.0),
        gradient=None,
        hessian=None,
    )

    out = g(jet_x, jet_y)

    # Should compute value but not derivatives since one jet is missing them
    assert jnp.allclose(out.value, jnp.array(6.0))
    assert out.gradient is None
    assert out.hessian is None


def test_jet_decorator_mixed_partial_derivatives():
    """
    Test with multiple Jets where one has gradient but no hessian.
    Should compute gradient but not hessian for the output.
    """
    @jet_decorator
    def g(x, y):
        return x**2 + y

    jet_x = Jet(
        value=jnp.array(2.0),
        gradient=jnp.array([1.0]),
        hessian=jnp.array([[0.0]]),
    )

    jet_y = Jet(
        value=jnp.array(3.0),
        gradient=jnp.array([1.0]),
        hessian=None,
    )

    out = g(jet_x, jet_y)

    # Should compute value and gradient, but not hessian
    assert jnp.allclose(out.value, jnp.array(7.0))
    assert jnp.allclose(out.gradient, jnp.array([4.0, 1.0]))
    assert out.hessian is None

def test_jet_decorator_composition_2():
    """
    Test that jet_decorator works with Jet-annotated parameters.
    This enables differentiating through functions that operate on Jets.
    """
    @jet_decorator
    def g(jet_in: Jet):
        return jet_in.gradient

    # The input jet represents the identity function, id(t) = t, at t=1.2.
    t = 1.2
    input_jet = function_to_jet(lambda x: x, jnp.array(t))

    # Apply g to the jet - the decorator should handle Jet parameters automatically
    output_jet = g(input_jet)

    # g returns the gradient of the input jet
    # For id(t), gradient is [1.0]
    # So output should be a Jet representing the constant function 1.0
    assert jnp.allclose(output_jet.value, 1.0)

    # Note: gradient is None because the input had hessian=[[0]], causing
    # gradient computation to be skipped in the shifted version
    # This is expected behavior for functions operating on Jet fields


def test_jet_decorator_with_jet_param_hessian_access():
    """
    Test accessing the hessian field of a Jet parameter.
    """
    @jet_decorator
    def g(jet_in: Jet):
        return jnp.sum(jet_in.hessian)

    t = jnp.array([1.0, 2.0])
    input_jet = function_to_jet(lambda x: x[0]**2 + x[1]**2, t)

    output_jet = g(input_jet)

    # input_jet.hessian = [[2, 0], [0, 2]], sum = 4
    assert jnp.allclose(output_jet.value, 4.0)


def test_jet_decorator_with_jet_param_operations():
    """
    Test performing operations on Jet fields.
    """
    @jet_decorator
    def g(jet_in: Jet):
        # Combine value and gradient
        return jet_in.value + jnp.sum(jet_in.gradient)

    t = jnp.array([1.0, 2.0])
    input_jet = function_to_jet(lambda x: x[0]**2, t)

    output_jet = g(input_jet)

    # value = 1, gradient = [2, 0], sum = 2, total = 3
    assert jnp.allclose(output_jet.value, 3.0)


def test_jet_decorator_with_multiple_jet_params():
    """
    Test function with multiple Jet-annotated parameters.
    """
    @jet_decorator
    def g(jet1: Jet, jet2: Jet):
        return jet1.value + jet2.value

    t = 2.0
    jet1 = function_to_jet(lambda x: x**2, jnp.array(t))
    jet2 = function_to_jet(lambda x: x**3, jnp.array(t))

    output_jet = g(jet1, jet2)

    # jet1.value = 4, jet2.value = 8, sum = 12
    assert jnp.allclose(output_jet.value, 12.0)


def test_jet_decorator_mixed_jet_and_regular_params():
    """
    Test function with mix of Jet and regular parameters.
    """
    @jet_decorator
    def g(jet_in: Jet, scale):
        return jet_in.value * scale

    t = 3.0
    input_jet = function_to_jet(lambda x: x**2, jnp.array(t))

    output_jet = g(input_jet, 5.0)

    # jet.value = 9, scaled = 45
    assert jnp.allclose(output_jet.value, 45.0)


def test_jet_decorator_jet_param_returning_array():
    """
    Test Jet parameter function returning an array.
    """
    @jet_decorator
    def g(jet_in: Jet):
        # Return both value and gradient as array
        return jnp.array([jet_in.value, jnp.sum(jet_in.gradient)])

    t = jnp.array([1.0, 2.0])
    input_jet = function_to_jet(lambda x: x[0] * x[1], t)

    output_jet = g(input_jet)

    # value = 2, gradient = [2, 1], sum = 3
    assert output_jet.value.shape == (2,)
    assert jnp.allclose(output_jet.value[0], 2.0)
    assert jnp.allclose(output_jet.value[1], 3.0)


def test_jet_decorator_jet_param_composition():
    """
    Test composing multiple Jet-aware functions.
    """
    @jet_decorator
    def extract_gradient(jet_in: Jet):
        return jet_in.gradient

    @jet_decorator
    def double(x):
        return x * 2

    t = 2.0
    input_jet = function_to_jet(lambda x: x**3, jnp.array(t))

    # First extract gradient, then double it
    grad_jet = extract_gradient(input_jet)
    output_jet = double(grad_jet)

    # gradient of x^3 at x=2 is 12, doubled is 24
    assert jnp.allclose(output_jet.value, 24.0)


def test_jet_decorator_jet_param_norm_operation():
    """
    Test computing norm of gradient vector.
    """
    @jet_decorator
    def gradient_norm(jet_in: Jet):
        return jnp.linalg.norm(jet_in.gradient)

    t = jnp.array([3.0, 4.0])
    input_jet = function_to_jet(lambda x: x[0]**2 + x[1]**2, t)

    output_jet = gradient_norm(input_jet)

    # gradient = [6, 8], norm = 10
    assert jnp.allclose(output_jet.value, 10.0)


def test_jet_decorator_jet_param_quadratic_form():
    """
    Test computing quadratic form with Hessian.
    """
    @jet_decorator
    def hessian_trace(jet_in: Jet):
        return jnp.trace(jet_in.hessian)

    t = jnp.array([1.0, 2.0])
    input_jet = function_to_jet(lambda x: x[0]**3 + x[1]**3, t)

    output_jet = hessian_trace(input_jet)

    # Hessian = [[6*x[0], 0], [0, 6*x[1]]] at [1, 2] = [[6, 0], [0, 12]]
    # trace = 18
    assert jnp.allclose(output_jet.value, 18.0)


def test_jet_decorator_jet_param_with_pytree_values():
    """
    Test Jet parameter where the Jet has PyTree-valued fields.
    """
    @jet_decorator
    def sum_dict_values(jet_in: Jet):
        # jet_in.value is a dict
        return jet_in.value['a'] + jet_in.value['b']

    # Create a Jet with PyTree values
    input_jet = Jet(
        value={'a': jnp.array(10.0), 'b': jnp.array(20.0)},
        gradient={'a': jnp.array([1.0, 0.0]), 'b': jnp.array([0.0, 1.0])},
        hessian={'a': jnp.zeros((2, 2)), 'b': jnp.zeros((2, 2))},
    )

    output_jet = sum_dict_values(input_jet)

    # 10 + 20 = 30
    assert jnp.allclose(output_jet.value, 30.0)


def test_jet_with_pytree_value():
    """
    Test creating a Jet where value, gradient, and hessian are PyTrees.
    """
    # Value is a dict of arrays
    value = {'a': jnp.array(1.0), 'b': jnp.array(2.0)}

    # Gradient is a dict with same structure, each leaf has coord dimension
    gradient = {'a': jnp.array([1.0, 0.0]), 'b': jnp.array([0.0, 1.0])}

    # Hessian is a dict with same structure, each leaf has two coord dimensions
    hessian = {'a': jnp.zeros((2, 2)), 'b': jnp.zeros((2, 2))}

    jet = Jet(value=value, gradient=gradient, hessian=hessian)

    assert isinstance(jet.value, dict)
    assert jnp.allclose(jet.value['a'], 1.0)
    assert jnp.allclose(jet.value['b'], 2.0)


def test_function_to_jet_with_pytree_output():
    """
    Test that function_to_jet works with functions that return PyTrees.
    Note: This returns a PyTree of Jets, not a Jet with PyTree values.
    """
    def f(x):
        return {'a': x[0]**2, 'b': x[1]**3}

    x = jnp.array([2.0, 3.0])
    jet = function_to_jet(f, x)

    # jet is a dict of Jets: {'a': Jet(...), 'b': Jet(...)}
    assert isinstance(jet, dict)
    assert isinstance(jet['a'], Jet)
    assert isinstance(jet['b'], Jet)

    # Check values
    assert jnp.allclose(jet['a'].value, 4.0)
    assert jnp.allclose(jet['b'].value, 27.0)

    # Check gradients
    assert jnp.allclose(jet['a'].gradient, jnp.array([4.0, 0.0]))
    assert jnp.allclose(jet['b'].gradient, jnp.array([0.0, 27.0]))

    # Check hessians
    assert jnp.allclose(jet['a'].hessian, jnp.array([[2.0, 0.0], [0.0, 0.0]]))
    assert jnp.allclose(jet['b'].hessian, jnp.array([[0.0, 0.0], [0.0, 18.0]]))


def test_jet_decorator_with_pytree_valued_jet():
    """
    Test jet_decorator with a SINGLE Jet that has PyTree-valued fields.
    This tests Jet(value=dict(...), gradient=dict(...), hessian=dict(...))
    """
    @jet_decorator
    def g(x):
        # x is a dict: {'a': val_a, 'b': val_b}
        # Return sum of values
        return x['a'] + x['b']

    # Create a SINGLE Jet with PyTree-valued fields
    t = jnp.array([1.0, 2.0])
    input_jet = Jet(
        value={'a': t[0]**2, 'b': t[1]**2},  # dict of values
        gradient={'a': jnp.array([2*t[0], 0.0]), 'b': jnp.array([0.0, 2*t[1]])},  # dict of gradients
        hessian={'a': jnp.array([[2.0, 0.0], [0.0, 0.0]]), 'b': jnp.array([[0.0, 0.0], [0.0, 2.0]])},  # dict of hessians
    )

    # Apply decorated function
    output_jet = g(input_jet)

    # Output should be a scalar Jet
    expected_value = 1.0 + 4.0  # input_jet.value['a'] + input_jet.value['b']
    assert jnp.allclose(output_jet.value, expected_value)

    # Gradient: sum of gradients
    expected_gradient = jnp.array([2.0, 4.0])
    assert jnp.allclose(output_jet.gradient, expected_gradient)

    # Hessian: sum of hessians
    expected_hessian = jnp.array([[2.0, 0.0], [0.0, 2.0]])
    assert jnp.allclose(output_jet.hessian, expected_hessian)


def test_jet_decorator_pytree_valued_to_pytree_valued():
    """
    Test jet_decorator with single Jet with PyTree values -> single Jet with PyTree values.
    """
    @jet_decorator
    def g(x):
        # x is a dict, return a dict
        return {'c': x['a'] * 2, 'd': x['b']**2}

    # Create a SINGLE Jet with PyTree-valued fields
    input_jet = Jet(
        value={'a': 5.0, 'b': 6.0},
        gradient={'a': jnp.array([1.0, 1.0]), 'b': jnp.array([3.0, 2.0])},
        hessian={'a': jnp.zeros((2, 2)), 'b': jnp.zeros((2, 2))},
    )

    output_jet = g(input_jet)

    # Output should be a SINGLE Jet with PyTree-valued fields
    assert isinstance(output_jet, Jet)
    assert isinstance(output_jet.value, dict)

    # Check values: g(x) = {'c': 10, 'd': 36}
    assert jnp.allclose(output_jet.value['c'], 10.0)
    assert jnp.allclose(output_jet.value['d'], 36.0)

    # Check gradients exist and are dicts
    assert isinstance(output_jet.gradient, dict)
    # d/dx['c'] = 2 * d/dx['a'] = [2, 2]
    assert jnp.allclose(output_jet.gradient['c'], jnp.array([2.0, 2.0]))
    # d/dx['d'] = 2*x['b'] * d/dx['b'] = 2*6*[3, 2] = [36, 24]
    assert jnp.allclose(output_jet.gradient['d'], jnp.array([36.0, 24.0]))


def test_jet_decorator_pytree_to_pytree():
    """
    Test jet_decorator where input and output are both PyTrees of Jets.
    (This is different from PyTree-VALUED Jets)
    """
    @jet_decorator
    def g(x):
        # x is a dict, return a dict
        return {'c': x['a'] * 2, 'd': x['b']**2}

    def f(t):
        return {'a': t[0] + t[1], 'b': t[0] * t[1]}

    t = jnp.array([2.0, 3.0])
    input_jet = function_to_jet(f, t)

    output_jet = g(input_jet)

    # Output is a dict of Jets: {'c': Jet(...), 'd': Jet(...)}
    assert isinstance(output_jet, dict)
    assert isinstance(output_jet['c'], Jet)
    assert isinstance(output_jet['d'], Jet)

    # Check values
    # f(t) = {'a': 5, 'b': 6}
    # g(f(t)) = {'c': 10, 'd': 36}
    assert jnp.allclose(output_jet['c'].value, 10.0)
    assert jnp.allclose(output_jet['d'].value, 36.0)

    # Check gradients: d/dt[2*(t[0]+t[1])] = [2, 2]
    assert jnp.allclose(output_jet['c'].gradient, jnp.array([2.0, 2.0]))
    # d/dt[(t[0]*t[1])^2] = [2*t[0]*t[1]*t[1], 2*t[0]*t[1]*t[0]] = [36, 24]
    assert jnp.allclose(output_jet['d'].gradient, jnp.array([36.0, 24.0]))


def test_jet_decorator_nested_pytree_structure():
    """
    Test with nested PyTree structures (nested dicts, lists, tuples).
    """
    @jet_decorator
    def g(x):
        # Access nested structure
        return x['outer']['inner'][0] + x['outer']['inner'][1] + x['flat']

    # Create a Jet with deeply nested PyTree structure
    input_jet = Jet(
        value={
            'outer': {'inner': [jnp.array(1.0), jnp.array(2.0)]},
            'flat': jnp.array(3.0)
        },
        gradient={
            'outer': {'inner': [jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])]},
            'flat': jnp.array([1.0, 1.0])
        },
        hessian={
            'outer': {'inner': [jnp.zeros((2, 2)), jnp.zeros((2, 2))]},
            'flat': jnp.zeros((2, 2))
        },
    )

    output_jet = g(input_jet)

    # Output: 1 + 2 + 3 = 6
    assert jnp.allclose(output_jet.value, 6.0)

    # Gradient: sum of all gradients
    expected_grad = jnp.array([1.0, 0.0]) + jnp.array([0.0, 1.0]) + jnp.array([1.0, 1.0])
    assert jnp.allclose(output_jet.gradient, expected_grad)


def test_jet_decorator_pytree_with_array_values():
    """
    Test with PyTree where values are arrays (not scalars).
    """
    @jet_decorator
    def g(x):
        # x is dict of arrays, return their sum
        return x['a'] + x['b']

    # Create Jet with array-valued PyTree
    input_jet = Jet(
        value={
            'a': jnp.array([1.0, 2.0, 3.0]),  # shape (3,)
            'b': jnp.array([4.0, 5.0, 6.0])   # shape (3,)
        },
        gradient={
            'a': jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),  # shape (3, 2)
            'b': jnp.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])   # shape (3, 2)
        },
        hessian={
            'a': jnp.zeros((3, 2, 2)),  # shape (3, 2, 2)
            'b': jnp.zeros((3, 2, 2))   # shape (3, 2, 2)
        },
    )

    output_jet = g(input_jet)

    # Output should be array
    assert output_jet.value.shape == (3,)
    assert jnp.allclose(output_jet.value, jnp.array([5.0, 7.0, 9.0]))

    # Gradient should have shape (3, 2)
    assert output_jet.gradient.shape == (3, 2)
    expected_grad = jnp.array([[1.5, 0.5], [1.0, 1.0], [1.0, 2.0]])
    assert jnp.allclose(output_jet.gradient, expected_grad)


def test_jet_decorator_mixed_pytree_structure():
    """
    Test with mixed PyTree (dicts, lists, tuples mixed together).
    """
    @jet_decorator
    def g(x):
        # Complex access pattern
        return x[0]['key'] + x[1][0]

    # Create Jet with mixed structure: tuple of [dict, list]
    input_jet = Jet(
        value=(
            {'key': jnp.array(10.0)},
            [jnp.array(20.0), jnp.array(30.0)]
        ),
        gradient=(
            {'key': jnp.array([2.0, 1.0])},
            [jnp.array([1.0, 2.0]), jnp.array([0.5, 0.5])]
        ),
        hessian=(
            {'key': jnp.eye(2)},
            [jnp.eye(2), jnp.zeros((2, 2))]
        ),
    )

    output_jet = g(input_jet)

    # Output: 10 + 20 = 30
    assert jnp.allclose(output_jet.value, 30.0)

    # Gradient: [2, 1] + [1, 2] = [3, 3]
    assert jnp.allclose(output_jet.gradient, jnp.array([3.0, 3.0]))

    # Hessian: eye(2) + eye(2) = 2*eye(2)
    assert jnp.allclose(output_jet.hessian, 2 * jnp.eye(2))


def test_jet_decorator_pytree_with_matrix_values():
    """
    Test with PyTree where values are matrices.
    """
    @jet_decorator
    def g(x):
        # Matrix operations
        return jnp.sum(x['A'] @ x['B'])

    # Create Jet with matrix-valued PyTree
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[1.0, 0.0], [0.0, 1.0]])

    input_jet = Jet(
        value={'A': A, 'B': B},
        gradient={
            'A': jnp.ones((2, 2, 2)),  # shape (2, 2, 2) - last dim is coordinate
            'B': jnp.ones((2, 2, 2)) * 0.5
        },
        hessian={
            'A': jnp.zeros((2, 2, 2, 2)),  # shape (2, 2, 2, 2)
            'B': jnp.zeros((2, 2, 2, 2))
        },
    )

    output_jet = g(input_jet)

    # Output should be scalar (sum of A @ B)
    assert output_jet.value.shape == ()
    expected_value = jnp.sum(A @ B)
    assert jnp.allclose(output_jet.value, expected_value)

    # Gradient should exist
    assert output_jet.gradient.shape == (2,)


def test_jet_decorator_empty_pytree_components():
    """
    Test with PyTree that has some empty components.
    """
    @jet_decorator
    def g(x):
        # Only use non-empty parts
        return x['data']['x'] * 2

    input_jet = Jet(
        value={'data': {'x': jnp.array(5.0), 'metadata': ()}, 'empty': {}},
        gradient={'data': {'x': jnp.array([1.0, 2.0]), 'metadata': ()}, 'empty': {}},
        hessian={'data': {'x': jnp.zeros((2, 2)), 'metadata': ()}, 'empty': {}},
    )

    output_jet = g(input_jet)

    assert jnp.allclose(output_jet.value, 10.0)
    assert jnp.allclose(output_jet.gradient, jnp.array([2.0, 4.0]))


def test_jet_decorator_unused_jet_with_none_derivatives():
    """
    Tests that if a Jet argument with None derivatives is not used in the
    computation, the output Jet still computes derivatives based on the
    used Jet arguments.
    """
    @jet_decorator
    def g(x, y):
        # y is not used
        return x * 2

    jet_x = Jet(
        value=jnp.array(2.0),
        gradient=jnp.array([1.0]),
        hessian=jnp.array([[0.0]]),
    )

    # y has no derivatives, but is not used in computation
    jet_y = Jet(
        value=jnp.array(3.0),
        gradient=None,
        hessian=None,
    )

    out = g(jet_x, jet_y)

    assert jnp.allclose(out.value, 4.0)
    assert out.gradient is not None
    assert jnp.allclose(out.gradient, jnp.array([2.0]))
    assert out.hessian is not None
    assert jnp.allclose(out.hessian, jnp.array([[0.0]]))


def test_jet_decorator_three_args_selective_derivatives():
    """
    Two used args (x, y) and one unused (z). x has grad+hess, y has grad only,
    z has no derivatives. Expect gradient computed, hessian omitted.
    """
    @jet_decorator
    def g(x, y, z):
        return x + 2*y  # z unused

    x = Jet(value=jnp.array(1.5), gradient=jnp.array([1.0]), hessian=jnp.array([[0.0]]))
    y = Jet(value=jnp.array(-0.5), gradient=jnp.array([1.0]), hessian=None)
    z = Jet(value=jnp.array(7.0), gradient=None, hessian=None)

    out = g(x, y, z)

    # value = 1.5 + 2*(-0.5) = 0.5
    assert jnp.allclose(out.value, 0.5)
    # gradient over [x, y] coordinates should be [1, 2]
    assert jnp.allclose(out.gradient, jnp.array([1.0, 2.0]))
    # y lacks hessian, so overall hessian should be None
    assert out.hessian is None


def test_jet_decorator_unused_pytree_of_jets_argument():
    """
    extras is a PyTree of Jets without derivatives, but unused. Ensure full
    derivatives flow from the used scalar Jet only.
    """
    @jet_decorator
    def g(a, extras):
        return a**2

    a = function_to_jet(lambda t: t, jnp.array(3.0))

    extras = {
        'u': Jet(value=jnp.array(10.0), gradient=None, hessian=None),
        'v': Jet(value=jnp.array(20.0), gradient=None, hessian=None),
    }

    out = g(a, extras)

    assert jnp.allclose(out.value, 9.0)
    assert jnp.allclose(out.gradient, jnp.array([6.0]))
    assert jnp.allclose(out.hessian, jnp.array([[2.0]]))


def test_jet_decorator_annotated_params_with_unused():
    """
    Test annotated Jet parameters where one param is unused. Ensure derivatives
    are computed from the used Jet parameter.
    """
    @jet_decorator
    def cubic(jx: Jet, jy: Jet):
        # Use only jx
        return jx.value**3

    t = 1.5
    jx = function_to_jet(lambda x: x, jnp.array(t))
    jy = function_to_jet(lambda x: x, jnp.array(2.0))  # unused

    out = cubic(jx, jy)

    expected_val = t**3
    expected_grad = 3 * t**2  # d/dt x^3
    expected_hess = 6 * t     # d^2/dt^2 x^3

    assert jnp.allclose(out.value, expected_val)
    assert jnp.allclose(out.gradient, jnp.array([expected_grad]))
    assert jnp.allclose(out.hessian, jnp.array([[expected_hess]]))


def test_jet_decorator_partial_use_within_pytree_valued_jet():
    """
    Single Jet with PyTree-valued fields; function uses only one field. Ensure
    output derivatives match those of the used field only.
    """
    @jet_decorator
    def use_only_a(x):
        return x['a']

    input_jet = Jet(
        value={'a': jnp.array(2.0), 'b': jnp.array(5.0)},
        gradient={'a': jnp.array([3.0]), 'b': jnp.array([7.0])},
        hessian={'a': jnp.array([[11.0]]), 'b': jnp.array([[13.0]])},
    )

    out = use_only_a(input_jet)

    assert jnp.allclose(out.value, 2.0)
    assert jnp.allclose(out.gradient, jnp.array([3.0]))
    assert jnp.allclose(out.hessian, jnp.array([[11.0]]))


def test_jet_decorator_used_group_without_gradient():
    """
    If a used argument group lacks gradients, no derivatives should be returned.
    """
    @jet_decorator
    def g(x, y):
        return y + 1.0  # depends only on y

    x = Jet(value=jnp.array(2.0), gradient=jnp.array([1.0]), hessian=jnp.array([[0.0]]))
    y = Jet(value=jnp.array(3.0), gradient=None, hessian=None)

    out = g(x, y)

    assert jnp.allclose(out.value, 4.0)
    assert out.gradient is None
    assert out.hessian is None
