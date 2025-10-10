import jax.numpy as jnp
import pytest
import jax
from local_coordinates.jet import Jet
from local_coordinates.basis import BasisVectors
from local_coordinates.connection import Connection
from local_coordinates.frame import Frame

def test_frame_instantiation_and_properties():
    # 1. Create dummy objects with compatible shapes
    p = jnp.array([1.0, 2.0])

    # Components Jet for BasisVectors
    # A 2D (N=2) space with 2 basis vectors (D=2)
    basis_jet = Jet(
        value=jnp.eye(2),  # Basis vectors
        gradient=jnp.zeros((2, 2, 2)),  # Second derivatives
        hessian=jnp.zeros((2, 2, 2, 2)) # Third derivatives
    )

    basis = BasisVectors(p=p, components=basis_jet)

    # Christoffel symbols Jet for Connection
    christoffel_jet = Jet(
        value=jnp.zeros((2, 2, 2)),  # Christoffel symbols
        gradient=jnp.zeros((2, 2, 2, 2)),
        hessian=None
    )

    connection = Connection(basis=basis, christoffel_symbols=christoffel_jet)

    # 2. Successfully create a Frame instance
    frame = Frame(basis=basis, connection=connection)

    # 3. Verify that the properties correctly delegate
    assert frame.basis is basis
    assert frame.connection is connection
    assert jnp.array_equal(frame.p, p)
    assert frame.components is basis_jet

def test_frame_basis_consistency_check():
    # Create two different BasisVectors objects
    p = jnp.array([1.0, 2.0])
    basis_jet1 = Jet(
        value=jnp.eye(2),
        gradient=jnp.zeros((2, 2, 2)),
        hessian=jnp.zeros((2, 2, 2, 2))
    )
    basis1 = BasisVectors(p=p, components=basis_jet1)

    basis_jet2 = Jet(
        value=jnp.eye(2) * 2, # Different values
        gradient=jnp.ones((2, 2, 2)),
        hessian=jnp.ones((2, 2, 2, 2))
    )
    basis2 = BasisVectors(p=p, components=basis_jet2)

    # Create a connection using the second basis
    christoffel_jet = Jet(
        value=jnp.zeros((2, 2, 2)),
        gradient=jnp.zeros((2, 2, 2, 2)),
        hessian=None
    )
    connection = Connection(basis=basis2, christoffel_symbols=christoffel_jet)

    # Attempting to create a Frame with mismatched bases should raise an error
    with pytest.raises(ValueError, match="must be the same object"):
        Frame(basis=basis1, connection=connection)

def test_frame_jit_compatibility():
    # Create a valid Frame
    p = jnp.array([1.0, 2.0])
    basis_jet = Jet(
        value=jnp.eye(2),
        gradient=jnp.zeros((2, 2, 2)),
        hessian=jnp.zeros((2, 2, 2, 2))
    )
    basis = BasisVectors(p=p, components=basis_jet)
    christoffel_jet = Jet(
        value=jnp.zeros((2, 2, 2)),
        gradient=jnp.zeros((2, 2, 2, 2)),
        hessian=None
    )
    connection = Connection(basis=basis, christoffel_symbols=christoffel_jet)
    frame = Frame(basis=basis, connection=connection)

    # Define a simple function that uses the Frame
    @jax.jit
    def get_point_from_frame(f: Frame) -> jnp.ndarray:
        return f.p

    # JIT-compile and run the function
    jit_p = get_point_from_frame(frame)

    assert jnp.array_equal(jit_p, p)
