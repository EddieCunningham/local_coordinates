import jax.numpy as jnp
import pytest
from local_coordinates.basis import BasisVectors
from local_coordinates.tensor import Tensor, TensorType, change_coordinates
from local_coordinates.jet import Jet
import jax

@pytest.fixture
def a_basis():
    jet = Jet(value=jnp.array([0., 0.]), gradient=jnp.eye(2), hessian=None)
    return BasisVectors(_jet=jet)

@pytest.fixture
def another_basis():
    angle = jnp.pi / 4
    rot = jnp.array([
        [jnp.cos(angle), -jnp.sin(angle)],
        [jnp.sin(angle), jnp.cos(angle)]
    ])
    jet = Jet(value=jnp.array([1., 1.]), gradient=rot, hessian=None)
    return BasisVectors(_jet=jet)

def test_tensortype_methods():
    tt = TensorType(k=2, l=3)
    assert tt.total_dims() == 5
    assert tt.k_names == ['k0', 'k1']
    assert tt.l_names == ['l0', 'l1', 'l2']
    assert tt.get_coordinate_indices() == 'k0 k1 l0 l1 l2'

def test_tensor_creation(a_basis):
    tt = TensorType(k=1, l=1)
    components = jnp.eye(2)
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=components)
    assert tensor.tensor_type == tt
    assert tensor.basis == a_basis
    assert jnp.array_equal(tensor.components, components)

def test_tensor_batch_size(a_basis):
    # No batch
    tt = TensorType(k=1, l=1)
    components = jnp.eye(2)
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=components)
    assert tensor.batch_size is None

    # 1D batch
    components_batch = jnp.stack([jnp.eye(2)] * 5)
    tensor_batch = Tensor(tensor_type=tt, basis=a_basis, components=components_batch)
    assert tensor_batch.batch_size == 5

    # Multi-D batch
    components_multibatch = jnp.stack([components_batch] * 3)
    tensor_multibatch = Tensor(tensor_type=tt, basis=a_basis, components=components_multibatch)
    assert tensor_multibatch.batch_size == (3, 5)


def test_change_coordinates_vector(a_basis, another_basis):
    # Vector is a contravariant tensor of rank 1
    tt = TensorType(k=0, l=1)
    vec = jnp.array([1., 0.])
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=vec)

    new_tensor = change_coordinates(tensor, another_basis)

    # Manual transformation
    T = jnp.linalg.inv(another_basis.basis_vectors)
    manual_new_vec = T @ vec

    assert new_tensor.basis == another_basis
    assert jnp.allclose(new_tensor.components, manual_new_vec)


def test_change_coordinates_covector(a_basis, another_basis):
    # Covector is a covariant tensor of rank 1
    tt = TensorType(k=1, l=0)
    covec = jnp.array([1., 0.])
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=covec)

    new_tensor = change_coordinates(tensor, another_basis)

    # Manual transformation
    T = jnp.linalg.inv(another_basis.basis_vectors)
    Tinv = jnp.linalg.inv(T)
    manual_new_covec = Tinv @ covec

    assert new_tensor.basis == another_basis
    assert jnp.allclose(new_tensor.components, manual_new_covec)

def test_change_coordinates_metric_tensor(a_basis, another_basis):
    # Metric tensor is a covariant tensor of rank 2
    tt = TensorType(k=2, l=0)
    metric = jnp.eye(2)
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=metric)

    new_tensor = change_coordinates(tensor, another_basis)

    # Manual transformation
    T = jnp.linalg.inv(another_basis.basis_vectors)
    Tinv = jnp.linalg.inv(T)
    manual_new_metric = Tinv @ metric @ Tinv.T

    assert new_tensor.basis == another_basis
    assert jnp.allclose(new_tensor.components, manual_new_metric)

def test_change_coordinates_tensor_0_2(a_basis, another_basis):
    tt = TensorType(k=0, l=2)
    components = jnp.array([[1., 2.], [3., 4.]])
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=components)

    new_tensor = change_coordinates(tensor, another_basis)

    # Manual transformation
    T = jnp.linalg.inv(another_basis.basis_vectors)
    manual_new_components = T @ components @ T.T

    assert new_tensor.basis == another_basis
    assert jnp.allclose(new_tensor.components, manual_new_components)

def test_change_coordinates_tensor_2_2(a_basis, another_basis):
    tt = TensorType(k=2, l=2)
    # Create a random 4D array for components
    key = jax.random.PRNGKey(0)
    components = jax.random.normal(key, (2, 2, 2, 2))
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=components)

    new_tensor = change_coordinates(tensor, another_basis)

    # Manual transformation
    T = jnp.linalg.inv(another_basis.basis_vectors)
    Tinv = jnp.linalg.inv(T)

    manual_new_components = jnp.einsum(
        'ijkl,mi,nj,ok,pl->mnop',
        components, Tinv, Tinv, T, T
    )

    assert new_tensor.basis == another_basis
    assert jnp.allclose(new_tensor.components, manual_new_components)
