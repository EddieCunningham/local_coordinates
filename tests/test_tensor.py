import jax.numpy as jnp
import pytest
from local_coordinates.basis import BasisVectors, get_basis_transform
from local_coordinates.tensor import Tensor, TensorType, change_coordinates
from local_coordinates.jet import Jet, function_to_jet
import jax
import jax.random as random

@pytest.fixture
def a_basis():
    p = jnp.array([0., 0.])
    basis_vectors = jnp.eye(2)
    components_jet = Jet(value=basis_vectors, gradient=None, hessian=None, dim=2)
    return BasisVectors(p=p, components=components_jet)

@pytest.fixture
def another_basis():
    p = jnp.array([1., 1.])
    angle = jnp.pi / 4
    rot = jnp.array([
        [jnp.cos(angle), -jnp.sin(angle)],
        [jnp.sin(angle), jnp.cos(angle)]
    ])
    components_jet = Jet(value=rot, gradient=None, hessian=None, dim=2)
    return BasisVectors(p=p, components=components_jet)

def test_tensortype_methods():
    tt = TensorType(k=2, l=3)
    assert tt.total_dims() == 5
    assert tt.k_names == ['k0', 'k1']
    assert tt.l_names == ['l0', 'l1', 'l2']
    assert tt.get_coordinate_indices() == 'k0 k1 l0 l1 l2'

def test_tensor_creation(a_basis):
    tt = TensorType(k=1, l=1)
    components = jnp.eye(2)
    key = jax.random.PRNGKey(0)
    gradient = jax.random.normal(key, (2, 2, 2))
    hessian = jax.random.normal(key, (2, 2, 2, 2))
    components_jet = Jet(value=components, gradient=gradient, hessian=hessian)
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=components_jet)
    assert tensor.tensor_type == tt
    assert tensor.basis == a_basis
    assert jnp.array_equal(tensor.components.value, components)
    assert jnp.array_equal(tensor.components.gradient, gradient)
    assert jnp.array_equal(tensor.components.hessian, hessian)

def test_tensor_batch_size(a_basis):
    # No batch
    tt = TensorType(k=1, l=1)
    components = jnp.eye(2)
    key = jax.random.PRNGKey(1)
    gradient = jax.random.normal(key, (2, 2, 2))
    hessian = jax.random.normal(key, (2, 2, 2, 2))
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=Jet(value=components, gradient=gradient, hessian=hessian))
    assert tensor.batch_size is None

    # 1D batch
    components_batch = jnp.stack([jnp.eye(2)] * 5)
    gradient_batch = jax.random.normal(key, (5, 2, 2, 2))
    hessian_batch = jax.random.normal(key, (5, 2, 2, 2, 2))
    tensor_batch = Tensor(tensor_type=tt, basis=a_basis, components=Jet(value=components_batch, gradient=gradient_batch, hessian=hessian_batch))
    assert tensor_batch.batch_size == 5

    # Multi-D batch
    components_multibatch = jnp.stack([components_batch] * 3)
    gradient_multibatch = jax.random.normal(key, (3, 5, 2, 2, 2))
    hessian_multibatch = jax.random.normal(key, (3, 5, 2, 2, 2, 2))
    tensor_multibatch = Tensor(tensor_type=tt, basis=a_basis, components=Jet(value=components_multibatch, gradient=gradient_multibatch, hessian=hessian_multibatch))
    assert tensor_multibatch.batch_size == (3, 5)


def test_change_coordinates_vector(a_basis, another_basis):
    # Vector is a contravariant tensor of rank 1
    tt = TensorType(k=0, l=1)
    vec = jnp.array([1., 0.])
    key = jax.random.PRNGKey(2)
    gradient = jax.random.normal(key, (2, 2))
    hessian = jax.random.normal(key, (2, 2, 2))
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=Jet(value=vec, gradient=gradient, hessian=hessian))

    new_tensor = change_coordinates(tensor, another_basis)

    # Manual transformation
    T = get_basis_transform(a_basis, another_basis)
    manual_new_vec = jnp.einsum('l,ml->m', vec, T.value)
    manual_new_grad = jnp.einsum('lr,ml->mr', gradient, T.value)
    manual_new_hess = jnp.einsum('lrs,ml->mrs', hessian, T.value)


    assert new_tensor.basis == another_basis
    assert jnp.allclose(new_tensor.components.value, manual_new_vec)
    assert jnp.allclose(new_tensor.components.gradient, manual_new_grad)
    assert jnp.allclose(new_tensor.components.hessian, manual_new_hess)


def test_change_coordinates_covector(a_basis, another_basis):
    # Covector is a covariant tensor of rank 1
    tt = TensorType(k=1, l=0)
    covec = jnp.array([1., 0.])
    key = jax.random.PRNGKey(3)
    gradient = jax.random.normal(key, (2, 2))
    hessian = jax.random.normal(key, (2, 2, 2))
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=Jet(value=covec, gradient=gradient, hessian=hessian))

    new_tensor = change_coordinates(tensor, another_basis)

    # Manual transformation
    T = get_basis_transform(a_basis, another_basis)
    Tinv = jnp.linalg.inv(T.value)
    manual_new_covec = jnp.einsum('k,mk->m', covec, Tinv)
    manual_new_grad = jnp.einsum('kr,mk->mr', gradient, Tinv)
    manual_new_hess = jnp.einsum('krs,mk->mrs', hessian, Tinv)

    assert new_tensor.basis == another_basis
    assert jnp.allclose(new_tensor.components.value, manual_new_covec)
    assert jnp.allclose(new_tensor.components.gradient, manual_new_grad)
    assert jnp.allclose(new_tensor.components.hessian, manual_new_hess)

def test_change_coordinates_metric_tensor(a_basis, another_basis):
    # Metric tensor is a covariant tensor of rank 2
    tt = TensorType(k=2, l=0)
    metric = jnp.eye(2)
    key = jax.random.PRNGKey(4)
    gradient = jax.random.normal(key, (2, 2, 2))
    hessian = jax.random.normal(key, (2, 2, 2, 2))
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=Jet(value=metric, gradient=gradient, hessian=hessian))

    new_tensor = change_coordinates(tensor, another_basis)

    # Manual transformation
    T = get_basis_transform(a_basis, another_basis)
    Tinv = jnp.linalg.inv(T.value)
    manual_new_metric = jnp.einsum('kl,mk,nl->mn', metric, Tinv, Tinv)
    manual_new_grad = jnp.einsum('klr,mk,nl->mnr', gradient, Tinv, Tinv)
    manual_new_hess = jnp.einsum('klrs,mk,nl->mnrs', hessian, Tinv, Tinv)

    assert new_tensor.basis == another_basis
    assert jnp.allclose(new_tensor.components.value, manual_new_metric)
    assert jnp.allclose(new_tensor.components.gradient, manual_new_grad)
    assert jnp.allclose(new_tensor.components.hessian, manual_new_hess)

def test_change_coordinates_tensor_0_2(a_basis, another_basis):
    tt = TensorType(k=0, l=2)
    components = jnp.array([[1., 2.], [3., 4.]])
    key = jax.random.PRNGKey(5)
    gradient = jax.random.normal(key, (2, 2, 2))
    hessian = jax.random.normal(key, (2, 2, 2, 2))
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=Jet(value=components, gradient=gradient, hessian=hessian))

    new_tensor = change_coordinates(tensor, another_basis)

    # Manual transformation
    T = get_basis_transform(a_basis, another_basis)
    manual_new_components = jnp.einsum('ij,mi,nj->mn', components, T.value, T.value)
    manual_new_grad = jnp.einsum('ijr,mi,nj->mnr', gradient, T.value, T.value)
    manual_new_hess = jnp.einsum('ijrs,mi,nj->mnrs', hessian, T.value, T.value)

    assert new_tensor.basis == another_basis
    assert jnp.allclose(new_tensor.components.value, manual_new_components)
    assert jnp.allclose(new_tensor.components.gradient, manual_new_grad)
    assert jnp.allclose(new_tensor.components.hessian, manual_new_hess)

def test_change_coordinates_tensor_2_2(a_basis, another_basis):
    tt = TensorType(k=2, l=2)
    # Create a random 4D array for components
    key = jax.random.PRNGKey(6)
    components = jax.random.normal(key, (2, 2, 2, 2))
    gradient = jax.random.normal(key, (2, 2, 2, 2, 2))
    hessian = jax.random.normal(key, (2, 2, 2, 2, 2, 2))
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=Jet(value=components, gradient=gradient, hessian=hessian))

    new_tensor = change_coordinates(tensor, another_basis)

    # Manual transformation
    T = get_basis_transform(a_basis, another_basis)
    Tinv = jnp.linalg.inv(T.value)

    manual_new_components = jnp.einsum(
        'ijkl,mi,nj,ok,pl->mnop',
        components, Tinv, Tinv, T.value, T.value
    )
    manual_new_grad = jnp.einsum(
        'ijklr,mi,nj,ok,pl->mnopr',
        gradient, Tinv, Tinv, T.value, T.value
    )
    manual_new_hess = jnp.einsum(
        'ijklrs,mi,nj,ok,pl->mnoprs',
        hessian, Tinv, Tinv, T.value, T.value
    )

    assert new_tensor.basis == another_basis
    assert jnp.allclose(new_tensor.components.value, manual_new_components)
    assert jnp.allclose(new_tensor.components.gradient, manual_new_grad)
    assert jnp.allclose(new_tensor.components.hessian, manual_new_hess)
