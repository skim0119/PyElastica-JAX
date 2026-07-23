# Style Conventions

- For this repo, try to use `jax.numpy` instead of `numpy`.

- Prefer ``jaxtyping`` for array annotations. Import shared aliases from
  ``elastica_jax.typing`` (``Array``, ``ArrayLike``, ``Float``, shaped Cosserat
  aliases). Use ``ArrayLike`` at host/device boundaries that accept
  ``float`` / ``numpy.ndarray`` / ``jax.Array``; use shaped ``Float[Array, "..."]``
  (or the Cosserat aliases) for pure device kernels. Prefer implicit promotion
  among python-float / numpy-float64 / jax-float64 — do not ``jnp.asarray`` every
  argument at the top of a kernel.
    - ex)
    ```python
    from elastica_jax.typing import Array, ArrayLike, Float, Nodes3

    def jax_apply_load(
        forces: Nodes3,
        load: ArrayLike,
    ) -> Nodes3:
        return forces + load[:, None]
    ```

- `configure_<name>` is a naming convention for factory function that configure an object with varying attributes. Always use keyword arguments to make front-API explicitly readable without mistakes.
    - ex)
    ```python
    def configure_func(
        *,  # keyword-only argument
        bias: float
    ) -> Type[ConfiguredFunc]:
        def func(A: jax.Array, x: jax.Array) -> jax.Array:
            return A @ x + bias
        return func

    f = configure_func(bias=1.0)  # bias explicitly named
    f = configure_func(1.0)  # not allowed. prone to mistake.
    ```

- Use new generic and type alias style:
    - Type Alias
    ```python
    type Vector = Float[Array, "3"]
    type Scalar = Float[Array, ""]
    ```
    - Generic Class
    ```python
    class MyClass[T]:
        def __init__(self, value: T):
    ```

- Function is named `jax_` prefix when it is meant to be pure JAX function.

- Operator constructors receive an injected ``_system`` kwarg at finalize time.
  Name it only when the constructor reads it (type as ``eaj.RodSystemLike``).
  If unused, absorb with ``**kwargs`` — do not declare ``_system`` just to
  ``del`` it.
    - ex)
    ```python
    class GravityForcesJax(eaj.NoOpsJax):
        def __init__(self, *, acc_gravity: np.ndarray, **kwargs: object) -> None:
            self.acc_gravity = acc_gravity

    class OneEndFixedJax(eaj.NoOpsJax):
        def __init__(self, *, _system: eaj.RodSystemLike) -> None:
            self.fixed_position = _system.position_collection[..., 0].copy()
    ```
