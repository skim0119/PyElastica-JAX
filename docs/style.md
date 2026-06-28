# Style Conventions

- For this repo, try to use `jax.numpy` instead of `numpy`.

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
    type Vector = jax.Array
    type Scalar = jax.Scalar
    ```
    - Generic Class
    ```python
    class MyClass[T]:
        def __init__(self, value: T):
    ```

- Function is named `jax_` prefix when it is meant to be pure JAX function.
