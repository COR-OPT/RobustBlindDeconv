# RobustBlindDeconv
This is a reference implementation of the subgradient and prox-linear methods
for robust blind deconvolution presented in \[[1](https://arxiv.org/abs/1901.01624)\].
It has been tested in Julia v1.0.

## Requirements

In the current version, the only external requirement is `Arpack`.

## Quick tour

To setup a problem with `d_1 = 100, d_2 = 200` and `2000` Gaussian measurements
with 25% of the measurements corrupted by additive white noise, it suffices to
write:

```julia
include("src/BlindDeconv.jl");

w, x = randn(100), randn(200)  # true signals
prob = BlindDeconv.gen_problem(2000, w, x, BlindDeconv.gaussian, BlindDeconv.gaussian, 0.25);
```

Then, one can use either of the two methods to solve the above problem. To use
the subgradient method with decaying step size, decaying by a factor of `q = 0.98`
at each iteration, for 500 iterations, we can write:

```julia
wf, xf, ds = BlindDeconv.subgradient_method(prob, 1.0, 0.98, 500);
println("Subgradient method error: ", ds[end])
```

For the prox-linear method with quadratic penalty parameter `a = 1.0` and 15
iterations, write:

```julia
wf, xf, ds = BlindDeconv.proximal_method(prob, 1.0, 15);
println("Proximal method error: ", ds[end])
```

### References

\[1\]: Vasileios Charisopoulos, Damek Davis, Mateo Díaz and Dmitriy Drusvyatskiy. "Composite optimization for robust blind deconvolution". arXiv preprint abs/1901.01624.
