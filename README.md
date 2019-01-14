# RobustBlindDeconv
This is a reference implementation of the subgradient and prox-linear methods
for robust blind deconvolution presented in \[[1](https://arxiv.org/abs/1901.01624)\].
It has been tested in Julia v1.0.

## Requirements

In the current version, the only external requirement for the implementation
under `src/` is `Arpack`. The dependencies for the experiments are:

1. `ArgParse`
2. `CSV`
3. `DataFrames`
4. `Images`  (for `test_images.jl, test_mnist.jl`)
5. `ImageMagick`  (for `test_images.jl, test_mnist.jl`)
6. `MLDatasets`  (only for `test_mnist.jl`)
7. `ColorTypes`  (required by `ImageUtils.jl`)

## Quick tour

The files under `src/` contain the implementation of both methods, as well as
auxiliary utilities such as quickly setting up problem instances, controlling
noise, etc.

To setup a problem with `d_1 = 100, d_2 = 200` and `2000` Gaussian measurements
with 25% of the measurements corrupted by additive white noise, it suffices to
write:

```julia
include("src/BlindDeconv.jl");

w, x = randn(100), randn(200)  # true signals
prob = BlindDeconv.gen_problem(2000, w, x, BlindDeconv.gaussian, BlindDeconv.gaussian, 0.25);
```

There is a wide selection of measurement matrices. An overview is available via

```julia
help?> BlindDeconv.MatType
```

As a quick example, in order to use a matrix with orthonormal columns as the
"left" measurement matrix instead of a Gaussian matrix, one can write:

```julia
prob = BlindDeconv.gen_problem(2000, w, x, BlindDeconv.ortho, BlindDeconv.gaussian, 0.25);
```

Then, one can use either of the two methods to solve the above problem. To use
the subgradient method with decaying step size, decaying by a factor of `q = 0.98`
at each iteration, for 500 iterations, we can write:

```julia
wf, xf, ds = BlindDeconv.subgradient_method(prob, 1.0, 0.98, 500);
println("Subgradient method error: ", ds[end])
```

If, instead, it was known that the problem instance is noiseless, i.e. all
measurements are exact, the Polyak step size performs much better empirically,
and is activated by setting the relevant keyword argument:

```julia
wf, xf, ds = BlindDeconv.subgradient_method(prob, 1.0, 0.0, 500, polyak_step=true)
```

For the prox-linear method with quadratic penalty parameter `a = 1.0` and 15
iterations, write:

```julia
wf, xf, ds = BlindDeconv.proximal_method(prob, 1.0, 15);
println("Proximal method error: ", ds[end])
```

### Running the experiments
The experiments are available under the `experiments/` folder. The choice of
parameters for each is documented in \[1\]. Most of the experiments output the
results in a `.csv` file, under an appropriate name. There is no code for
reproducing figures included, since most of them were written in Latex / Tikz.

### References

\[1\]: Vasileios Charisopoulos, Damek Davis, Mateo Díaz and Dmitriy Drusvyatskiy. "Composite optimization for robust blind deconvolution". arXiv preprint abs/1901.01624.
