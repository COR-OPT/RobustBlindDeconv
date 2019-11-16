# RobustBlindDeconv
This is a reference implementation of the subgradient and prox-linear methods
for robust blind deconvolution presented in \[[1](https://arxiv.org/abs/1901.01624)\].
It has been tested in Julia v1.0.

## Requirements

In the current version, the only external requirements for the implementation
under `src/` are `Arpack` and `FFTW`. The dependencies for the experiments are:

1. `ArgParse`
2. `Images`
3. `ImageMagick`
4. `MLDatasets`
5. `ColorTypes`
6. `PyPlot`

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
prob = BlindDeconv.genProblem(2000, w, x, BlindDeconv.gaussian, BlindDeconv.gaussian, 0.25);
```

There is a wide selection of measurement matrices. An overview is available via

```julia
help?> BlindDeconv.MatType
```

As a quick example, in order to use a partial DFT matrix as the
"left" measurement matrix instead of a Gaussian matrix, one can write:

```julia
prob = BlindDeconv.genProblem(2000, w, x, BlindDeconv.pdft, BlindDeconv.complex_gaussian, 0.25);
```

In addition, the user can set up a problem where they fully control the
measurement matrices and the true signals, as the following snippet shows:

```julia
Lmat = randn(1000, 100); Rmat = randn(1000, 200)
w = randn(100); x = rand([-1.0, 0.0, 1.0], 200)
prob = BlindDeconv.genProblem(w, x, Lmat, Rmat, 0.25)
```

Then, one can use either of the two methods to solve the above problem. To use
the subgradient method with decaying step size, decaying by a factor of `q = 0.98`
at each iteration, for 500 iterations, we can write:

```julia
wf, xf, ds = BlindDeconv.subgradientMethod(prob, 1.0, 0.98, 500);
println("Subgradient method error: ", ds[end])
```

If, instead, it was known that the problem instance is noiseless, i.e. all
measurements are exact, the Polyak step size performs much better empirically,
and is activated by setting the relevant keyword argument:

```julia
wf, xf, ds = BlindDeconv.subgradientMethod(prob, 1.0, 0.0, 500, use_polyak=true)
```

For the prox-linear method with quadratic penalty parameter `a = 1.0` and 15
iterations, write:

```julia
wf, xf, ds = BlindDeconv.proximal_method(prob, 1.0, 15);
println("Proximal method error: ", ds[end])
```

### Running the experiments
In order to reproduce the results using the `PyPlot` visualization library in
Julia, it is recommended to use the `experiments/run_experiment.jl` script. Make sure your
working directory is `experiments/`, and run

```bash
$ julia run_experiment.jl --help
```

to see a list of available commands. Each command corresponds to a different
experiment presented in the paper, with options that can be listed via

```bash
$ julia run_experiment.jl <cmd> --help
```

where `cmd` is one of `decay_eval, synthetic_test, matvec_eval, color_img_test,
mnist_img_test`. For example, to reproduce the bottom row of Figure 10, use

```bash
$ cd experiments
$ julia run_experiment.jl mnist_img_test --idx_w 4096 --idx_x 8192 --iters 800
```

The figures appearing in the paper were prepared using tikz/LaTeX, which are
unfortunately not that easy to invoke using Julia's plotting solutions.

The choice of parameters for each experiment is documented in \[1\].

### References

\[1\]: Vasileios Charisopoulos, Damek Davis, Mateo Díaz and Dmitriy Drusvyatskiy. "Composite optimization for robust blind deconvolution". arXiv preprint abs/1901.01624.
