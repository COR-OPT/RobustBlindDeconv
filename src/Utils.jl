#!/usr/bin/env julia

"""
A module providing utilities for an efficient implementation of robust
blind deconvolution algorithms. The variant of the spectral initialization
here is adapted from [1].

[1]: John Duchi, Feng Ruan. "Solving (most) of a set of quadratic equalities:
Composite optimization for robust phase retrieval." arXiv:1705.02356.
"""
module Utils

	using LinearAlgebra
	using Random
	using Statistics

	export binary_matrix, correlation, direction_dist, frobdist_norm
	export robust_loss, signpat, randn_complex
	export A_safe!, AT_safe!, specinit_op


	"""
		binary_matrix(m, d)

	Returns an ``m \\times d`` matrix with entries sampled uniformly at random
	from the set ``\\{-1, 0, 1\\}``.
	"""
	binary_matrix(m, d) = rand([-1, 0, 1], m, d)


	"""
		signpat(n, k)

	Creates a ``n \\times k`` random sign pattern matrix.
	"""
	signpat(n, k) = rand([-1.0, 1.0], n, k)

	"""
		randn_complex(d...)

	Creates an array of i.i.d complex standard normals.
	Specifically, every element is ``X + iY``, where ``X, Y \\sim
	\\mathcal{N}\\left(0, \\frac{1}{\\sqrt{2}}\\right)``.
	"""
	randn_complex(d...) = (1 / sqrt(2)) * complex.(randn(d...), randn(d...))

	# Various metrics to check solutions
	correlation(x, y) = normalize(x)' * normalize(y)

	"""
		direction_dist(w::Array, x::Array, wbar::Array, xbar::Array)

	Computes the distance between the directions of ``w x^\\top`` and
	``\\bar{w} \\bar{x}^\\top``.
	"""
	direction_dist(w::Array, x::Array, wbar::Array, xbar::Array) = begin
		xn, wn, xbn, wbn = normalize.([x, w, xbar, wbar])
		norm(wn * xn' - wbn * xbn')
	end

	"""
		frobdist_norm(w::Array, x::Array, wbar::Array, xbar::Array)

	Computes the normalized frobenius distance, defined as follows:

	``
	\\left\\| w x^\\top - \\bar{w} \\bar{x}^\\top \\right\\_F /
	\\left\\| \\bar{w} \\bar{x}^\\top \\right\\_F
	``
	"""
	frobdist_norm(w::Array, x::Array, wbar::Array, xbar::Array) = begin
		return norm(w * x' - wbar * xbar') / norm(wbar * xbar')
	end


	"""
		fwht_no_scale!(y)

	Performs the fast Walsh-Hadamard transform, without scaling the resulting
	vector by ``1 / \\sqrt{n}``. The vector is modified in place.
	"""
	function fwht_no_scale!(y::Array{Float64, 1})
		n = length(y)
		o = trailing_zeros(n) - 1
		m = i = n >> 1
		while i > 0
			for j = 0:m-1
				s = j + j >> o << o + 1
				t = s + i
				p, q = y[s], y[t]
				y[s], y[t] = p + q, p - q
			end
			i >>= 1; o -= 1
		end
	end


	"""
		fwht_no_scale!(y, off, N)

	Performs the fast Walsh-Hadamard transform, without scaling the resulting
	vector by ``1 / \\sqrt{n}``, starting from index `off + 1` and ending at
	`off + N`. The array is modified in place.
	"""
	function fwht_no_scale!(y::Array{Float64, 1}, off, n)
		ispow2(n) || throw(ArgumentError("Non power-of-2 HT length"))
		o = trailing_zeros(n) - 1
		m = i = n >> 1
		while i > 0
			for j = 0:m-1
				s = j + j >> o << o + 1
				t = s + i
				p, q = y[s+off], y[t+off]
				y[s+off], y[t+off] = p + q, p - q
			end
			i >>= 1; o -= 1
		end
	end


	"""
		fwht!(y)

	Performs the fast Walsh-Hadamard transform, scaling the resulting vector
	appropriately.
	"""
	function fwht!(y::Array{Float64, 1})
		n = length(y)
		ispow2(n) || throw(ArgumentError("Non power-of-2 HT length"))
		fwht_no_scale!(y)
		rmul!(y, 1.0 / sqrt(n))
	end


	"""
		fwht!(y, x)

	Performs the fast Walsh-Hadamard transform on `x`, scaling the resulting
	vector appropriately. The result is stored on `y`.
	"""
	function fwht!(y::Array{Float64, 1}, x::Array{Float64, 1})
		n = length(x)
		ispow2(n) || throw(ArgumentError("Non power-of-2 HT length"))
		copyto!(y, x)
		fwht_no_scale!(y)
		rmul!(y, 1.0 / sqrt(n))
	end


	"""
		A_safe!(r, x, s, k, n)

	A memory - safe implementation of ``r = Ax`` using the fast Walsh-Hadamard
	transform. `n` stands for the length of the input signal `x`, `k` is the
	number of Hadamard sensing matrices, and `s` is a collection of random sign
	vectors. The result is modified in-place.

	Dimensions:
		`x` : `n`
		`s` : `n x k`
		`r` : `k x n`
	"""
	function A_safe!(r::Array{Float64, 1}, x::Array{Float64, 1}, s, k, n)
		for i = 1:k
			stri, endi = (i - 1) * n + 1, i * n
			broadcast!(*, view(r, stri:endi), x, s[:, i])
			fwht_no_scale!(r, stri - 1, n)
		end
		# perform scaling at the end
		rmul!(r, 1 / sqrt(n))
	end


	"""
		AT_safe!(r, w, s, k, n, temp)

	A memory-safe implementation of ``r = A^\\top w`` using the Walsh-Hadamard
	transform. `n` stands for the length of the output signal `r`, `k` is the
	number of Hadamard sensing matrices, and `s` is a collection of random sign
	vectors. The result is modified in place.

	Dimensions:
		`w` : `k x n`
		`s` : `n x k`
		`r` : `n`
	"""
	function AT_safe!(r::Array{Float64, 1}, w::Array{Float64, 1}, s, k, n, temp)
		fill!(r, 0.0)
		for i = 1:k
			# copy w[(i - 1) * n + 1 : i * n] to temp
			copyto!(temp, 1, w, (i - 1) * n + 1, n)
			fwht!(temp)
			broadcast!(*, temp, temp, s[:, i])
			broadcast!(+, r, r, temp)
		end
	end


	"""
		Adet_safe!(r, x)

	Computes the mapping ``x \\mapsto A x`` with ``A`` being a partial Hadamard
	matrix formed by keeping the first `length(x)` columns of ``H_m``. The
	result is stored in `r`.
	"""
	function Adet_safe!(r, x)
		fwht!(r, vcat(x, fill(0., length(r) - length(x))))
	end


	"""
		ATdet_safe!(r, w, d, temp)

	Computes the mapping ``w \\mapsto A^\\top w``, with ``A`` being a partial
	Hadamard matrix formed by keeping the first ``d`` columns of ``H_m``. `temp`
	is a placeholder for temporary results, and must satisfy
	`length(temp) == length(w)`.
	"""
	function ATdet_safe!(r, w, d, temp)
		fwht!(temp, w); copyto!(r, 1, temp, 1, d)
	end


	"""
		dft_partial!(r, x, d)

	Computes the partial DFT of `x` after keeping only the first `length(x)`
	columns of the DFT matrix. The result is stored in `r`.
	"""
	function dft_partial!(r, x)
		# pad x with zeros and compute fft
		copyto!(r, fft(vcat(x, fill(0., length(r) - length(x))), 1))
	end


	"""
		dftT_partial!(r, w, d)

	Computes the result of ``r = (F R_{1:d})^H w``, where ``F`` is the
	DFT matrix and ``R_{1:d}`` is its restriction to the first `d` columns.
	"""
	function dftT_partial!(r, w, d)
		copyto!(r, bfft(w, 1)[1:d])
	end


	"""
		specinit_op(x, ind, k, A, AT, temp; itm = 200, tol = 1e-6)

	Performs a robust variant of the Duchi initialization for the vector
	direction. `x` is the initial guess, `ind` is a 0-1 vector which denotes
	whether indices belong to the set of included measurements, `A` is a
	callable mapping ``x \\to Ax`` and `AT` is a callable that corresponds
	to ``x \\to A^\\top x``. `k` is the number of sign pattern matrices used
	in the case of random Hadamard matrix ensembles.

	Returns an estimate of the minimal eigenvector of the operator
	``A_I A_I^\\top``, where ``A_I`` is the subset of the measurement
	matrix that satisfies the median condition.
	"""
	function specinit_op!(x, ind, k, A, AT; itm = 200, tol = 1e-6)
		# initialize temporary vectors.
		temp, r, d = zero(x), fill(zero(eltype(x)), length(ind)), length(x)
		normalize!(x)
		x⁻  = copy(x)
		err, noi = 1.0, 0
		while err ≥ tol
			A(r, x)
			broadcast!(*, r, r, ind)
			AT(x, r, temp)
			rmul!(x, -1.0)
			BLAS.axpy!(k, x⁻, x)
			normalize!(x)
			err = norm(x - x⁻)
			noi += 1
			noi ≥ itm && break
			copyto!(x⁻, x)
		end
	end


	"""
		fmin_alin(f, gradf, eta, xcurr, iters; eps=1e-6, beta=0.5)

	Minimizes a (sub)differentiable function using linesearch at every step.
	`f` and `gradf` are callables implementing the function and one of its
	subgradients when evaluated at `xcurr`. `iters` is the number of iterations
	to run for. `eta` is the initial step size and `beta` is the factor to
	decrease by at every iteration.
	"""
	function fmin_alin(f, gradf, eta, xcurr, iters; eps=1e-6, beta=0.5)
		costo, costn = f(xcurr), f(xcurr)
		for i = 1:iters
			lr = approx_linesearch(f, gradf, eta, xcurr, beta)
			xcurr = xcurr - lr * gradf(xcurr)
			costn = f(xcurr)
			if (costo - costn) <= eps
				return xcurr
			end
			costo = costn
		end
		return xcurr
	end


	"""
		approx_linesearch(f, gradf, eta, xcurr, beta)

	Performs an approximate linesearch, given a function `f` and its gradient
	implementation `gradf`, an initial rate `eta` and the current iterate
	`xcurr`. At every step, decrease the rate by a factor `beta`.
	"""
	function approx_linesearch(f, gradf, eta, xcurr, beta)
		mincost = f(xcurr - eta * gradf(xcurr))
		mineta = eta
		while eta > 1e-6
			eta *= beta
			cnew = f(xcurr - eta * gradf(xcurr))
			if cnew < mincost
				mincost = cnew
				mineta = eta
			end
		end
		return mineta
	end


	"""
		frob_opt(w0, x0, w, x)

	A memory-efficient version of computing the Frobenius norm of
	``w_0 x_0^\\top - w x^\\top``.
	"""
	function frob_opt(w0, x0, w, x)
		# use max(0, x) to avoid underflow
		return sqrt(
			max(0, (norm(w0) * norm(x0))^2 + (norm(w) * norm(x))^2
				- 2 * (w0' * w * x0' * x)))
	end

end
