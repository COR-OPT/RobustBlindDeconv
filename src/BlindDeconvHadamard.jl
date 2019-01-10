"""
BlindDeconvHadamard.jl: Implementation of the robust algorithm for blind
deconvolution using Hadamard sensing matrices.
"""
module BlindDeconvHadamard

	include("Utils.jl")

	using LinearAlgebra
	using Random
	using SparseArrays
	using Statistics

	struct BCProb
		L
		R
		LT
		RT
		y :: Array{Float64, 1}
		w :: Array{Float64, 1}
		x :: Array{Float64, 1}
	end


	"""
		function genmat(m::Int, d::Int)

	It returns two callables that correspond to the operators ``x \\to Ax`` and
	``x \\to A^\\top x``.
	The first callable accepts two argument, the result (to be modified in
	place) and the input vector, while the second callable accepts an extra
	argument which has to be a temporary vector of length `d`.
	"""
	function genmat(m::Int, d::Int)
		ispow2(d) || throw(ArgumentError("d must be a power of 2!"))
		k = trunc(Int, m / d)
		s = Utils.signpat(d, k)
		# return a pair of callables
		Aop = ((r, x) -> Utils.A_safe!(r, x, s, k, d))  # Aop => r = Ax
		# ATop => r = A^T x
		ATop = ((r, x, tmp) -> Utils.AT_safe!(r, x, s, k, d, tmp))
		return Aop, ATop
	end

	"""
		robust_loss(r, w, x, y, L, R, Lw, Rx)

	Compute the robust loss of the blind deconvolution objective, given
	the iterates `w` and `x`.
	"""
	function robust_loss(r, w, x, y, L, R, Lw, Rx)
		# Lw = L * w, Rx = R * x
		L(Lw, w); R(Rx, x)
		# r := Lw .* Rx .- y
		broadcast!(*, r, Lw, Rx)
		broadcast!(-, r, r, y)
		rmul!(r, 1.0 / length(y))
		return norm(r, 1)
	end


	"""
		subgrad!(gw, gx, r, L, R, LT, RT, y, w, x, Lw, Rx, tmpw, tmpx)

	Compute the subgradient of the robust loss in-place, given the problem data
	`L`, `R`, `LT`, `RT`, `y`. `r` is a vector that is assumed to have size
	equal to `prob.n`. `w` and `x` are assumed to be the input signals. Two
	placeholders for temporary results, `Lw` and `Rx`, must be provided. Two more
	placeholders `tmpw, tmpx` for the adjoint operators, ``L^\\top, R^\\top``
	should be provided.

	The subgradient of the robust loss is given by
	``
	\\frac{1}{m} \\cdot
	\\sum_{i=1}^m
		\\mathrm{sign}(
			\\langle \\ell_i, w \\rangle \\cdot
			\\langle r_i, x \\rangle - y_i)
		\\cdot
	\\begin{pmatrix}
		\\langle r_i, x \\rangle \\ell_i^\\top \\
		\\langle \\ell_i, w \\rangle r_i^\\top
	\\end{pmatrix}
	``
	for w and x, respectively.

	Sizes: `Lw`, `Rx` should be `m`. `tmpw`, `tmpx` should have sizes `d1` and
	`d2`, respectively.
	"""
	function subgrad!(gw::Array{Float64, 1}, gx::Array{Float64, 1}, r, L, R,
					  LT, RT, y, w::Array{Float64, 1}, x::Array{Float64, 1},
					  Lw::Array{Float64, 1}, Rx::Array{Float64, 1},
					  tmpw::Array{Float64, 1}, tmpx::Array{Float64, 1})
		# compute residual and sign first
		# Lw = L * w, Rx = R * x
		L(Lw, w); R(Rx, x)
		# r := Lw .* Rx .- y
		broadcast!(*, r, Lw, Rx)
		broadcast!(-, r, r, y)
		rmul!(r, 1.0 / length(y))
		# get the subgradient
		map!(sign, r, r)  # sign(e_i)
		# Lw = L * w .* r; Rx = R * x .* r => coefficients for subgrads
		broadcast!(*, Rx, Rx, r); broadcast!(*, Lw, Lw, r)
		LT(gw, Rx, tmpw)
		RT(gx, Lw, tmpx)
	end


	function generate_measurements!(y, L, R, w, x, pfail, noise=nothing)
		L(y, w)
		tempr = zero(y); R(tempr, x)
		broadcast!(*, y, y, tempr)
		num_corr = trunc.(Int, pfail * length(y))
		if noise == nothing
			noise = randn(num_corr)
			y[randperm(length(y))[1:num_corr]] += noise
		else
			# replace the indices with the noise provided
			y[randperm(length(y))[1:num_corr]] = noise[1:num_corr]
		end
	end


	function add_noise!(y, noise, pfail, replace=false)
		m = length(y); ncorr = trunc(Int, pfail * m)
		(length(noise) == m) || throw(ArgumentError(
			"[noise] must have same length as [y]"))
		indices = randperm(m)[1:ncorr]
		if replace == true
			y[indices] = noise[indices]
		else
			y[indices] += noise[indices]
		end
	end


	"""
		direction_init!(wev, xev, y, L, R LT, RT; kw=1, kx=1)

	Computes an initial estimate for the directions of the solution using a
	variant of Duchi's spectral initialization method. `wev, xev` are initial
	estimates of the directions, which are modified in place. `y` is the set
	of measurements, `L, R` are either the measurement matrices or callables
	implementing ``x \\to Lx, \\; x \\to Rx`` respectively, and `LT, RT` are
	the corresponding callables implementing the adjoint operators.
	`kw` and `kx` are the number of sign pattern matrices that were used, in
	the case of random Hadamard ensembles. If left unspecified, they are set
	to `1` by default.
	"""
	function direction_init!(wev, xev, y, L, R, LT=nothing, RT=nothing;
							 kw=1, kx=1)
		med = median(abs.(y))
		ind = abs.(y) .<= med
		Utils.specinit_op!(wev, trunc.(Int, ind), kw, L, LT, itm=100)
		Utils.specinit_op!(xev, trunc.(Int, ind), kx, R, RT, itm=100)
	end


	function radius_init_subgrad!(wev, xev, Lw, Rx, y, L, R)
		m = length(y)
		L(Lw, wev); R(Rx, xev); broadcast!(*, Lw, Lw, Rx)
		f = (r -> (1 / m) * norm(r * Lw - y, 1))
		gradf = (r -> (1 / m) * sum(
			sign.(r * Lw - y) .* Lw))
		r_curr = sqrt(length(wev) * length(xev))
		iters = 5000
		for i = 1:iters
			g = gradf(r_curr)
			r_curr -= (1 / (i + 1)) * g / norm(g)
		end
		# pick best of +/- r
		r_curr = (f(r_curr) > f(-r_curr)) ? -r_curr : r_curr
		copyto!(wev, sign(r_curr) * sqrt(abs(r_curr)) * wev)
		copyto!(xev, sqrt(abs(r_curr)) * xev)
	end


	function project2ball!(x, gamma=1.0)
		if norm(x) > gamma
			rmul!(x, gamma / norm(x))
		end
	end


	"""
		init_method(w, x, L, R, LT, RT, y; init="direction")

	Runs the initialization for the blind deconvolution problem given the
	problem data.
	"""
	function init_method(w, x, L, R, LT, RT, y; init="direction")
		m, d1, d2 = length(y), length(w), length(x)
		wk = randn(d1); xk = randn(d2)
		Lw = fill(0.0, m); Rx = fill(0.0, m)
		if init == "spectral"
			direction_init!(wk, xk, y, L, R, LT, RT, kw=(m / d1), kx=(m / d2))
			radius_init_subgrad!(wk, xk, Lw, Rx, y, L, R)
		elseif init == "random"
			normalize!(wk, 2); normalize!(xk, 2)
			rmul!(wk, norm(w)); rmul!(xk, norm(x))
			# run one step of subgradient method
			wk, xk, _  = subgradient_method(w, x, L, R, LT, RT, y, Lw, Rx, 1.0,
											1.0, 1, polyak_step=true, wk=wk,
											xk=xk)
		elseif init == "direction"
			direction_init!(wk, xk, y, L, R, LT, RT, kw=(m / d1), kx=(m / d2))
			rmul!(wk, norm(w)); rmul!(xk, norm(x))
			L(Lw, wk); R(Rx, xk); broadcast!(*, Lw, Lw, Rx)
			cost_pos = (1 / m) * norm(Lw - y, 1);
			cost_neg = (1 / m) * norm(-Lw - y, 1)
			if cost_pos > cost_neg   # if cost is better, set wk negative
				copyto!(wk, -wk)
			end
		end
		# return estimates
		return wk, xk
	end

	function init_method(prob::BCProb; init="direction")
		return init_method(prob.w, prob.x, prob.L, prob.R, prob.LT, prob.RT,
						   prob.y, init=init)
	end


	"""
		subgradient_method(w, x, L, R, LT, RT, y, Lw, Rx, lr, qexp, iters;
						   eps=1e-12, gamma=nothing, polyak_step=false,
						   wk=nothing, xk=nothing)

	Runs the subgradient method for the robust blind deconvolution objective
	using a geometrically decaying rate, ``\\lambda q^k``, where
	``\\lambda > 0, q \\in (0, 1)``. If `gamma == nothing`, the norm constraint
	is not enforced. Additionally, if the normalized frobenius distance falls
	below `eps`, terminates. `eps` is set to 1e-12 by default.
	`Lw` and `Rx` are temporary vectors used to hold intermediate results.

	Returns the two approximate solutions found and an array containing the
	frobenius distance between the vector estimates and the true signals.
	"""
	function subgradient_method(w, x, L, R, LT, RT, y, Lw, Rx, lr, qexp, iters;
								eps=1e-12, gamma=nothing, polyak_step=false,
								wk=nothing, xk=nothing)
		m = length(y); d1, d2 = length(w), length(x)
		# use broadcasting operator to avoid overheads
		wxf = norm(w) * norm(x)
		frob = (wv, xv) -> Utils.frob_opt(wv, xv, w, x) / wxf
		# initialize via Duchi
		if wk == nothing
			wk = randn(d1)
		end
		if xk == nothing
			xk = randn(d2)
		end
		oups = fill(0.0, iters)
		# initialize placeholders for gradients, etc.
		gw, gx, r = zero(wk), zero(xk), zero(y)
		# initialize temporary placeholders
		tmpw = zero(wk); tmpx = zero(xk)
		if gamma != nothing
			postmap = (x::Array{Float64, 1}) -> project2ball!(x, gamma)
		else
			postmap = (x::Array{Float64, 1}) -> x
		end
		rate = lr * qexp
		for i = 1:iters
			subgrad!(gw, gx, r, L, R, LT, RT, y, wk, xk, Lw, Rx, tmpw, tmpx)
			norm_mag = norm(gw)^2 + norm(gx)^2
			if polyak_step   # use polyak step size
				loss = robust_loss(r, wk, xk, y, L, R, Lw, Rx)
				step = loss / norm_mag
				LinearAlgebra.BLAS.axpy!(-d1 * step, gw, wk)
				LinearAlgebra.BLAS.axpy!(-d2 * step, gx, xk)
			else
				rmul!(gw, (1 / norm_mag)); rmul!(gx, (1 / norm_mag))
				# update wk, xk
				LinearAlgebra.BLAS.axpy!(-rate, gw, wk)
				LinearAlgebra.BLAS.axpy!(-rate, gx, xk)
			end
			oups[i] = frob(wk, xk)
			if oups[i] <= eps
				oups = oups[1:i]
				break
			end
			rate *= qexp
		end
		return wk, xk, oups
	end


	"""
		subgradient_solver(prob::BCProb, lr, qexp, iters; init)

	Runs the subgradient method for the robust blind deconvolution objective
	given a blind deconvolution problem instance, `prob`.
	"""
	function subgradient_solver(prob, lr, qexp, iters; init="direction",
								polyak_step=false)
		m = length(prob.y)
		wk, xk = init_method(prob, init=init)
		return subgradient_method(prob.w, prob.x, prob.L, prob.R, prob.LT,
								  prob.RT, prob.y, fill(0.0, m), fill(0.0, m),
								  lr, qexp, iters, polyak_step=polyak_step,
								  wk=wk, xk=xk)
	end


	function subgradient_solver_noinit(prob, lr, qexp, iters; polyak_step=false,
									   wk=nothing, xk=nothing)
		m = length(prob.y)
		return subgradient_method(prob.w, prob.x, prob.L, prob.R, prob.LT,
								  prob.RT, prob.y, fill(0.0, m), fill(0.0, m),
								  lr, qexp, iters, polyak_step=polyak_step,
								  wk=wk, xk=xk)
	end



	function gen_problem(m :: Int, w::Array{Float64, 1}, x::Array{Float64, 1},
						 pfail=0.0, adv_noise=false)
		Lw = fill(0.0, m); Rx = fill(0.0, m)
		d1 = length(w); d2 = length(x); y = fill(0.0, m)
		L, LT = genmat(m, d1); R, RT = genmat(m, d2)
		if adv_noise
			# generate measurements and impute a different signal
			generate_measurements!(y, L, R, w, x, 0.0)
			L(Lw, randn(d1)); R(Rx, randn(d2))
			# replace indices
			add_noise!(y, (Lw .* Rx), pfail, true)
		else
			generate_measurements!(y, L, R, w, x, pfail)
		end
		return BCProb(L, R, LT, RT, y, w, x)
	end


	function load_problem(m::Int, w::Array{Float64, 1}, x::Array{Float64, 1},
						  sl::Array{Float64, 2}, sr::Array{Float64, 2},
						  y::Array{Float64, 1})
		Lw = fill(0.0, m); Rx = fill(0.0, m)
		d1, d2 = length(w), length(x); k = size(sl)[2];
		L = ((r, v) -> Utils.A_safe!(r, v, sl, k, d1))
		LT = ((r, v, tmp) -> Utils.AT_safe!(r, v, sl, k, d1, tmp))
		R = ((r, v) -> Utils.A_safe!(r, v, sr, k, d2))
		RT = ((r, v, tmp) -> Utils.AT_safe!(r, v, sr, k, d2, tmp))
		return BCProb(L, R, LT, RT, y, w, x)
	end


	function solve_problem(m::Int, w::Array{Float64, 1}, x::Array{Float64, 1},
						   lr, qexp, iters, pfail=0.0, adv_noise=false)
		Lw = fill(0.0, m); Rx = fill(0.0, m)
		d1 = length(w); d2 = length(x); y = fill(0.0, m)
		L, LT = genmat(m, d1); R, RT = genmat(m, d2);
		if adv_noise
			generate_measurements!(y, L, R, w, x, 0.0)
			# generate signal and replace at indices
			L(Lw, randn(d1)); R(Rx, randn(d2)); broadcast!(*, Lw, Lw, Rx)
			add_noise!(y, Lw, pfail, true)
		else
			generate_measurements!(y, L, R, w, x, pfail)
		end
		return subgradient_method(w, x, L, R, LT, RT, y, Lw, Rx,
								  lr, qexp, iters)
	end

end
