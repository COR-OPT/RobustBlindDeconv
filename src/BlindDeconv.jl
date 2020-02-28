#!/usr/bin/env julia

"""
A module implementing subgradient and prox-linear algorithms for optimizing
a robust objective for the real-valued blind deconvolution problem.
Additionally, it employs a spectral implementation procedure for provably
obtaining initial points which are sufficiently close to optima.
"""
module BlindDeconv

    include("Utils.jl")

    using LinearAlgebra
    using LinearMaps
    using Random
    using Statistics
    using Arpack  # required for eigs
    using FFTW

    # counter of matrix-vector multiplications
    matvecCount = [0]


    """
        @enum matType gaussian complex_gaussian pdft randhadm

    Distinguish between different matrix designs. `pdft` stands for partial
    DFT matrix, and `randhadm` stands for randomized Hadamard ensemble.
    """
    @enum matType gaussian complex_gaussian pdft randhadm


    """
        mutable struct BCProb

    A struct for holding the parameters of a blind deconvolution problem
    instance. A breakdown of the parameters follows:

        L      # left measurement matrix, or a callable implementing it
        R      # right measurement matrix, or a callable implementing it
        LT     # a callable implementing the adjoint operator of L
        RT     # a callable implementing the adjoint operator of R
        y :: Array{<:Number, 1}   # the array of measurements
        w :: Array{<:Number, 1}   # true signal `w`
        x :: Array{<:Number, 1}   # true signal `x`
        w0 :: Array{<:Number, 1}  # initial guess for `w`
        x0 :: Array{<:Number, 1}  # initial guess for `x`
        pfail :: Float64         # corruption level
    """
    mutable struct BCProb
        L
        R
        LT
        RT
        y :: Array{<:Number, 1}
        w :: Array{<:Number, 1}
        x :: Array{<:Number, 1}
        w0 :: Array{<:Number, 1}
        x0 :: Array{<:Number, 1}
        pfail :: Float64
    end


    """
        opAx!(r, A, x; nconj=false)

    Compute the matrix-vector multiplication ``r = Ax``, when `A` is either
    given as a matrix or as an operator implementing the multiplication. The
    result is stored in `r`. If `nconj` is set to true, does not conjugate `A`
    before computing the result.
    """
    function opAx!(r, A, x; nconj=false)
        if isa(A, Array{<:Number, 2})
            copyto!(r, (nconj ? A : conj(A)) * x)
        else
            A(r, x)
        end
        matvecCount[1] += 1
    end


    """
        measOp(L, R, w, x) -> r
    """
    function measOp(L, R, w, x)
        matvecCount[1] += 2
        return conj(L * conj(w)) .* (R * conj(x))
    end


    """
        residual(prob::BCProb, w, x) -> r

    Compute the residual between estimates and measurements at current iterates
    `w` and `x`. Return:
    - `r`: the residual vector
    """
    function residual(prob::BCProb, w, x)
        return measOp(prob.L, prob.R, w, x) .- prob.y
    end


    """
        robustLoss(prob::BCProb, w, x) -> ℓ

    Compute the robust (ℓ₁) loss at the current iterates `w, x`.
    """
    function robustLoss(prob::BCProb, w, x)
        return mean(abs.(residual(prob, w, x)))
    end


    function smoothLoss(prob::BCProb, w, x)
        return (1 / length(prob.y)) * norm(residual(prob, w, x))^2
    end


    csign(x) = (abs(x) <= 1e-15) ? zero(x) : (x / (2 * abs(x)))



    """
        subgrad(prob::BCProb, w, x) -> (gw, gx)

    Compute a subgradient of the robust loss at the current iterates `w, x`.
    Return:
    - `gw`: the subgradient with respect to `w`
    - `gx`: the subgradient with respect to `x`
    """
    function subgrad(prob::BCProb, w, x)
        m = length(prob.y)
        # compute Lw/Rx terms
        Lw = conj(prob.L * conj(w)); Rx = conj(prob.R * conj(x))
        r  = (Lw .* conj(Rx) .- prob.y)
        if isa(eltype(prob.L), Complex) || isa(eltype(prob.R), Complex)
            map!(csign, r, r)
        else
            map!(sign, r, r)
        end
        gw = transpose(prob.L) * (Rx .* r)
        gx = transpose(prob.R) * (Lw .* conj(r))
        return (gw / m), (gx / m)
    end


    function gradSmooth(prob::BCProb, w, x)
        m = length(prob.y)
        r = residual(prob, w, x)
        # compute Lw/Rx terms
        Lw = conj(prob.L * conj(w)); Rx = conj(prob.R * conj(x))
        gw = transpose(prob.L) * (Rx .* r)
        gx = transpose(prob.R) * (Lw .* conj(r))
        return (gw / m), (gx / m)
    end


    """
        genMat(m, d, mtype::matType)

    Generate a `m x d` matrix for the blind deconvolution problem, using one of
    the available designs under `matType`. In the case of randomized Hadamard
    ensembles, returns a pair of callables instead.
    """
    function genMat(m, d, mtype::matType)
        if mtype == gaussian
            return randn(m, d), nothing
        elseif mtype == complex_gaussian
            return (1 / sqrt(2)) * complex.(randn(m, d), randn(m, d)), nothing
        elseif mtype == pdft
            return LinearMap{Complex}(
                X -> Utils.dft_partial(m, X),
                X -> Utils.dftT_partial(X, d),
                m, d), nothing
        else
            ispow2(d) || throw(ArgumentError("d must be a power of 2"))
            k = trunc(Int, m / d)
            s = Utils.signpat(d, k)
            # return a pair of callables
            Aop = (r, x) -> Utils.A_safe!(r, x, s, k, d)
            ATop = (r, x, tmp) -> Utils.AT_safe!(r, x, s, k, d, tmp)
            return Aop, ATop
        end
    end


    """
        generateMeasurements(L, R, w, x, pfail, noise=nothing) -> y

    Generate a set of measurements `y` given the measurement matrices `L, R`,
    the true signals `w` and `x`, the corruption probability `pfail` and the
    type of `noise` (optional).
    """
    function generateMeasurements(L, R, w, x, pfail, noise=nothing)
        y = measOp(L, R, w, x)
        num_corr = trunc.(Int, pfail * length(y))
        if noise == nothing
            noise = randn(num_corr)
            y[randperm(length(y))[1:num_corr]] += noise
        else
            # replace indices with noise provided
            y[randperm(length(y))[1:num_corr]] = noise[1:num_corr]
        end
        return y
    end


    """
        spectralInit(y, L, R)

    Compute the spectral initialization of Li et al. (2016) given matrices `L`
    and `R` and measurements `y`, with `L` assumed to be the partial DFT matrix.
    """
    function spectralInit(y, L, R)
        m = length(y); B = L / sqrt(m)
        # form operator
        svdObj, _ = svds(((y / sqrt(m)) .* B)'R, nsv=1); d = svdObj.S[1]
        h₀ = svdObj.U[:]; x₀ = svdObj.V[:]
        # range for upper bound
        ϵ = 0.1; ub = 2 * sqrt(d / m) * ϵ
        wev = B' * Utils.ℓ₂ProjComplex(sqrt(d) * (B * h₀), ub)
        xev = sqrt(d) * x₀
        return wev, xev
    end


    """
        directionInit!(wev, xev, y, L, R LT, RT; kw=1, kx=1)

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
    function directionInit!(wev, xev, y, L, R, LT=nothing, RT=nothing;
                            kw=1, kx=1)
        med = median(abs.(y)); d1, d2 = length(wev), length(xev)
        ind = abs.(y) .<= med
        if isa(L, Array{<:Number, 2})
            Lind = L[ind, :]
            _, wev[:] = eigs(Lind'Lind, nev=1, which=:SM)
        else
            Utils.specinit_op!(wev, trunc.(Int, ind), kw, L, LT)
        end
        if isa(R, Array{<:Number, 2})
            Rind = R[ind, :]
            _, xev[:] = eigs(Rind'Rind, nev=1, which=:SM)
        else
            Utils.specinit_op!(xev, trunc.(Int, ind), kx, R, RT)
        end
    end


    """
        radiusInit!(wev, xev, y, L, R)

    Compute the optimal magnitude of the initial estimates of the spectral
    initialization `wev, xev`, given the measurements `y` and the matrices
    `L, R`.
    """
    function radiusInit!(wev, xev, y, L, R)
        m = length(y)
        Lw = fill(zero(eltype(y)), m); Rx = fill(zero(eltype(y)), m)
        Lw = conj(L) * wev; Rx = R * conj(xev)
        ips = Rx .* Lw
        f = r -> (1 / m) * norm(r * ips - y, 1)
        if eltype(y) <: Complex
            g = r -> (1 / m) * sum(csign.(r * ips - y) .* ips)
        else
            g = r -> (1 / m) * sum(sign.(r * ips - y) .* ips)
        end
        η = 5
        rbest = Utils.fmin_alin(f, g, η, 1.0, 1000, ϵ=1e-8)
        rmul!(wev, sign(rbest) * sqrt(abs(rbest)))
        rmul!(xev, sqrt(abs(rbest)))
    end


    """
        project2ball!(x, γ=1.0)

    Projects a vector `x` to the ``\\ell_2``-norm ball, scaled by ``\\gamma``.
    The modification is performed in place.
    """
    function project2ball!(x, γ=1.0)
        if norm(x) > γ
            rmul!(x, γ / norm(x))
        end
    end


    """
        subgradientMethod(prob, λ, q, T; ϵ=1e-10, γ=nothing, use_polyak=false) -> (wk, xk, dists)

    Run the subgradient method for the robust blind deconvolution objective
    using a geometrically decaying step size, ``\\lambda * q^k``, for `T`
    iterations. Optionally, if `use_polyak=true`, uses a Polyak step.

    Return:
    - `wk, xk`: the iterates found by the method
    - `dists`: the history of iterate distances
    """
    function subgradientMethod(prob, λ, q, T; ϵ=1e-10, γ=nothing,
                               use_polyak=false)
        d1, d2 = length(prob.w), length(prob.x)
        wxf = norm(prob.w) * norm(prob.x)
        wxM = prob.w .* prob.x'
        distFun = (w, x) -> norm(w .* x' - wxM) / wxf
        dists = fill(0.0, T)
        xk, wk = copy(prob.x0), copy(prob.w0)
        η = λ
        for i = 1:T
            dists[i] = distFun(wk, xk)
            (dists[i] ≤ ϵ) && return wk, xk, dists[1:i]
            gw, gx = subgrad(prob, wk, xk)
            gMag = norm(vcat(gw, gx))
            if use_polyak
                loss = robustLoss(prob, wk, xk)
                step = loss / (gMag^2)
                wk[:] = wk - step * gw
                xk[:] = xk - step * gx
            else
                wk[:] = wk - η * (gw / gMag)
                xk[:] = xk - η * (gx / gMag)
            end
            η *= q
            if (γ != nothing)
                project2ball!(wk, γ); project2ball!(xk, γ)
            end
        end
        return wk, xk, dists
    end


    function gradientMethod(prob, T, λ=1.0, q=1.0; γ=1.0, ϵ=1e-10, use_polyak=true)
        d1, d2 = length(prob.w), length(prob.x)
        wxf    = norm(prob.w) * norm(prob.x)
        wxM    = prob.w .* prob.x'
        distFun = (w, x) -> norm(w .* x' - wxM) / wxf
        dists = fill(0.0, T)
        xk, wk = copy(prob.x0), copy(prob.w0)
        η = λ
        for i = 1:T
            dists[i] = distFun(wk, xk)
            (dists[i] ≤ ϵ) && return wk, xk, dists[1:i]
            gw, gx = gradSmooth(prob, wk, xk)
            gMag = norm(vcat(gw, gx))
            if use_polyak
                loss = smoothLoss(prob, wk, xk)
                step = loss / (gMag^2)
                wk[:] = wk - step * gw
                xk[:] = xk - step * gx
            else
                wk[:] = wk - η * (gw / gMag)
                xk[:] = xk - η * (gx / gMag)
            end
            if (γ != nothing)
                project2ball!(wk, γ); project2ball!(xk, γ)
            end
            η *= q
        end
        return wk, xk, dists
    end


    """
        proximal_method(prob, α, k, γ; get_iters, ϵ=1e-12, inner_ϵ, ρ=nothing) -> (wk, xk, dists, [inIters])

    Run the prox-linear method on the robust loss for `k` iterations with a prox
    step size of `α`. Return the two approximate solutions found and an array
    containing the frobenius distance of the vector estimates and the true signals.

    The subproblems solved at every step look like

    ``
        \\arg \\min_x f_{x_k}(x) + \\frac{1}{2 \\alpha} \\| x - x_k \\|^2
        s.t. \\| x \\|_2 \\leq \\gamma
    ``
    If `γ` is `nothing`, then the norm constraint is not enforced.
    If `get_iters` is set to true, returns the total number of "inner"
    iterations performed for all subproblems. `inner_eps` can be either an
    absolute number or a callable with a single integer parameter (representing
    the current iteration).

    Return:
    - `wk, xk`: the final iterates found by the method
    - `dists`: an array containing the distance history of iterates
    - `inIters`: optionally, return the total number of inner iterations performed
    """
    function proximal_method(prob::BCProb, α, k, γ=nothing;
        get_iters=false, ϵ=1e-12, inner_ϵ=(j -> min(1e-3, 2.0^(-j))), ρ=nothing)
        # stop user if Hadamard matrices are used
        (isa(prob.L, Array{Float64, 2}) && isa(prob.R, Array{Float64, 2})) ||
            throw(ArgumentError(
                "Proximal method is currently unavailable for matrices " *
                "implemented as callables."))
        wxfnorm = sqrt(sum((prob.w).^2) * sum((prob.x).^2))
        frob = (wv, xv) -> Utils.frob_opt(wv, xv, prob.w, prob.x) / wxfnorm
        xk, wk = copy(prob.x0), copy(prob.w0)
        d, n, m = length(wk), length(xk), length(prob.y)
        dists = fill(0., k); initers = 0
        for i = 1:k
            dists[i] = frob(wk, xk)
            wk, xk, itcount = pogs_solve(
                wk, xk, prob, α, γ, inner_ϵ(i), ρ)
            dists[i] = frob(wk, xk)
            initers += itcount
            if dists[i] <= ϵ
                dists = dists[1:i]
                break  # stop if desired accuracy was reached
            end
        end
        if get_iters
            return wk, xk, dists, initers
        else
            return wk, xk, dists
        end
    end


    """
        soft_thres(x, α)

    Implements the soft thresholding operator with threshold `α`.
    """
    function soft_thres(x, α)
        return sign.(x) .* max.(abs.(x) .- α, 0)
    end


    """
        pogs_solve(wk, xk, prob, α, γ, tol=1e-3, ρ=nothing) -> (wk, xk, itcount)

    Solves the following problem:

    ``
    \\mbox{Minimize } \\frac{1}{m} \\left\\| Ax - c \\right\\|_1 +
    \\frac{1}{2a} \\left\\| x - d \\right\\|_2^2 \\
    \\mbox{s.t. } \\| x \\|_2 \\leq \\gamma
    ``

    which corresponds to the proximal subproblems of the linearized blind
    deconvolution objective. If `γ == nothing`, the norm constraint is
    not enforced.

    Return:
    - `wk, xk`: the iterates found by the method
    - `itcount`: the total number of iterations performed
    """
    function pogs_solve(wk, xk, prob, α, γ, tol=1e-3, ρ=nothing)
        m = length(prob.y); d = length(wk); n = length(xk)
        Rx = prob.R * xk; Lw = prob.L * wk
        # update counts
        matvecCount[1] += 2
        # matrix A
        A = hcat(Rx .* prob.L, Lw .* prob.R)
        # offset vector c
        c = prob.y - (Lw .* Rx)
        # variables z, y, initialized to 0
        z = fill(0., length(wk) + length(xk)); y = fill(0., m)
        zold = copy(z); yold = copy(y)
        # cholesky fact of 1.0I + Aᵀ A
        Cchol = cholesky(1.0I + A' * A)
        # update counts (take into account matrix structure)
        matvecCount[1] += size(A)[2]
        # dual variables
        ℓ = fill(0., length(z)); ν = fill(0., m);
        # set ρ parameter
        ρ = (ρ == nothing) ? (1 / m) : ρ
        fx = ρ * α; fy = (ρ * m)
        # residuals
        res_p = res_d = 0
        # accuracies
        stopn = sqrt(length(z))
        eps_p = (v1, v2) -> tol * (stopn + max(norm(v1), norm(v2)))
        eps_d = (ℓ₁, ℓ₂) -> tol * (stopn + max(norm(ℓ₁), norm(ℓ₂)))
        iters = 1
        while true
            zhalf = (fx / (1 + fx)) * (z - ℓ)
            if γ != nothing
                project2ball!(view(zhalf, 1:d), γ)   # project w-part
                project2ball!(view(zhalf, (d+1):(d+n)), γ)  # project x-part
            end
            yhalf = c + soft_thres(y - ν - c, 1 / fy)
            tk = zhalf + ℓ + A' * (yhalf + ν)
            # update counts
            matvecCount[1] += 1
            copyto!(z, Cchol.U \ (Cchol.L \ tk)); copyto!(y, A * z)
            # update counts
            matvecCount[1] += 2
            ℓ += zhalf - z; ν += yhalf - y
            res_p = norm(vcat(z - zhalf, y - yhalf))
            res_d = ρ * (norm(vcat(zold - z, yold - y)))
            if (res_p < eps_p(z, y)) && (res_d < eps_d(ℓ, ν))
                break
            end
            copyto!(zold, z); copyto!(yold, y)
            iters += 1
        end
        return z[1:d] + wk, z[(d+1):end] + xk, iters
    end


    function addNoise!(y, noise, pfail)
        m = length(y); numCorr = trunc(Int, m * pfail)
        y[randperm(1:m)][1:numCorr] = noise[1:numCorr]
    end

    """
        genProblem(m::Int, w::Array, x::Array, Ltype::MatType, Rtype::MatType,
                   pfail=0.0, adv_noise=false)

    Set up a problem where matrix `L` is of `Ltype` and `R` is of `Rtype`,
    given the true signals `w` and `x`. If `adv_noise` is `true`, impute a
    randomly generated signal into the set of measurements, at indices chosen
    uniformly at random.
    Return the corresponding problem struct, including all the problem data up
    until initialization.
    """
    function genProblem(m::Int, w::Array{<:Number, 1}, x::Array{<:Number, 1},
                        Ltype::matType, Rtype::matType, pfail=0.0, adv_noise=false)
        Lw = fill(0., m); Rx = fill(0., m)
        d1 = length(w); d2 = length(x); y = fill(0., m)
        L, LT = genMat(m, d1, Ltype); R, RT = genMat(m, d2, Rtype)
        # allocate measurement vector
        if (Ltype == randhadm) || (Rtype == randhadm)
            y = fill(0.0, m)
        else
            y = fill(zero(eltype(L)), m)
        end
        if adv_noise
            # generate measurements and impute a different signal
            y[:] = generateMeasurements(L, R, w, x, 0.0)
            opAx!(Lw, L, randn(d1))
            opAx!(Rx, R, randn(d2))
            # replace indices
            addNoise!(y, (Lw .* Rx), pfail)
        else
            y[:] = generateMeasurements(L, R, w, x, pfail)
        end
        # reset matvec counts
        matvecCount[1] = 0
        # choose appropriate initialization
        if (Ltype == pdft) || (Rtype == pdft)
            w0, x0 = spectralInit(y, L, R)
            return BCProb(L, R, LT, RT, y, w, x, w0, x0, pfail)
        else
            w0 = fill(zero(eltype(y)), d1); x0 = fill(zero(eltype(y)), d2)
            directionInit!(w0, x0, y, L, R, LT, RT, kw=(m / d1), kx=(m / d2))
            radiusInit!(w0, x0, y, L, R)
            return BCProb(L, R, LT, RT, y, w, x, w0, x0, pfail)
        end
    end


    """
        genProblem(m::Int, d1::Int, d2::Int, Ltype::MatType, Rtype::MatType,
                   pfail::Float64)

    If only the dimensions of the input signals are given instead, generate
    them randomly.
    """
    function genProblem(m::Int, d1::Int, d2::Int,
        Ltype::matType, Rtype::matType, pfail=0.0, adv_noise=false)
        w = randn(d1); x = randn(d2); normalize!(w); normalize!(x)  # unit norm
        return genProblem(m, w, x, Ltype, Rtype, pfail, adv_noise)
    end


    """
        genProblem(w::Array{<:Number, 1}, x::Array{<:Number, 1},
                   Lmat::Array{<:Number, 2}, Rmat::Array{<:Number, 2},
                   pfail::Float64)

    If the user supplies the measurement matrices instead, generate the
    measurements accordingly.
    """
    function genProblem(w::Array{<:Number, 1}, x::Array{<:Number, 1},
                        Lmat::Array{<:Number, 2}, Rmat::Array{<:Number, 2},
                        pfail::Float64)
        d1, d2 = length(w), length(x); m = size(Lmat)[1]
        y = generateMeasurements(Lmat, Rmat, w, x, pfail)
        w0 = fill(zero(eltype(y)), d1); x0 = fill(zero(eltype(y)), d2)
        directionInit!(w0, x0, y, Lmat, Rmat, nothing, nothing,
                       kw=(m / d1), kx=(m / d2))
        radiusInit!(w0, x0, y, Lmat, Rmat)
        return BCProb(Lmat, Rmat, nothing, nothing, y, w, x, w0, x0, pfail)
    end


    """
        genCoherentProblem(d, m, λ, δ, pfail=0.0)

    Generate a "coherent" problem with `m` measurements and signal dimension
    `d`, with the "left" signal generated as the `λ`-convex combination of the
    normalized all-ones vector and the first canonical basis vector. In this
    setting, the "left" measurement matrix is a partial DFT matrix and the
    "right" measurement matrix is a complex Gaussian matrix.
    """
    function genCoherentProblem(d, m, λ, δ, pfail=0.0)
        w = Utils.genCoherentVec(d, λ); x = normalize(randn(d))
        # dft-type measurements
        Ltype = BlindDeconv.pdft; Rtype = BlindDeconv.complex_gaussian
        L, LT = genMat(m, d, Ltype); R, RT = genMat(m, d, Rtype)
        # generate measurements
        y = fill(zero(eltype(R)), m)
        y[:] = generateMeasurements(L, R, w, x, pfail)
        # initialize close to the truth, make sure both vectors are complex
        w₀   = fill(zero(eltype(y)), d); x₀ = fill(zero(eltype(y)), d)
        w₀   = w + δ * normalize(complex(randn(d)) + complex(randn(d))im) * norm(w)
        x₀   = x + δ * normalize(complex(randn(d)) + complex(randn(d))im) * norm(x)
        return BCProb(L, R, LT, RT, y, w, x, w₀, x₀, pfail)
    end


    """
        clearMatvecCount()

    Resets the counter of matrix-vector products.
    """
    function clearMatvecCount()
        matvecCount[1] = 0;
    end

    """
        getMatvecCount()

    Return the counter of matrix-vector products.
    """
    function getMatvecCount()
        return matvecCount[1]
    end
end
