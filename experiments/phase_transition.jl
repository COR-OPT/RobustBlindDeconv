#!/usr/bin/env julia

#=
Generates a set of synthetic problem instances for a given ratio
m / (d₁ + d₂) and a range of failure probabilities. Generates 100 instances
with the same parameters and records the percentage of successful recoveries.
=#

include("../src/BlindDeconv.jl")

using ArgParse
using LinearAlgebra
using Printf
using Random
using Statistics

function main()
	# parse arguments
	s = ArgParseSettings(description="""
						 Generates a set of synthetic problem instances for
						 a given ratio m / (d_1 + d_2) and a range of failure
						 probabilities, solves them using spectral initialization
						 and subgradient descent or the prox-linear method, and
						 outputs the percentage of successful recoveries.""")
	@add_arg_table s begin
		"--dim"
			help = "The problem dimension"
			arg_type = Int
			default = 100
		"--seed"
			help = "The seed of the RNG"
			arg_type = Int
			default = 999
		"--adv_noise"
			help = """
				Set to true if adversarial noise is to be used, i.e. a
				signal will be imputed to the measurements"""
			action = :store_true
		"--prob_type"
			help =
				"The type of the problem. 'gaussian' results in problems " *
				"where both measurement matrices are gaussian. 'hadamard' " *
				"results in problems where both measurement matrices are " *
				"randomized Hadamard matrices. 'mixed' results in problems " *
				"where the left measurement matrix is deterministic and " *
				"orthonormal and the right matrix is Gaussian. 'dethadm' " *
				"results in problems where the left matrix is a partial " *
				"Hadamard matrix, and the right matrix is Gaussian."
			range_tester = (x -> lowercase(x) in [
				"gaussian", "hadamard", "mixed", "dethadm"])
			default = "gaussian"
		"--method"
			help = "Choose between subgradient method and proximal method"
			range_tester = (x -> x in ["proximal", "subgradient"])
			default = "subgradient"
		"--iters"
			help = "The number of iterations for minimization"
			arg_type = Int
			default = 750
		"--repeats"
			help = "The number of repeats for generating success rates"
			arg_type = Int
			default = 100
		"--success_dist"
			help = """The desired reconstruction distance. Iterates whose
				   normalized distance is below this threshold are considered
				   exact recoveries."""
			arg_type = Float64
			default = 1e-5
		"--q"
			help = "The exponent of the geometrically decreasing learning rate"
			arg_type = Float64
			default = 0.99
		"--i"
			help = "The ratio of measurements to d_1 + d_2"
			arg_type = Int
			default = 1
	end
	parsed = parse_args(s)
	d, anoise, rnd_seed = parsed["dim"], parsed["adv_noise"], parsed["seed"]
	method, iters = parsed["method"], parsed["iters"]
	sdist, repeats = parsed["success_dist"], parsed["repeats"]
	prob_type, i = parsed["prob_type"], parsed["i"]
	# setup parameters
	λ, q = 1, parsed["q"]
	# seed RNG
	Random.seed!(rnd_seed)
	pgen =  # setup problem generation
	(nfail -> begin
		if prob_type == "gaussian"
			BlindDeconv.gen_problem(i * 2 * d, d, d, BlindDeconv.gaussian,
								   BlindDeconv.gaussian, nfail, anoise)
		elseif prob_type == "hadamard"
			BlindDeconv.gen_problem(i * 2 * d, d, d, BlindDeconv.randhadm,
								   BlindDeconv.randhadm, nfail, anoise)
		elseif prob_type == "mixed"
			BlindDeconv.gen_problem(i * 2 * d, d, d, BlindDeconv.ortho,
								   BlindDeconv.gaussian, nfail, anoise)
		elseif prob_type == "dethadm"
			m = trunc(Int, 2^(ceil(log2(i * 2 * d))))
			BlindDeconv.gen_problem(m, d, d, BlindDeconv.dethadm,
								   BlindDeconv.gaussian, nfail, anoise)
		else
			throw(ErrorException("Something went wrong!"))
		end end)
	solver =  # setup problem solver
	 	if method == "subgradient"
			(p -> BlindDeconv.subgradient_method(p, λ, q, iters, eps=sdist))
		else
			(p -> BlindDeconv.proximal_method(p, 1, iters, eps=sdist))
		end
	if anoise
		noise_range = 0:0.02:0.38
	else
		noise_range = 0:0.02:0.48
	end
	for noise in noise_range
		# run [repeats] trials in parallel
		nsucc = 0
		for k in 1:repeats
			prob = pgen(noise)
			_, _, ds = solver(prob)
			nsucc += trunc(Int, ds[end] < sdist)
		end
		@printf("%d, %.2f, %.2f\n", i, noise, nsucc / repeats)
	end
end

main()
