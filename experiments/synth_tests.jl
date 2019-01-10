#!/usr/bin/env julia

#=
synth_tests.jl: Evaluates the performance of either the subgradient or the
prox-linear methods on synthetic instances of the blind deconvolution problem.
=#

include("../src/BlindDeconv.jl")

using ArgParse
using DataFrames
using CSV
using LinearAlgebra
using Random
using Statistics

function main()
	# parse arguments
	s = ArgParseSettings(description="""
						 Generates a synthetic problem instance and
						 solves it using spectral initialization and
						 subgradient descent or the prox-linear method, saving
						 the relative error of the iterates on a .csv file.
						 """)
	@add_arg_table s begin
		"--d1"
			help = "The left-dimension of the problem"
			arg_type = Int
			default = 100
		"--d2"
			help = "The right-dimension of the problem"
			arg_type = Int
			default = 100
		"--seed"
			help = "The seed of the RNG"
			arg_type = Int
			default = 999
		"--noise"
			help = "The level of noise"
			arg_type = Float64
			default = 0.25
		"--noise_variance"
			help = "The variance of white noise, if any"
			arg_type = Float64
			default = 1.0
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
				"randomized Hadamard matrices."
			range_tester = (x -> lowercase(x) in ["gaussian", "hadamard"])
			default = "gaussian"
		"--method"
			help = "Choose between subgradient method and proximal method"
			range_tester = (x -> x in ["proximal", "subgradient"])
			default = "subgradient"
		"--iters"
			help = "The number of iterations for minimization"
			arg_type = Int
			default = 750
		"--success_dist"
			help = """The desired reconstruction distance. Iterates whose
				   normalized distance is below this threshold are considered
				   exact recoveries."""
			arg_type = Float64
			default = 1e-12
		"--q"
			help = "The exponent of the geometrically decreasing learning rate"
			arg_type = Float64
			default = 0.98
		"--i"
			help = "The coefficient of m = i * (d_1 + d_2)"
			arg_type = Int
			default = 4
		"--polyak_step"
			help = """Set to use the Polyak step size in the
					  subgradient method"""
			action = :store_true
	end
	parsed = parse_args(s)
	d1, d2 = parsed["d1"], parsed["d2"]
	anoise, rnd_seed = parsed["adv_noise"], parsed["seed"]
	method, iters = parsed["method"], parsed["iters"]
	sdist, pfail = parsed["success_dist"], parsed["noise"]
	prob_type, use_polyak = parsed["prob_type"], parsed["polyak_step"]
	noise_var = parsed["noise_variance"]
	i_coeff = parsed["i"]
	# seed RNG
	Random.seed!(rnd_seed)
	pgen =  # setup problem generation
	(nfail -> begin
		if prob_type == "gaussian"
			BlindDeconv.gen_problem(i_coeff * (d1 + d2), d1, d2,
									BlindDeconv.gaussian,
									BlindDeconv.gaussian, nfail, anoise)
		elseif prob_type == "hadamard"
			BlindDeconv.gen_problem(i_coeff * (d1 + d2), d1, d2,
									BlindDeconv.randhadm,
									BlindDeconv.randhadm, nfail, anoise)
		else
			throw(ErrorException("Something went wrong!"))
		end end)
	solver =  # setup problem solver
	 	if method == "subgradient"
			println("Using subgradient method")
			(p -> BlindDeconv.subgradient_method(p, λ, q, iters, eps=sdist,
												 polyak_step=use_polyak))
		else
			println("Using proximal method")
			(p -> BlindDeconv.proximal_method(p, 1, iters))
		end
	# setup parameters
	λ, q = 1, parsed["q"]
	ds = fill(0.0, iters)
	if anoise  # impute signal
		prob = pgen(pfail)
	else   # use info for noise variance
		prob = pgen(0.0)
		BlindDeconv.corrupt_measurements(
			prob, pfail, noise_var * randn(i_coeff * (d1 + d2)),
			"additive")
	end
	_, _, hist = solver(prob)
	# put information in columns
	df = DataFrame(iters=collect(1:length(hist)), err=hist)
	slabel = if anoise "synth_erradv_" else "synth_err_" end
	slabel = if use_polyak string(slabel, "nonoise_") else slabel end
	sprefix = if (method == "proximal") string("prox_", slabel) else slabel end
	fname = string(sprefix, d1, "x", d2, "_", i_coeff, ".csv")
	CSV.write(fname, df)
end

main()
