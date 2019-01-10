#!/usr/bin/env julia

#=
step_eval.jl: Evaluates the subgradient method's initial step size in
synthetic instances.
=#

include("../src/BlindDeconv.jl")

using Random
using Printf

using ArgParse
using Logging
using DataFrames
using CSV

function main()
	# parse arguments
	s = ArgParseSettings(description="""
						 Generates synthetic problem instances and solves them
						 using spectral initialization and subgradient descent
						 with decaying learning rate. This script computes the
						 average distance of the final iterate over 50 runs for
						 a range of values λ.""")
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
		"--adv_noise"
			help = """
				Set to true if adversarial noise is to be used, i.e. a
				signal will be imputed to the measurements"""
			action = :store_true
		"--iters"
			help = "The number of iterations for minimization"
			arg_type = Int
			default = 1000
		"--i"
			help = "The coefficient of (d1 + d2) in m = i * (d1 + d2)"
			arg_type = Int
			default = 1
		"--q"
			help = "The exponent of the geometric decay of the learning rate"
			arg_type = Float64
			default = 0.98
	end
	parsed = parse_args(s)
	d1, d2, i = parsed["d1"], parsed["d2"], parsed["i"]
	anoise, rnd_seed, q = parsed["adv_noise"], parsed["seed"], parsed["q"]
	iters, pfail = parsed["iters"], parsed["noise"]
	# seed RNG
	Random.seed!(rnd_seed)
	# run algorithm
	m = i * (d1 + d2)
	final_dists = zero(randn(20, 50))
	# generate lambda range
	λ = 2.0.^(collect(linspace(-4, 2, 20)))
	for idx in 1:20
		for reps in 1:50
			prob = BlindDeconv.gen_problem(
				m, d1, d2, BlindDeconv.gaussian,
				BlindDeconv.gaussian, pfail, anoise)
			_, _, ds = BlindDeconv.subgradient_method(prob, λ[idx], q, iters)
			# save last distance
			final_dists[idx, reps] = ds[end]
		end
	end
	# get sample mean, sample standard dev
	mean_ds = mean(final_dists, 2); dev_ds = sqrt.(var(final_dists, 2))
	# choose correct label
	df = DataFrame(lambda=λ, means=mean_ds[:], devs=dev_ds[:])
	fname = string("subgrad_eval_", d1, "x", d2, "_", parsed["i"], "_noise_",
				   @sprintf("%.2f", pfail), ".csv")
	CSV.write(fname, df)
end

main()
