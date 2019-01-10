#!/usr/bin/env julia

#=
prox_eval.jl: Evaluates the proximal method in a synthetic instance.
=#

include("../src/BlindDeconv.jl")

using ArgParse
using DataFrames
using CSV
using Printf
using Random

function main()
	# parse arguments
	s = ArgParseSettings(description="""
						 Generates a synthetic problem instance and
						 solves them using spectral initialization and
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
		"--adv_noise"
			help = """
				Set to true if adversarial noise is to be used, i.e. a
				signal will be imputed to the measurements"""
			action = :store_true
		"--iters"
			help = "The number of iterations for minimization"
			arg_type = Int
			default = 15
		"--i"
			help = "The coefficient of (d1 + d2) in m = i * (d1 + d2)"
			arg_type = Int
			default = 1
		"--inner_eps_base"
			help = """The base for the exponentially decaying accuracy for the
				inner iterations."""
			arg_type = Float64
			default = 2.0
	end
	parsed = parse_args(s)
	d1, d2, i = parsed["d1"], parsed["d2"], parsed["i"]
	anoise, rnd_seed = parsed["adv_noise"], parsed["seed"]
	iters, pfail = parsed["iters"], parsed["noise"]
	ebase = parsed["inner_eps_base"]
	# seed RNG
	Random.seed!(rnd_seed)
	# run algorithm
	m = i * (d1 + d2)
	prob = BlindDeconv.gen_problem(m, d1, d2, BlindDeconv.gaussian,
								   BlindDeconv.gaussian, pfail, anoise)
	eps_fun = (j -> min(1e-2, ebase^(-j)))
	_, _, ds = BlindDeconv.proximal_method(prob, 1, iters, inner_eps=eps_fun)
	# pad ds with ending value if necessary
	append!(ds, ds[end] * ones(iters - length(ds)))
	df = DataFrame(reshape(ds, length(ds), 1))
	fname = string("prox_eval_", d1, "x", d2, "_", parsed["i"], "_noise_",
				   @sprintf("%.2f", pfail), ".csv")
	CSV.write(fname, df, writeheader=false)
end

main()
