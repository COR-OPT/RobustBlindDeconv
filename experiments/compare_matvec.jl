#!/usr/bin/env julia

#=
compare_matvec.jl: Compares the number of matrix-vector multiplications and
"inner" iterations required by the proximal and subgradient methods to reach
a desired normalized accuracy.
=#

include("../src/BlindDeconv.jl")

using ArgParse
using Random
using Statistics
using Printf

function main()
	# parse arguments
	s = ArgParseSettings(description="""
						 Compares the number of matrix-vector multiplications
						 and inner iterations required by the proximal and
						 subgradient methods to reach a desired normalized
						 accuracy, printing the results to stdout.
						 """)
	@add_arg_table s begin
		"--seed"
			help = "The seed of the RNG"
			arg_type = Int
			default = 999
		"--noise"
			help = "The level of noise"
			arg_type = Float64
			default = 0.00
		"--dim"
			help = "The problem dimension"
			arg_type = Int
			default = 50
		"--eps"
			help = "The desired accuracy to reach"
			arg_type = Float64
			default = 1e-5
		"--adv_noise"
			help = """
				Set to true if adversarial noise is to be used, i.e. a
				signal will be imputed to the measurements"""
			action = :store_true
		"--prox_iters"
			help = "The number of iterations of the prox-linear method"
			arg_type = Int
			default = 15
		"--subgrad_iters"
			help = "The number of iterations of the subgradient method"
			arg_type = Int
			default = 1000
		"--i"
			help = "The coefficient of (d1 + d2) in m = i * (d1 + d2)"
			arg_type = Int
			default = 4
		"--reps"
			help = "The number of repeats for the prox-linear method"
			arg_type = Int
			default = 15
	end
	parsed = parse_args(s)
	i, pfail, eps = parsed["i"], parsed["noise"], parsed["eps"]
	anoise, rnd_seed = parsed["adv_noise"], parsed["seed"]
	prox_iters, sub_iters = parsed["prox_iters"], parsed["subgrad_iters"]
	# joint dimensions
	dim = parsed["dim"]
	# seed RNG
	Random.seed!(rnd_seed)
	m = i * 2 * dim
	reps = parsed["reps"]
	matvecs = fill(0, reps); initers = fill(0, reps)
	vecsperiter = fill(0.0, reps)
	# average [reps] problems
	for i in 1:reps
		prob = BlindDeconv.gen_problem(m, dim, dim, BlindDeconv.gaussian,
									   BlindDeconv.gaussian, pfail, anoise)
		_, _, _, totit = BlindDeconv.proximal_method(prob, 1, prox_iters,
													 get_iters=true, eps=eps)
		initers[i] = totit
		# get matrix-vector multiplication count
		matvecs[i] = BlindDeconv.getMatvecCount()
		BlindDeconv.clearMatvecCount()
		vecsperiter[i] = matvecs[i] / totit
	end
	# get all metrics for prox-linear method
	prox_avgcount = mean(matvecs); prox_stdcount = std(matvecs)
	prox_avgiters = mean(initers); prox_stditers = std(initers)
	prox_avgperiter = mean(vecsperiter); prox_stdperiter = std(vecsperiter)
	# get subgradient stats
	prob = BlindDeconv.gen_problem(m, dim, dim, BlindDeconv.gaussian,
								   BlindDeconv.gaussian, pfail, anoise)
	_, _, ds = BlindDeconv.subgradient_method(prob, 1, 0.99, sub_iters,
											  eps=eps)
	sub_count = BlindDeconv.getMatvecCount()
	BlindDeconv.clearMatvecCount()
	# print stats
	@printf("Dim: %d\n", dim)
	@printf("Prox Count: %.3f +/- %.3f\n", prox_avgcount, prox_stdcount)
	@printf("Prox Iters: %.3f +/- %.3f\n", prox_avgiters, prox_stditers)
	@printf("Prox per-iter: %.3f +/- %.3f\n", prox_avgperiter, prox_stdperiter)
	@printf("Subgrad {iters, count, periter}: (%d, %d, %.3f)\n",
			length(ds), sub_count, (sub_count / length(ds)))
end

main()
