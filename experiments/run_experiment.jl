#!/usr/bin/env julia

#=
run_all.jl: Run all the experiments from the paper with the parameters
mentioned in the main text.
=#

include("../src/BlindDeconv.jl")

using ArgParse
using LinearAlgebra
using Random
using Printf
using Statistics
using PyPlot


"""
    run_decay_eval(c::Int, pfail::Float64, reps::Int)

Run the decay evaluation experiment for the given noise level `pfail`, when
the number of measurements is ``m = c \\cdot (d_1 + d_2)``.
"""
function run_decay_eval(c::Int, pfail::Float64, reps::Int)
    final_dists = zero(randn(20, reps))
    d1 = d2 = 100; m = c * (d1 + d2)
    # generate q range
	q = collect(range(0.9, stop=0.995, length=20))
	for idx in 1:20
		@printf("Testing q: %.3f\n", q[idx])
		for rep in 1:reps
			prob = BlindDeconv.gen_problem(
				m, d1, d2, BlindDeconv.gaussian, BlindDeconv.gaussian, pfail)
			_, _, ds = BlindDeconv.subgradient_method(prob, 1.0, q[idx], 1000)
			# save last nonzero distance
			final_dists[idx, rep] = ds[end - 1]
		end
	end
	# get sample mean, sample standard dev
	mean_ds = mean(final_dists, dims=2); dev_ds = sqrt.(var(final_dists, dims=2))
	# plot them
	yscale("log")  # log scale for y
	errorbar(q, mean_ds, yerr=dev_ds)
	title_string = latexstring("\\text{Avg. error}, c = $c");
	title(title_string); xlabel(L"$ q $"); show()
end


"""
	run_synth_test(method, c, d1, d2, pfail, q, iters)

Run a synthetic test, generating a problem with Gaussian measurement vectors,
for a given choice of dimensions and noise.
"""
function run_synth_test(method, c, d1, d2, pfail, q, iters)
	m = c * (d1 + d2)
	prob = BlindDeconv.gen_problem(m, d1, d2, BlindDeconv.gaussian,
								   BlindDeconv.gaussian, pfail)
	if method == "proximal"
		_, _, ds = BlindDeconv.proximal_method(prob, 1, iters)
	else
		# use polyak step by default without noise
		_, _, ds = BlindDeconv.subgradient_method(prob, 1.0, q, iters,
												  polyak_step=(pfail == 0.0))
	end
	yscale("log")
	semilogy(collect(1:length(ds)), ds)
	xlabel(L"$ k $"); title("Normalized error, $method method, c = $c")
	show()
end

"""
	run_matvec_test(method, c, reps)

Run the matrix-vector multiplication experiment, for pfail in the range
[0.0, 0.25], until the required accuracy is achieved.
"""
function run_matvec_eval(method, c, reps)
	d1 = d2 = 100; m = c * (d1 + d2)
	pfs = collect(range(0.0, stop=0.25, length=6))
	matvecs = zero(randn(6, reps))
	if method == "proximal"
		solver = (p -> BlindDeconv.proximal_method(p, 1, 15, eps=1e-5))
	else
		solver = (p -> BlindDeconv.subgradient_method(p, 1, 0.98, 1000,
													  eps=1e-5))
	end
	for k = 1:6
		@printf("Testing pfail: %.2f\n", pfs[k])
		for i = 1:reps
			prob = BlindDeconv.gen_problem(m, d1, d2, BlindDeconv.gaussian,
										   BlindDeconv.gaussian, pfs[k])
			_, _, ds = solver(prob)
			matvecs[k, i] = BlindDeconv.getMatvecCount()
			BlindDeconv.clearMatvecCount()
		end
	end
	avgcount = mean(matvecs, dims=2); stdcount = std(matvecs, dims=2)
	semilogy(pfs, avgcount)
	title("Matrix - vector multiplications, $method method, c = $c")
	xlabel(L"$ k $"); show()
end

function main()
	s = ArgParseSettings(description="""
						 Runs the experiments from the robust blind
						 deconvolution paper with their default parameters.""")
	@add_arg_table s begin
		"decay_eval"
			help = "Run the step size decay evaluation experiment"
			action = :command
		"synthetic_test"
			help = """
				Set up a synthetic instance and evaluate the performance of
				a selected method."""
			action = :command
		"matvec_eval"
			help = "Run the matrix-vector multiplication experiment"
			action = :command
		"--seed"
			help = "The random seed"
			arg_type = Int
			default = 999
	end

	# options for decay eval
	@add_arg_table s["decay_eval"] begin
		"--c"
			help = "The coefficient c in m = c * (d_1 + d_2)"
			required = true
			arg_type = Int
		"--pfail"
			help = "The fraction of measurements corrupted with additive noise"
			required = true
			arg_type = Float64
		"--reps"
			help = "The number of repetitions for each q"
			arg_type = Int
			default = 50
	end

	# options for synthetic test
	@add_arg_table s["synthetic_test"] begin
		"--c"
			help = "The coefficient c in m = c * (d_1 + d_2)"
			required = true
			arg_type = Int
		"--pfail"
			help = "The fraction of measurements corrupted with additive noise"
			required = true
			arg_type = Float64
		"--d1"
			help = "The dimension of w"
			arg_type = Int
			required = true
		"--d2"
			help = "The dimension of x"
			arg_type = Int
			required = true
		"--iters"
			help = "The number of iterations"
			arg_type = Int
			required = true
		"--q"
			help = "The step size decay"
			arg_type = Float64
			default = 0.98
		"--method"
			help = "Choose between subgradient method and proximal method"
			range_tester = (x -> x in ["proximal", "subgradient"])
			default = "subgradient"
	end

	# options for matvec eval
	@add_arg_table s["matvec_eval"] begin
		"--c"
			help = "The coefficient c in m = c * (d_1 + d_2)"
			required = true
			arg_type = Int
		"--method"
			help = "Choose between subgradient method and proximal method"
			range_tester = (x -> x in ["proximal", "subgradient"])
			default = "subgradient"
		"--reps"
			help = "The number of repetitions for each corruption level"
			arg_type = Int
			default = 50
	end

	# parse everything
	parsed = parse_args(s)
	Random.seed!(parsed["seed"])
	if parsed["%COMMAND%"] == "decay_eval"
		sb = parsed["decay_eval"]
		run_decay_eval(sb["c"], sb["pfail"], sb["reps"])
	elseif parsed["%COMMAND%"] == "synthetic_test"
		sb = parsed["synthetic_test"]
		run_synth_test(sb["method"], sb["c"], sb["d1"], sb["d2"],
					   sb["pfail"], sb["q"], sb["iters"])
	elseif parsed["%COMMAND%"] == "matvec_eval"
		sb = parsed["matvec_eval"]
		run_matvec_eval(sb["method"], sb["c"], sb["reps"])
	end
end

main()
