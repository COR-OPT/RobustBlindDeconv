#!/usr/bin/env julia

#=
run_all.jl: Run all the experiments from the paper with the parameters
mentioned in the main text.
=#

include("../src/BlindDeconv.jl")
include("../src/BlindDeconvHadamard.jl")
include("ImageUtils.jl")

using ArgParse
using LinearAlgebra
using Random
using Printf
using Statistics

using Images
using ImageMagick
using MLDatasets
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

"""
	run_mnist(method, c, idx_w, idx_x, iters, pfail, q)

Load 2 digits from the MNIST training dataset, with indices `idx_w` and `idx_x`
respectively, set up a blind deconvolution problem using them as the real
signals, and show the recovered images.
"""
function run_mnist(method, c, idx_w, idx_x, iters, pfail, q)
	train_x, train_y = MNIST.traindata()  # load images
	w_img = Array(train_x[:, :, idx_w]')
	x_img = Array(train_x[:, :, idx_x]')
	w_lbl, x_lbl = train_y[idx_w], train_y[idx_x]  # labels
	w_sig = ImageUtils.gray2features(w_img)
	x_sig = ImageUtils.gray2features(x_img)
	wx, wy = size(w_img); xx, xy = size(x_img)
	d1, d2 = wx * wy, xx * xy
	# create save path
	save_path = string("mnist_", idx_w, "_", idx_x, "_")
	# setup problem
	prob = BlindDeconv.gen_problem(c * (d1 + d2), w_sig, x_sig,
		BlindDeconv.gaussian, BlindDeconv.gaussian, pfail)
	w_init = ImageUtils.features2gray(prob.w0, wx, wy)
	x_init = ImageUtils.features2gray(prob.x0, xx, xy)
	if method == "proximal"
		wk, xk, error_hist = BlindDeconv.proximal_method(prob, 1.0, iters)
	else
		wk, xk, error_hist = BlindDeconv.subgradient_method(prob, 1.0, q, iters)
	end
	w_imgf = ImageUtils.features2gray(wk, wx, wy)
	x_imgf = ImageUtils.features2gray(xk, xx, xy)
	# show images
	subplot(2, 2, 1)
	imshow(w_init, cmap="gray"); title("$idx_w: initial")
	subplot(2, 2, 2)
	imshow(w_imgf, cmap="gray"); title("$idx_w: recovered")
	subplot(2, 2, 3)
	imshow(x_init, cmap="gray"); title("$idx_x: original")
	subplot(2, 2, 4)
	imshow(x_imgf, cmap="gray"); title("$idx_x: recovered")
	show()
end


function run_color(pathw, pathx, c, iters, num_saves)
    wimg = Images.load(pathw); ximg = Images.load(pathx)
    wr, wg, wb, dwx, dwy = ImageUtils.rgbim2channelfeatures(wimg)
    xr, xg, xb, dxx, dxy = ImageUtils.rgbim2channelfeatures(ximg)
    flag = (dwx == dxx) && (dxx == dxy) && (dwy == dxy)
    m = c * dwx * dwy
    flag || throw(ArgumentError("
        Images need to be square and have the same dimensions!"))
    rProb = BlindDeconvHadamard.gen_problem(m, wr, xr, 0.0)
    gProb = BlindDeconvHadamard.gen_problem(m, wg, xg, 0.0)
    bProb = BlindDeconvHadamard.gen_problem(m, wb, xb, 0.0)
    wRed, xRed = BlindDeconvHadamard.init_method(rProb, init="direction")
    wGreen, xGreen = BlindDeconvHadamard.init_method(gProb, init="direction")
    wBlue, xBlue = BlindDeconvHadamard.init_method(bProb, init="direction")
    # save initializations
    @printf("Saving initialization...")
    rgbw = ImageUtils.rescale_from_chan(wRed, wGreen, wBlue, dwx, dwy)
    rgbx = ImageUtils.rescale_from_chan(xRed, xGreen, xBlue, dwx, dwy)
    Images.save("init_w.jpg", rgbw)
    Images.save("init_x.jpg", rgbx)
    # setup error histories
    rHist = fill(0.0, 0); gHist = fill(0.0, 0); bHist = fill(0.0, 0)
    # divide iterations into [num_saves] groups
    div_epochs = Int(ceil(iters / num_saves))
    for k = 1:num_saves
        println(@sprintf("Recovering red channel at %d...", k))
        wRed, xRed, er = BlindDeconvHadamard.subgradient_solver_noinit(
            rProb, 1.0, 1.0, div_epochs, polyak_step=true, wk=wRed, xk=xRed)
        println(@sprintf("Recovering green channel at %d...", k))
        wGreen, xGreen, eg = BlindDeconvHadamard.subgradient_solver_noinit(
            gProb, 1.0, 1.0, div_epochs, polyak_step=true, wk=wGreen, xk=xGreen)
        println(@sprintf("Recovering blue channel at %d...", k))
        wBlue, xBlue, eb = BlindDeconvHadamard.subgradient_solver_noinit(
            bProb, 1.0, 1.0, div_epochs, polyak_step=true, wk=wBlue, xk=xBlue)
        println(@sprintf("Saving iterate at %d...", k))
        append!(rHist, er[:]); append!(gHist, eg[:]); append!(bHist, eb[:])
		println(@sprintf("Current errors: %.6f, %.6f, %.6f",
						 er[end], eg[end], eb[end]))
        # save iterates
        rgbw = ImageUtils.rescale_from_chan(wRed, wGreen, wBlue, dwx, dwy)
        rgbx = ImageUtils.rescale_from_chan(xRed, xGreen, xBlue, dwx, dwy)
        Images.save(@sprintf("w_iter_%d.jpg", k), rgbw)
        Images.save(@sprintf("x_iter_%d.jpg", k), rgbx)
    end
    # equalize history lengths
    maxLen = maximum(length.((rHist, gHist, bHist)))
    append!(rHist, fill(0.0, maxLen - length(rHist)))
    append!(gHist, fill(0.0, maxLen - length(gHist)))
    append!(bHist, fill(0.0, maxLen - length(bHist)))
    semilogy(collect(1:maxLen), rHist, color="red", label="red")
    semilogy(collect(1:maxLen), gHist, color="green", label="green")
    semilogy(collect(1:maxLen), bHist, color="blue", label="blue")
    legend(); title("Error history for different channels"); show()
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
		"color_img_test"
			help = "Run the color image recovery test"
			action = :command
		"mnist_img_test"
			help = "Run the mnist image recovery test"
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

	# options for the mnist img test
	@add_arg_table s["mnist_img_test"] begin
        "--idx_w"
            help = "The index of the first digit [1-60000]"
            arg_type = Int
        "--idx_x"
            help = "The index of the second digit [1-60000]"
            arg_type = Int
        "--c"
            help = "The coefficient `c` of `m = c * (d_1 + d_2)`"
            arg_type = Int
            default = 8
        "--method"
            help = "Choose between subgradient method and proximal method"
            range_tester = (x -> x in ["proximal", "subgradient"])
            default = "subgradient"
        "--iters"
            help = "The number of iterations for minimization"
            arg_type = Int
            default = 1000
        "--noise"
            help = "The level of noise"
            arg_type = Float64
            default = 0.45
        "--q"
            help = "The step size decay parameter"
            arg_type = Float64
            default = 0.98
	end

	# options for the color img test
	@add_arg_table s["color_img_test"] begin
        "--imgw"
            help = "The path of the first image"
            arg_type = String
        "--imgx"
            help = "The path of the second image"
            arg_type = String
        "--c"
            help = "The coefficient `c` of `m = c * (d_1 + d_2)`"
            arg_type = Int
            default = 8
        "--iters"
            help = "The number of iterations for minimization"
            arg_type = Int
            default = 1000
		"--num_saves"
			help = "The number of (equispaced) iterates to save"
			arg_type = Int
			default = 18
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
	elseif parsed["%COMMAND%"] == "mnist_img_test"
		sb = parsed["mnist_img_test"]
		run_mnist(sb["method"], sb["c"], sb["idx_w"], sb["idx_x"], sb["iters"],
				  sb["noise"], sb["q"])
	elseif parsed["%COMMAND%"] == "color_img_test"
		sb = parsed["color_img_test"]
		run_color(sb["imgw"], sb["imgx"], sb["c"], sb["iters"], sb["num_saves"])
	end
end

main()
