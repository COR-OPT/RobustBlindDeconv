#!/usr/bin/env julia

#=
A script to test using coloured images as the true signals in the
blind deconvolution problem, using Hadamard sensing matrices.
=#

include("../src/BlindDeconvHadamard.jl")
include("ImageUtils.jl")

using ArgParse
using CSV
using DataFrames
using LinearAlgebra
using ImageMagick
using Images
using Printf
using Random
using Statistics


#= Load images and convert them to Float64 signals. Set up a corresponding
   Hadamard blind deconvolution problem. =#
function recoverImgs(pathw, pathx, p::Int, iters::Int, init;
					 num_saves=9, save_first=false)
	wimg = Images.load(pathw); ximg = Images.load(pathx)
	wr, wg, wb, dwx, dwy = ImageUtils.rgbim2channelfeatures(wimg)
	xr, xg, xb, dxx, dxy = ImageUtils.rgbim2channelfeatures(ximg)
	flag = (dwx == dxx) && (dxx == dxy) && (dwy == dxy)
	m = p * dwx * dwy
	flag || throw(ArgumentError("
		Images need to be square and have the same dimensions!"))
	rProb = BlindDeconvHadamard.gen_problem(m, wr, xr, 0.0)
	gProb = BlindDeconvHadamard.gen_problem(m, wg, xg, 0.0)
	bProb = BlindDeconvHadamard.gen_problem(m, wb, xb, 0.0)
	wRed, xRed = BlindDeconvHadamard.init_method(rProb, init=init)
	wGreen, xGreen = BlindDeconvHadamard.init_method(gProb, init=init)
	wBlue, xBlue = BlindDeconvHadamard.init_method(bProb, init=init)
	# save initializations
	rgbw = ImageUtils.rescale_from_chan(wRed, wGreen, wBlue, dwx, dwy)
	rgbx = ImageUtils.rescale_from_chan(xRed, xGreen, xBlue, dwx, dwy)
	Images.save(@sprintf("init_w_%s.jpg", init), rgbw)
	Images.save(@sprintf("init_x_%s.jpg", init), rgbx)
	# setup error histories
	rHist = fill(0.0, 0); gHist = fill(0.0, 0); bHist = fill(0.0, 0)
	if save_first   # save all images for the first [num_saves] iterations
		for k = 1:num_saves
			println(@sprintf("Iteration: %d", k))
			wRed, xRed, er = BlindDeconvHadamard.subgradient_solver_noinit(
				rProb, 1.0, 1.0, 1, polyak_step=true, wk=wRed, xk=xRed)
			wGreen, xGreen, eg = BlindDeconvHadamard.subgradient_solver_noinit(
				gProb, 1.0, 1.0, 1, polyak_step=true, wk=wGreen, xk=xGreen)
			wBlue, xBlue, eb = BlindDeconvHadamard.subgradient_solver_noinit(
				bProb, 1.0, 1.0, 1, polyak_step=true, wk=wBlue, xk=xBlue)
			append!(rHist, er[:]); append!(gHist, eg[:]); append!(bHist, eb[:])
			# save iterates
			rgbw = ImageUtils.rescale_from_chan(wRed, wGreen, wBlue, dwx, dwy)
			rgbx = ImageUtils.rescale_from_chan(xRed, xGreen, xBlue, dwx, dwy)
			Images.save(@sprintf("w_iter_%d_%s.jpg", k, init), rgbw)
			Images.save(@sprintf("x_iter_%d_%s.jpg", k, init), rgbx)
		end
		# run for the rest of the iterations
		remn = iters - num_saves
		wRed, xRed, er = BlindDeconvHadamard.subgradient_solver_noinit(
			rProb, 1.0, 1.0, remn, polyak_step=true, wk=wRed, xk=xRed)
		wGreen, xGreen, eg = BlindDeconvHadamard.subgradient_solver_noinit(
			gProb, 1.0, 1.0, remn, polyak_step=true, wk=wGreen, xk=xGreen)
		wBlue, xBlue, eb = BlindDeconvHadamard.subgradient_solver_noinit(
			bProb, 1.0, 1.0, remn, polyak_step=true, wk=wBlue, xk=xBlue)
		append!(rHist, er[:]); append!(gHist, eg[:]); append!(bHist, eb[:])
	else  # divide iterations into [num_saves] groups
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
			@show er[end], eg[end], eb[end]
			# save iterates
			rgbw = ImageUtils.rescale_from_chan(wRed, wGreen, wBlue, dwx, dwy)
			rgbx = ImageUtils.rescale_from_chan(xRed, xGreen, xBlue, dwx, dwy)
			Images.save(@sprintf("w_iter_%d_%s.jpg", k, init), rgbw)
			Images.save(@sprintf("x_iter_%d_%s.jpg", k, init), rgbx)
		end
	end
	# equalize history lengths
	maxLen = maximum(length.((rHist, gHist, bHist)))
	append!(rHist, fill(0.0, maxLen - length(rHist)))
	append!(gHist, fill(0.0, maxLen - length(gHist)))
	append!(bHist, fill(0.0, maxLen - length(bHist)))
	df = DataFrame(iter=collect(1:maxLen), err_red=rHist,
				   err_green=gHist, err_blue=bHist)
	fname = @sprintf("img_errors_%dx%d.csv", dwx, dwy)
	CSV.write(fname, df)
end


function main()
	# parse arguments
	s = ArgParseSettings(description="""
						 Given the paths of two square images of identical size,
						 sets up a blind deconvolution problem instance using
						 Hadamard matrices and solves it using spectral
						 initialization and the subgradient method.""")
	@add_arg_table s begin
		"--imgw"
			help = "The path of the first image"
			arg_type = String
		"--imgx"
			help = "The path of the second image"
			arg_type = String
		"--k"
			help = "The number of sign pattern matrices to use"
			arg_type = Int
			default = 10
		"--seed"
			help = "The seed of the RNG"
			arg_type = Int
			default = 999
		"--iters"
			help = "The number of iterations for minimization"
			arg_type = Int
			default = 1000
		"--init_method"
			help = "The init method to use"
			arg_type = String
			default = "direction"
		"--num_saves"
			help = "The number of (equispaced) iterates to save"
			arg_type = Int
			default = 9
		"--save_first"
			help = """
				Set to only save the first [num_saves] iterations, where
				the visual error is non-negligible
				"""
			action = :store_true
	end
	parsed = parse_args(s)
	pathw, pathx, k = parsed["imgw"], parsed["imgx"], parsed["k"]
	rnd_seed = parsed["seed"]; Random.seed!(rnd_seed)  # set random seed
	iters, init = parsed["iters"], parsed["init_method"]
	# solve the problem
	recoverImgs(pathw, pathx, k, iters, init,
				num_saves=parsed["num_saves"], save_first=parsed["save_first"])
end

main()
