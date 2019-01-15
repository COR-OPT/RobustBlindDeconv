#!/usr/bin/env julia

#=
A script to test using Mnist images as the true signals in a blind
deconvolution problem.
=#

include("../src/BlindDeconv.jl")
include("ImageUtils.jl")

using ArgParse
using CSV
using DataFrames
using LinearAlgebra
using MLDatasets
using ImageMagick
using Images
using Printf
using Random
using Statistics

#= Load a pair of images and set up a corresponding blind deconv. problem =#
function solve_subgradient(
	idx_w::Int, idx_x::Int, p::Int, iters::Int, noise_lvl, q)
	train_x, train_y = MNIST.traindata()  # load images
	w_img = Array(train_x[:, :, idx_w]')
	x_img = Array(train_x[:, :, idx_x]')
	w_lbl, x_lbl = train_y[idx_w], train_y[idx_x]  # labels
	w_sig = ImageUtils.gray2features(w_img)
	x_sig = ImageUtils.gray2features(x_img)
	wx, wy = size(w_img); xx, xy = size(x_img)
	d1, d2 = wx * wy, xx * xy
	# create save path
	save_path = string("mnist_data/mnist_", idx_w, "_", idx_x, "_")
	# setup problem
	prob = BlindDeconv.gen_problem(p * (d1 + d2), w_sig, x_sig,
		BlindDeconv.gaussian, BlindDeconv.gaussian, noise_lvl)
	w_init = ImageUtils.features2gray(prob.w0, wx, wy)
	x_init = ImageUtils.features2gray(prob.x0, xx, xy)
	# save init images and original images
	Images.save(string(save_path, "w_orig.jpg"), w_img)
	Images.save(string(save_path, "x_orig.jpg"), x_img)
	Images.save(string(save_path, "w_init.jpg"), w_init)
	Images.save(string(save_path, "x_init.jpg"), x_init)
	div_epochs = Int(ceil(iters / 9))
	error_hist = fill(0.0, 0);
	for k = 1:9
		# run subgradient method for [div_epochs]
		# set exponent properly
		qs = q^((k - 1) * div_epochs)
		wk, xk, err = BlindDeconv.subgradient_method(prob, qs, q, div_epochs)
		@show err[end]
		append!(error_hist, err[:])  # append error history
		copyto!(prob.w0, wk); copyto!(prob.x0, xk)  # renew estimates
		# save iterates
		Images.save(string(save_path, "w_", k, ".jpg"),
					ImageUtils.features2gray(wk, wx, wy))
		Images.save(string(save_path, "x_", k, ".jpg"),
					ImageUtils.features2gray(xk, xx, xy))
	end
	df = DataFrame(iter=collect(1:length(error_hist)), err=error_hist)
	CSV.write(string("mnist_data/mnist_error_", idx_w, "_", idx_x, ".csv"), df)
end

function solve_proximal(
	idx_w::Int, idx_x::Int, p::Int, iters::Int, noise_lvl)
	train_x, train_y = MNIST.traindata()  # load images
	w_img = Array(train_x[:, :, idx_w]')
	x_img = Array(train_x[:, :, idx_x]')
	w_lbl, x_lbl = train_y[idx_w], train_y[idx_x]  # labels
	w_sig = ImageUtils.gray2features(w_img)
	x_sig = ImageUtils.gray2features(x_img)
	wx, wy = size(w_img); xx, xy = size(x_img)
	d1, d2 = wx * wy, xx * xy
	# setup problem
	prob = BlindDeconv.gen_problem(p * (d1 + d2), w_sig, x_sig,
		BlindDeconv.gaussian, BlindDeconv.gaussian, noise_lvl)
	wk, xk, error_hist = BlindDeconv.proximal_method(prob, 1.0, iters)
	df = DataFrame(iter=collect(1:length(error_hist)), err=error_hist)
	CSV.write(string("mnist_data/mnist_error_proximal_",
					 idx_w, "_", idx_x, ".csv"), df)
end

function main()
	# parse arguments
	s = ArgParseSettings(description="""
						 Given the indices of two MNIST digits in the training
						 set, sets up a blind deconvolution problem instance
						 Gaussian matrices and solves it using spectral
						 initialization and either the subgradient or proximal
						 methods. It also stores the images and the normalized
						 error of the iterates.""")
	@add_arg_table s begin
		"--idx_w"
			help = "The index of the first digit [1-60000]"
			arg_type = Int
		"--idx_x"
			help = "The index of the second digit [1-60000]"
			arg_type = Int
		"--c"
			help = "The coefficient `c` of `m = c * (d_1 + d_2)`"
			arg_type = Int
			default = 4
		"--seed"
			help = "The seed of the RNG"
			arg_type = Int
			default = 999
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
			default = 0.0
		"--q"
			help = "The step size decay parameter"
			arg_type = Float64
			default = 0.98
	end
	parsed = parse_args(s)
	idx_w, idx_x = parsed["idx_w"], parsed["idx_x"]
	c, iters, q = parsed["c"], parsed["iters"], parsed["q"]
	noise = parsed["noise"]
	rnd_seed = parsed["seed"]; Random.seed!(rnd_seed)  # set random seed
	method = parsed["method"]
	if method == "subgradient"
		solve_subgradient(idx_w, idx_x, c, iters, noise, q)
	else
		solve_proximal(idx_w, idx_x, c, iters, noise)
	end
end

main()
