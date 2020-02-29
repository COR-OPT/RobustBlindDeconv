#!/usr/bin/env julia

#=
Generates a set of synthetic problem instances for a given ratio
m / (d₁ + d₂) and a range of failure probabilities. Generates 100 instances
with the same parameters and records the percentage of successful recoveries.
=#

include("../src/BlindDeconv.jl")

using ArgParse
using CSV
using DataFrames
using FFTW
using LinearAlgebra
using Logging
using Printf
using Random
using Statistics


"""
    fixedStepChoice(prob::BlindDeconv.BCProb, T, ϵ=1e-15)

Try a sequence of fixed stepsizes (2^(-1) to 2^(-10)) and select the one giving
the best final normalized distance, returning the iterate distance history for
that choice.
"""
function fixedStepChoice(prob::BlindDeconv.BCProb, T, ϵ=1e-15)
    # for a fixed problem, try all settings
    finalDs = fill(0.0, 15)
    for idx in 1:length(finalDs)
        η = 2.0^(-idx)
        _, _, dsSm = BlindDeconv.gradientMethod(prob, η, T, ϵ=ϵ, use_polyak=false)
        finalDs[idx] = dsSm[end]
    end
    finalDs = map(x -> isnan(x) ? Inf : x, finalDs)
    bestIdx = argmin(finalDs); ηBest = 2.0^(-bestIdx)
    _, _, dsSm = BlindDeconv.gradientMethod(prob, ηBest, T, ϵ=ϵ, use_polyak=false)
    return dsSm
end

function vpad(z, n)
    return vcat(z, fill(0.0, n - length(z)))
end

"""
    computeConvergence(λ, m, d, T)

For a given incoherence level `λ`, `m` measurements and dimension `d`, run all 3
methods (nonsmooth / smooth Polyak, smooth with best constant stepsize) and save
the iterate distance history in .CSV files.
"""
function computeConvergence(λ, m, d, T)
    ϵ = 1e-15 # run until end
    prob = BlindDeconv.genCoherentProblem(d, m, λ, random_init=true)
    _, _, dsSmPolyak = BlindDeconv.gradientMethod(prob, 1.0, T, ϵ=ϵ, use_polyak=true)
    dsSmFixedStep    = fixedStepChoice(prob, T, ϵ)
    _, _, dsNsPolyak = BlindDeconv.subgradientMethod(prob, 1.0, 1.0, T, use_polyak=true, ϵ=ϵ)
    maxLen = max(length.((dsSmPolyak, dsSmFixedStep, dsNsPolyak))...)
    CSV.write("convergence_$(d)_$(@sprintf("%.2f", λ)).csv",
              DataFrame(k=1:maxLen, nsPolyak=vpad(dsNsPolyak, maxLen),
                        smPolyak=vpad(dsSmPolyak, maxLen),
                        smFixed=vpad(dsSmFixedStep, maxLen)))
end


"""
    mainLoop(i, d, ϵ, reps, λLength)

Run the main loop, ranging incoherence from 0.01 to 1.0 via equispaced
intervals using `i * 2d` measurements. The results are stored in a .csv file
called "pdft_iters_[d].csv", where `[d]` is the number used for the dimension
`d`.
"""
function mainLoop(m, d, T, λLength)
    λRng = range(0.01, stop=1.0, length=λLength)
    for λ in λRng
        @info("Running for λ=$(λ)")
        computeConvergence(λ, m, d, T)
    end
end

function main()
    # parse arguments
    s = ArgParseSettings(description="""
                         Generates a set of synthetic problem instances for a
                         given dimension and number of measurements, and stores
                         the iterate distance history for Polyak subgradient,
                         Polyak gradient, and fixed-step gradient methods for
                         the nonsmooth and smooth formulations of the problem.""")
    @add_arg_table s begin
        "--dim"
            help     = "The problem dimension"
            arg_type = Int
            default  = 100
        "--seed"
            help     = "The seed of the RNG"
            arg_type = Int
            default  = 999
        "--iters"
            help     = "The number of iterations"
            arg_type = Int
            default  = 1000
        "--lambda_length"
            help     = "The number of equally spaced lambdas to use in the interval [0.0, 1.0]"
            arg_type = Int
            default  = 10
        "--i"
            help     = "The ratio of measurements to d_1 + d_2"
            arg_type = Int
            default  = 8
    end
    parsed  = parse_args(s); Random.seed!(parsed["seed"])
    d, i, T = parsed["dim"], parsed["i"], parsed["iters"]
    λLength = parsed["lambda_length"]
    # seed RNG
    mainLoop(i * 2 * d, d, T, λLength)
end

main()
