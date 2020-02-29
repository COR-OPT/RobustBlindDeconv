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
    fixedStepChoice(prob::BlindDeconv.BCProb, T; ϵ)

Try a sequence of fixed stepsizes (2^(-1) to 2^(-10)) and select the one giving
the lower number of iterations elapsed to converge to accuracy `ϵ`, returning
the iterate distance history for that choice.
"""
function fixedStepChoice(prob::BlindDeconv.BCProb, T; ϵ)
    # for a fixed problem, try all settings
    finalIt = fill(0, 15)
    for idx in 1:length(finalIt)
        η = 2.0^(-idx)
        _, _, dsSm = BlindDeconv.gradientMethod(prob, η, T, ϵ=ϵ, use_polyak=false)
        finalIt[idx] = length(dsSm)
    end
    finalIt = map(x -> isnan(x) ? Inf : x, finalIt)
    bestIdx = argmin(finalIt); ηBest= 2.0^(-bestIdx)
    return BlindDeconv.gradientMethod(prob, ηBest, T, ϵ=ϵ, use_polyak=false)
end

"""
    computeRecProb(λ, m, d, T, ϵ, reps)

Compute the empirical recovery probability for a given configuration of
parameters with threshold `ϵ` and `reps` repeats.
"""
function computeIters(λ, m, d, ϵ, reps)
    itersSm = fill(0, reps); itersNs = fill(0, reps); itersFx = fill(0, reps)
    for rep = 1:reps
        @info("[m] - $(m), [λ] - $(λ), [it]: $(rep)")
        prob = BlindDeconv.genCoherentProblem(d, m, λ, random_init=true)
        _, _, dsSm = BlindDeconv.gradientMethod(prob, 1.0, 2500, ϵ=ϵ)
        _, _, dsNs = BlindDeconv.subgradientMethod(prob, 1.0, 1.0, 2500,
                                                   use_polyak=true, ϵ=ϵ)
        _, _, dsFx = fixedStepChoice(prob, 2500, ϵ=ϵ)
        itersSm[rep] = length(dsSm)
        itersNs[rep] = length(dsNs)
        itersFx[rep] = length(dsFx)
    end
    return mean(itersSm), mean(itersFx), mean(itersNs), std(itersSm), std(itersFx), std(itersNs)
end


"""
    mainLoop(i, d, ϵ, reps, λLength)

Run the main loop, ranging incoherence from 0.01 to 1.0 via equispaced
intervals using `i * 2d` measurements. The results are stored in a .csv file
called "pdft_iters_[d].csv", where `[d]` is the number used for the dimension
`d`.
"""
function mainLoop(i, d, ϵ, reps, λLength)
    λRng = range(0.01, stop=1.0, length=λLength)
    df   = DataFrame(mu=Int[],
                     itMeanSm=Float64[], itMeanFx=Float64[], itMeanNs=Float64[],
                     itStdSm=Float64[], itStdFx=Float64[], itStdNs=Float64[])
    m    = i * 2 * d
    for λ in λRng
        nNnz = trunc(Int, ceil(λ * d))
        # compute mean / std of iterations to reach accuracy ε
        meanSm, meanFx, meanNs, stdSm, stdFx, stdNs = computeIters(λ, m, d, ϵ, reps)
        push!(df, (nNnz, meanSm, meanFx, meanNs, stdSm, stdFx, stdNs))
    end
    CSV.write("pdft_iters_$(d).csv", df)
end

function main()
    # parse arguments
    s = ArgParseSettings(description="""
                         Generates a set of synthetic problem instances for
                         a given ratio m / (d_1 + d_2) and a range of incoherence
                         parameters, and compute the average +/- stddev of the
                         number of iterations required to reach a given level
                         of accuracy for the Polyak subgradient, Polyak gradient,
                         and fixed-step gradient methods.""")
    @add_arg_table s begin
        "--dim"
            help = "The problem dimension"
            arg_type = Int
            default = 100
        "--seed"
            help = "The seed of the RNG"
            arg_type = Int
            default = 999
        "--repeats"
            help = "The number of repeats for generating statistics"
            arg_type = Int
            default = 25
        "--success_dist"
            help = """The desired reconstruction distance. Iterates whose
                   normalized distance is below this threshold are considered
                   exact recoveries."""
            arg_type = Float64
            default = 1e-5
        "--lambda_length"
            help = "The number of equally spaced lambdas to use in the interval [0.0, 1.0]"
            arg_type = Int
            default  = 10
        "--i"
            help = "The ratio of measurements to d_1 + d_2"
            arg_type = Int
            default = 8
    end
    parsed  = parse_args(s); Random.seed!(parsed["seed"])
    d, i, ϵ = parsed["dim"], parsed["i"], parsed["success_dist"]
    reps    = parsed["repeats"]
    λLength = parsed["lambda_length"]
    # seed RNG
    mainLoop(i, d, ϵ, reps, λLength)
end

main()
