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
using LinearAlgebra
using Logging
using Printf
using Random
using Statistics


"""
    computeRecProb(λ, m, d, T, ϵ, reps)

Compute the empirical recovery probability for a given configuration of
parameters with threshold `ϵ` and `reps` repeats.
"""
function computeRecProb(λ, m, d, T, ϵ, reps)
    succRep = fill(0, reps)
    for rep = 1:reps
        @info("[m] - $(m), [λ] - $(λ), [it]: $(rep)")
        prob = BlindDeconv.genCoherentProblem(d, m, λ)
        _, _, ds = BlindDeconv.subgradientMethod(prob, 1.0, 1.0, T,
                                                 use_polyak=true)
        succRep[rep] = trunc(Int, ds[end] < ϵ)
    end
    return mean(succRep)
end


"""
    mainLoop(iMax, d, T, ϵ, reps, λLength)

Run the main loop, with the number of measurements ranging from `2d` to
`iMax * 2d`. The results are stored in a .csv file called
"pdft_recovery_[d].csv", where `[d]` is the number used for the dimension `d`.
"""
function mainLoop(iMax, d, T, ϵ, reps, λLength)
    λRng = range(0.05, stop=1.0, length=λLength)
    df   = DataFrame(i=Int64[], lambda=[], succ=[])
    for i = 1:iMax
        m = i * 2 * d
        for (idx, λ) in enumerate(λRng)
            push!(df, (i, λ, computeRecProb(λ, m, d, T, ϵ, reps)))
        end
    end
    CSV.write("pdft_recovery_$(d).csv", df)
end

function main()
    # parse arguments
    s = ArgParseSettings(description="""
                         Generates a set of synthetic problem instances for
                         a given ratio m / (d_1 + d_2) and a range of incoherence
                         parameters, and solves them using subgradient descent
                         outputting the percentage of successful recoveries.""")
    @add_arg_table s begin
        "--dim"
            help = "The problem dimension"
            arg_type = Int
            default = 100
        "--seed"
            help = "The seed of the RNG"
            arg_type = Int
            default = 999
        "--iters"
            help = "The number of iterations for minimization"
            arg_type = Int
            default = 500
        "--repeats"
            help = "The number of repeats for generating success rates"
            arg_type = Int
            default = 100
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
        "--iMax"
            help = "The maximum ratio of measurements to d_1 + d_2"
            arg_type = Int
            default = 8
    end
    parsed  = parse_args(s); Random.seed!(parsed["seed"])
    d, T, i = parsed["dim"], parsed["iters"], parsed["iMax"]
    ϵ, reps = parsed["success_dist"], parsed["repeats"]
    λLength = parsed["lambda_length"]
    # seed RNG
    mainLoop(i, d, T, ϵ, reps, λLength)
end

main()
