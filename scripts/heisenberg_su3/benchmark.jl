using TensorKitBenchmarks
using TOML
using ArgParse
using LinearAlgebra: BLAS

# Parameters
# ----------
repetitions = 5
maxdims = Dict("SU3" => typemax(Int), "U1xU1" => 16000, "Trivial" => 2000)

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1

# Utility functions
# -----------------

# Argument parser
# ---------------
const filename_in = joinpath(@__DIR__, "heisenberg_sun_spaces.toml")
const filename_out = joinpath(@__DIR__, "heisenberg_sun_results")
const eltypes = (Float64,)

const argparser = ArgParseSettings()
@add_arg_table! argparser begin
    "--blasthreads", "-b"
    help = "number of threads given to BLAS"
    arg_type = Int
    default = Threads.nthreads()
    "--in"
    help = "input file"
    arg_type = String
    default = filename_in
    "--out"
    help = "output file"
    arg_type = String
    default = filename_out
    "--mkl"
    help = "using MKL"
    action = :store_true
end

# Script
# ------
function (@main)(ARGS=[])
    settings = parse_args(ARGS, argparser)

    if settings["mkl"]
        @eval Main using MKL
        @info "using MKL"
    end
    BLAS.set_num_threads(settings["blasthreads"])
    @info "using BLAS threads: $(BLAS.get_num_threads())"

    SUITE = BenchmarkGroup()

    isfile(settings["in"]) || error("invalid input file $(settings["in"])")
    allparams = open(settings["in"], "r") do io
        return TOML.parse(io)
    end

    suite_ac = addgroup!(SUITE, "AC")
    for (kind, params) in allparams
        g = addgroup!(suite_ac, kind)
        for paramset in params
            for T in eltypes
                generate_AC_benchmark!(g, kind, T, paramset; maxdims)
            end
        end
    end
    suite_ac2 = addgroup!(SUITE, "AC2")
    for (kind, params) in allparams
        g = addgroup!(suite_ac2, kind)
        for paramset in params
            for T in eltypes
                generate_AC2_benchmark!(g, kind, T, paramset; maxdims)
            end
        end
    end

    results = run(SUITE; verbose=true)

    file_out = settings["out"] * "_b$(settings["blasthreads"]).json"
    BenchmarkTools.save(file_out, results)
end

