using BenchmarkTools
using TensorKit
using TensorKit: dim
using LinearAlgebra.BLAS: BLAS
using TOML
using ArgParse

# Utility functions
# -----------------

trivialspace(d::Int) = ComplexSpace(d)
u1space(d::Dict) = u1space(d["sectors"], d["dims"])
u1space(sectors, dims) = U1Space(U1Irrep(c / 2) => d for (c, d) in zip(sectors, dims))
su2space(d::Dict) = su2space(d["sectors"], d["dims"])
su2space(sectors, dims) = SU2Space(SU2Irrep((c - 1) / 2) => d for (c, d) in zip(sectors, dims))

function instantiate_su2_spaces(d::Dict)
    issetequal(keys(d), ("virtual_mps", "virtual_mpo", "physical")) || error("invalid dict")
    Vmps = su2space.(d["virtual_mps"])
    Vmpo = su2space.(d["virtual_mpo"])
    Vphys = su2space.(d["physical"])
    return Vphys, Vmps, Vmpo
end
function instantiate_u1_spaces(d::Dict)
    issetequal(keys(d), ("virtual_mps", "virtual_mpo", "physical")) || error("invalid dict")
    Vmps = u1space.(d["virtual_mps"])
    Vmpo = u1space.(d["virtual_mpo"])
    Vphys = u1space.(d["physical"])
    return Vphys, Vmps, Vmpo
end
function instantiate_trivial_spaces(d::Dict)
    issetequal(keys(d), ("virtual_mps", "virtual_mpo", "physical")) || error("invalid dict")
    Vmps = trivialspace.(d["virtual_mps"])
    Vmpo = trivialspace.(d["virtual_mpo"])
    Vphys = trivialspace.(d["physical"])
    return Vphys, Vmps, Vmpo
end

function init_tensors(T, Vphys, Vmps, Vmpo)
    A = randn(T, Vmps[1] ⊗ Vphys[1] ⊗ Vmps[2]')
    M = randn(T, Vmpo[1] ⊗ Vphys[1] ⊗ Vphys[1]' ⊗ Vmpo[2]')
    FL = randn(T, Vmps[1] ⊗ Vmpo[1]' ⊗ Vmps[1]')
    FR = randn(T, Vmps[2] ⊗ Vmpo[2] ⊗ Vmps[2]')
    return A, M, FL, FR
end

function run_benchmark(A, M, FL, FR)
    return @tensor FL[4, 2, 1] * A[1, 3, 6] * M[2, 5, 3, 7] * conj(A[4, 5, 8]) * FR[6, 7, 8]
end

function generate_benchmark(T, Vphys, Vmps, Vmpo)
    local init
    let T = T, Vphys = Vphys, Vmps = Vmps, Vmpo = Vmpo
        init() = init_tensors(T, Vphys, Vmps, Vmpo)
    end
    @benchmarkable run_benchmark(A, M, FL, FR) setup = ((A, M, FL, FR) = $init())
end

function generate_benchmark!(suite, kind, T, params)
    Vphys, Vmps, Vmpo = if kind == "su2"
        instantiate_su2_spaces(params)
    elseif kind == "u1"
        instantiate_u1_spaces(params)
    elseif kind == "trivial"
        instantiate_trivial_spaces(params)
    else
        error()
    end
    Dphys = dim.(Vphys)
    Dmps = dim.(Vmps)
    Dmpo = dim.(Vmpo)
    suite[(Dphys, Dmps, Dmpo)] = generate_benchmark(T, Vphys, Vmps, Vmpo)
    return suite
end

# Parameters
# ----------
filename_in = joinpath(@__DIR__, "heisenberg_spaces.toml")
filename_out = joinpath(@__DIR__, "heisenberg_results.json")
eltypes = (Float64, ComplexF64)

# Script
# ------

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

function (@main)(ARGS=[])
    settings = parse_args(ARGS, argparser)

    if settings["mkl"]
        @eval Main using MKL
    end
    BLAS.set_num_threads(settings["blasthreads"])

    SUITE = BenchmarkGroup()

    isfile(settings["in"]) || error("invalid input file $(settings["in"])")
    allparams = open(settings["in"], "r") do io
        return TOML.parse(io)
    end

    for (kind, params) in allparams
        g = addgroup!(SUITE, kind)
        for paramset in params
            for T in eltypes
                generate_benchmark!(g, kind, T, paramset)
            end
        end
    end

    results = run(SUITE; verbose=true)

    isfile(settings["out"]) || error("invalid output file $(settings["out"])")
    BenchmarkTools.save(settings["out"], results)
end
