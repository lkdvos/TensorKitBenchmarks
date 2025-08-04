using TensorKitBenchmarks

using TensorKit: dim
using MPSKitModels
using TOML
using MKL

using LinearAlgebra
using TensorOperations.Strided: Strided

LinearAlgebra.BLAS.set_num_threads(16)
MPSKit.Defaults.set_scheduler!(:serial)
Strided.disable_threads()

# Parameters
# ----------

const T = ComplexF64
const symmetry = SU2Irrep

verbosity = 3
tol = 1e-5
maxiter = 20

Ds = round.(Int, logrange(8, 2^14, 32))
filename = joinpath(@__DIR__, "heisenberg_spaces")

# Script
# ------
function (@main)(ARGS=[])
    H = heisenberg_XXX(T, symmetry, InfiniteChain(1); spin=1)
    O = make_time_mpo(H, 0.1, TaylorCluster(; N =1))
    psi0 = InfiniteMPS(physicalspace(H), [SU2Space(1//2 => 5, 3//2 => 3, 5//2 => 1)])

    alg = VUMPS(; tol, maxiter, verbosity)
    psi, envs, = find_groundstate(psi0, H, alg);

    for D in Ds
        D_current = dim(left_virtualspace(psi, 1))
        D ≤ D_current && continue
        @info "Expanding from $D_current to $D"
        trscheme = truncdim(D - D_current)
        psi_tmp, envs, = changebonds(psi, H, OptimalExpand(; trscheme), envs)
        psi, = approximate(psi_tmp, (O, psi), VOMPS(; maxiter=3, verbosity))
        psi, envs, = find_groundstate(psi, H, alg)
    end

    @info "Reached maximal expansion -- converging"
    alg_conv = VUMPS(; tol=1e-8, maxiter=100, verbosity)
    psi, envs, = find_groundstate(psi, H, alg_conv)

    # spaces_su2 = Dict{String,spacetype(H)}()
    spaces_su2 = []
    for D in reverse(Ds)
        psi = changebonds(psi, SvdCut(; trscheme = truncdim(D)))
        psi, envs, = find_groundstate(psi, H, alg_conv)
        # spaces_su2[string(D)] = left_virtualspace(psi, 1)
        push!(spaces_su2, summarize_mps_spaces(psi, H))
    end

    spaces_u1 = map(spaces_su2) do dict
        return Dict(k => desymmetrize(U1Space, v) for (k, v) in dict)
    end
    spaces_trivial = map(spaces_su2) do dict
        return Dict(k => desymmetrize(ComplexSpace, v) for (k, v) in dict)
    end

    open(filename * ".toml", "w") do io
        TOML.print(tomlify, io, Dict("SU2" => spaces_su2, "U1" => spaces_u1, "Trivial" => spaces_trivial))
    end

    return 1
end
