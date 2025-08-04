using TensorKitBenchmarks
using TensorKit: dim, ×
using SUNRepresentations
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
const irrep = SU3Irrep("8")

verbosity = 3
tol = 1e-5
maxiter = 20

filename = joinpath(@__DIR__, "heisenberg_sun_spaces")

# Script
# ------
function (@main)(ARGS=[])
    H = sun_heisenberg(T, InfiniteChain(1); irrep)
    O = make_time_mpo(H, 0.1, TaylorCluster(; N =1))
    psi0 = InfiniteMPS(physicalspace(H), physicalspace(H))

    alg = VUMPS(; tol, maxiter, verbosity)
    psi, envs, = find_groundstate(psi0, H, alg);

    Ds = Int[]
    Δt = 0
    while Δt < 1000
        D_current = dim(left_virtualspace(psi, 1))
        D = max(64, round(Int, D_current * 1.5))
        push!(Ds, D)
        @info "Expanding from $D_current to $D"
        trscheme = truncdim(D - D_current)
        psi_tmp, envs, = changebonds(psi, H, OptimalExpand(; trscheme), envs)
        psi, = approximate(psi_tmp, (O, psi), VOMPS(; maxiter=3, verbosity))
        t0 = Base.time()
        psi, envs, = find_groundstate(psi, H, alg)
        Δt = Base.time() - t0
    end

    @info "Reached maximal expansion -- converging"
    alg_conv = VUMPS(; tol=1e-8, maxiter=100, verbosity)
    psi, envs, = find_groundstate(psi, H, alg_conv)

    # spaces_su2 = Dict{String,spacetype(H)}()
    spaces_su3 = []
    for D in reverse(Ds)
        psi = changebonds(psi, SvdCut(; trscheme = truncdim(D)))
        psi, envs, = find_groundstate(psi, H, alg_conv)
        # spaces_su2[string(D)] = left_virtualspace(psi, 1)
        push!(spaces_su3, summarize_mps_spaces(psi, H))
    end

    spaces_u1_u1 = map(spaces_su3) do dict
        return Dict(k => desymmetrize(Rep[U₁ × U₁], v) for (k, v) in dict)
    end
    spaces_trivial = map(spaces_su3) do dict
        return Dict(k => desymmetrize(ComplexSpace, v) for (k, v) in dict)
    end

    open(filename * ".toml", "w") do io
        TOML.print(tomlify, io, Dict("SU3" => spaces_su3, "U1xU1" => spaces_u1_u1, "Trivial" => spaces_trivial))
    end

    return 1
end
