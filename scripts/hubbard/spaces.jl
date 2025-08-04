using TensorKitBenchmarks

using TensorKit: dim
using MPSKitModels
using TOML
using MKL

using LinearAlgebra
using TensorOperations.Strided: Strided
using TensorKitTensors.HubbardOperators

LinearAlgebra.BLAS.set_num_threads(16)
MPSKit.Defaults.set_scheduler!(:serial)
Strided.disable_threads()

# Parameters
# ----------

const T = ComplexF64
spin_symmetry = particle_symmetry = SU2Irrep

verbosity = 3
tol = 1e-5
maxiter = 20

filename = joinpath(@__DIR__, "hubbard_spaces")
# L = 6
# alg1 = DMRG2(; tol=1e-4, maxiter=20, trscheme=truncdim(128))
# alg2 = DMRG(; tol, maxiter, verbosity);
# t = 1
# U = 1

# begin
# spin_symmetry = particle_symmetry = U1Irrep
# H = hubbard_model(T, particle_symmetry, spin_symmetry, FiniteChain(L); t, U, n=1, mu=0);
# psi_finite = insertleftunit(insertleftunit(randn(scalartype(H), prod(physicalspace(H))), 1), L + 2);
# As = MPSKit.decompose_localmps(psi_finite);
# psi0 = FiniteMPS(As);

# psi, envs, = find_groundstate(psi0, H, alg1);
# psi, envs, = find_groundstate(psi, H, alg2);
# E1 = expectation_value(psi, H, envs) 
# end

# begin
# spin_symmetry = particle_symmetry = SU2Irrep
# H = hubbard_model(T, particle_symmetry, spin_symmetry, FiniteChain(L); t, U, n=1);
# psi_finite = insertleftunit(insertleftunit(randn(scalartype(H), prod(physicalspace(H))), 1), L + 2);
# As = MPSKit.decompose_localmps(psi_finite);
# psi0 = FiniteMPS(As);

# psi, envs, = find_groundstate(psi0, H, alg1);
# psi, envs, = find_groundstate(psi, H, alg2);
# E2 = expectation_value(psi, H, envs)
# end


# Script
# ------
function (@main)(ARGS=[])
    H = hubbard_model(T, particle_symmetry, spin_symmetry, InfiniteChain(2); t=1, U=1, n=1)

    O = make_time_mpo(H, 0.1, TaylorCluster(; N =1))

    psi_finite = insertleftunit(insertleftunit(randn(scalartype(H), prod(physicalspace(H))), 1), 4)
    As = MPSKit.decompose_localmps(psi_finite)
    psi0 = InfiniteMPS(As)

    alg = VUMPS(; tol, maxiter, verbosity)
    psi, envs, = find_groundstate(psi0, H, alg)

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
    alg_conv = VUMPS(; tol=1e-7, maxiter=100, verbosity)
    psi, envs, = find_groundstate(psi, H, alg_conv)

    # spaces_su2 = Dict{String,spacetype(H)}()
    spaces_su2xsu2 = []
    for D in reverse(Ds)
        psi = changebonds(psi, SvdCut(; trscheme=truncdim(D)))
        psi, envs, = find_groundstate(psi, H, alg_conv)
        # spaces_su2[string(D)] = left_virtualspace(psi, 1)
        push!(spaces_su2xsu2, summarize_mps_spaces(psi, H))
    end

    spaces_u1xsu2 = map(spaces_su2xsu2) do dict
        return Dict(k => desymmetrize(Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep], v) for (k, v) in dict)
    end
    spaces_u1xu1 = map(spaces_su2xsu2) do dict
        return Dict(k => desymmetrize(Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep], v) for (k, v) in dict)
    end
    spaces_trivial = map(spaces_su2xsu2) do dict
        return Dict(k => desymmetrize(Vect[FermionParity], v) for (k, v) in dict)
    end

    open(filename * ".toml", "w") do io
        TOML.print(tomlify, io, Dict("fZ2xSU2xSU2" => spaces_su2xsu2, "fZ2xU1xSU2" => spaces_u1xsu2, "fZ2xU1xU1" => spaces_u1xu1, "fZ2" => spaces_trivial))
    end

    return 1
end
