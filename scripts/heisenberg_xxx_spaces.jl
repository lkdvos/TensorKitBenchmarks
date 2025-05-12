using TensorKit
using TensorKit: dim
using MPSKit
using MPSKitModels
using TOML

# Utility functions
# -----------------

function summarize_spaces(psi, H)
    Vmps = left_virtualspace(psi)
    Vmpo = map(splat(TensorKit.oplus), left_virtualspace(H))
    return Dict("virtual_mps" => Vmps, "virtual_mpo" => Vmpo, "physical" => physicalspace(psi))
end

function to_u1space(V::SU2Space)
    u1_dims = TensorKit.SectorDict{U1Irrep,Int}()
    for c in sectors(V)
        d = dim(V, c)
        for j in -(c.j):c.j
            u1_dims[U1Irrep(j)] = get(u1_dims, U1Irrep(j), 0) + d
        end
    end
    return U1Space(u1_dims)
end
to_u1space(d::Dict) = Dict(k => to_u1space.(v) for (k, v) in d)

to_trivialspace(V::SU2Space) = ComplexSpace(dim(V))
to_trivialspace(d::Dict) = Dict(k => to_trivialspace.(v) for (k, v) in d)

tomlify(c::SU2Irrep) = dim(c)
tomlify(c::U1Irrep) = Int(2 * c.charge)
tomlify(V::ElementarySpace) = Dict("sectors" => collect(sectors(V)), "dims" => dim.(Ref(V), sectors(V)))
tomlify(V::ComplexSpace) = dim(V)

# Parameters
# ----------

T = Float64
symmetry = SU2Irrep
Ds = round.(Int, logrange(2, 2^14, 24))
filename = joinpath(@__DIR__, "heisenberg_spaces.toml")

# Script
# ------
function (@main)(ARGS=[])
    H = heisenberg_XXX(T, symmetry, InfiniteChain(2); spin=1 // 2)

    psi_finite = insertleftunit(insertleftunit(rand(scalartype(H), prod(physicalspace(H))), 1), 4)
    As = MPSKit.decompose_localmps(psi_finite)
    psi_init = InfiniteMPS(As)

    heisenberg_spaces = Dict("su2" => Dict[], "u1" => Dict[], "trivial" => Dict[])
    D, Drest... = Ds
    alg = IDMRG2(; tol=1e-5, trscheme=truncdim(D), maxiter=30)
    psi, envs, = find_groundstate(psi_init, H, alg)

    d_su2 = summarize_spaces(psi, H)
    push!(heisenberg_spaces["su2"], d_su2)
    push!(heisenberg_spaces["u1"], to_u1space(d_su2))
    push!(heisenberg_spaces["trivial"], to_trivialspace(d_su2))

    for D in Drest
        alg = IDMRG2(; tol=1e-5, trscheme=truncdim(D), maxiter=30)
        psi, envs, = find_groundstate(psi, H, alg)

        d_su2 = summarize_spaces(psi, H)
        push!(heisenberg_spaces["su2"], d_su2)
        push!(heisenberg_spaces["u1"], to_u1space(d_su2))
        push!(heisenberg_spaces["trivial"], to_trivialspace(d_su2))
    end

    open(filename, "w") do io
        TOML.print(tomlify, io, heisenberg_spaces)
    end
end

