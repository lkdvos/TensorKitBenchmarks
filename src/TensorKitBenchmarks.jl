module TensorKitBenchmarks

using Reexport

export initialize_mps
export tomlify, desymmetrize, summarize_mps_spaces
export generate_AC_benchmark!, generate_AC2_benchmark!
export sun_heisenberg

@reexport using TensorKit
@reexport using MPSKit
@reexport using MPSKitModels
@reexport using BenchmarkTools

using TensorKitTensors.HubbardOperators
using SUNRepresentations
using SUNRepresentations: rowsum

# Models
# ------

function get_column_numbers(s::SUNIrrep{N}) where {N}
    col_ind = Int64[]
    append!(col_ind, N * ones(Int64, s.I[N]))
    for x in reverse(1:N-1)
        append!(col_ind, x * ones(Int64, s.I[x] - s.I[x+1]))
    end
    return col_ind
end

"""
    casimir(s::SUNIrrep) -> Float64

Return the (quadratic) Casimir operator for the irrep `s` of SU(N), defined as
``C = 1/2 ( p(N - p/N) + sum_{row} l_r^2 - sum_{col} l_c^2)`` where `p` is the number of
boxes, `l_r` is the length of the `r`-th row and `l_c` is the length of the `c`-th column.
"""
function casimir(s::SUNIrrep{N}) where {N}
    rows = s.I
    cols = get_column_numbers(s)
    p = sum(rows)
    return (p * (N - p / N) + sum(rows .^ 2) - sum(cols .^ 2)) / 2
end

"""
    heisenberg_term([T=ComplexF64], I₁::I, [I₂::I=I₁]) where {N,I<:SUNIrrep{N}} -> TensorMap{Vec[I],2,2}

Create the TensorMap for the Heisenberg interaction between two sites with irreps `I1` and `I2`.
"""
function heisenberg_term(::Type{T}, I₁::I, I₂::I=I₁) where {N,I<:SUNIrrep{N},T<:Number}
    P₁ = Vect[I](I₁ => 1)
    P₂ = Vect[I](I₂ => 1)
    t = TensorMap(zeros, T, P₁ ⊗ P₂ ← P₁ ⊗ P₂)
    for (c, b) in blocks(t)
        @inbounds for i in axes(b, 1)
            b[i, i] = (casimir(c) - casimir(I₁) - casimir(I₂)) / 2
        end
    end
    return t
end
heisenberg_term(I::SUNIrrep...) = heisenberg_term(ComplexF64, I...)

function sun_heisenberg(::Type{T}, lattice; J::Real=1.0, irrep::SUNIrrep{N}) where {T<:Number,N}
    SS = scale!(heisenberg_term(T, irrep, irrep), J)
    return @mpoham sum(nearest_neighbours(lattice)) do (i, j)
        return SS{i,j}
    end
end

function initialize_mps(H::InfiniteMPOHamiltonian)
    P = oneunit(spacetype(H)) ⊗ prod(physicalspace(H)) ← oneunit(spacetype(H))
    psi_finite = randn(scalartype(H), P)
    As = MPSKit.decompose_localmps(psi_finite)
    return InfiniteMPS(As)
end

function half_ud_num(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{SU2Irrep})
    t = HubbardOperators.single_site_operator(elt, SU2Irrep, SU2Irrep)
    I = sectortype(t)
    block(t, I((0, 1 // 2, 0))) .= 0
    block(t, I((1, 0, 1 // 2))) .= 1 / 2
    return t
end

function MPSKitModels.hubbard_model(elt::Type{<:Number}, ::Type{SU2Irrep}, ::Type{SU2Irrep}, lattice::AbstractLattice;
    t=1, U=1, mu=U / 2, n=1)
    @assert mu ≈ U / 2 && n == 1
    kinetic = -t * e_hop(elt, SU2Irrep, SU2Irrep)
    interaction = U * half_ud_num(elt, SU2Irrep, SU2Irrep)
    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return kinetic{i,j}
        end +
        sum(vertices(lattice)) do i
            return interaction{i}
        end
    end
end

# Utility 
# -------

tomlify(c::Trivial) = 0
tomlify(c::FermionParity) = c.isodd
tomlify(c::U1Irrep) = Int(2 * c.charge)
tomlify(c::SU2Irrep) = dim(c)
tomlify(c::SU3Irrep) = collect(dynkin_label(c))
tomlify(c::ProductSector) = collect(map(tomlify, c.sectors))
tomlify(V::ElementarySpace) = Dict("sectors" => collect(sectors(V)), "dims" => dim.(Ref(V), sectors(V)))

untomlify(::Type{Trivial}, val) = Trivial()
untomlify(::Type{FermionParity}, val) = FermionParity(val)
untomlify(::Type{U1Irrep}, val) = U1Irrep(val // 2)
untomlify(::Type{FermionParity ⊠ U1Irrep ⊠ U1Irrep}, val) = FermionParity(val[1]) ⊠ U1Irrep(val[2]) ⊠ U1Irrep(val[3])
untomlify(::Type{FermionParity ⊠ U1Irrep ⊠ SU2Irrep}, val) = FermionParity(val[1]) ⊠ U1Irrep(val[2]) ⊠ SU2Irrep(val[3])
untomlify(::Type{FermionParity ⊠ SU2Irrep ⊠ SU2Irrep}, val) = FermionParity(val[1]) ⊠ SU2Irrep(val[2]) ⊠ SU2Irrep(val[3])
untomlify(::Type{U1Irrep ⊠ U1Irrep}, val) = ⊠(U1Irrep.(val)...)
untomlify(::Type{SU2Irrep}, val) = SU2Irrep((val - 1) // 2)
untomlify(::Type{SU3Irrep}, val) = SU3Irrep(val)

untomlify(::Type{ComplexSpace}, val) = ComplexSpace(only(val["dims"]))
untomlify(::Type{V}, val) where {V<:GradedSpace} = V(untomlify(sectortype(V), c) => d for (c, d) in zip(val["sectors"], val["dims"]))

function desymmetrize(V′::Type{<:ElementarySpace}, V::ElementarySpace)
    I = sectortype(V′)
    dims = TensorKit.SectorDict{I,Int}()
    for c in sectors(V)
        d = dim(V, c)
        for (j, d′) in desymmetrize(I, c)
            dims[j] = get(dims, j, 0) + d * d′
        end
    end
    return V′(dims)
end

desymmetrize(::Type{U1Irrep}, c::SU2Irrep) = (U1Irrep(j) => 1 for j in -(c.j):c.j)
desymmetrize(::Type{Trivial}, c::Sector) = (Trivial() => dim(c),)
function desymmetrize(I::Type{U1Irrep ⊠ U1Irrep}, c::SU3Irrep)
    dims = TensorKit.SectorDict{I,Int}()
    for m in basis(c)
        y, i = U1Irrep.(Zweight(m))
        dims[y⊠i] = get(dims, y ⊠ i, 0) + 1
    end
    return dims
end
function desymmetrize(I::Type{FermionParity ⊠ U1Irrep ⊠ SU2Irrep}, c::(FermionParity ⊠ SU2Irrep ⊠ SU2Irrep))
    dims = TensorKit.SectorDict{I,Int}()
    for (i, d1) in desymmetrize(U1Irrep, c[2])
        dims[c[1]⊠i⊠c[3]] = get(dims, c[1] ⊠ i ⊠ c[3], 0) + d1
    end
    return dims
end
function desymmetrize(I::Type{FermionParity ⊠ U1Irrep ⊠ U1Irrep}, c::(FermionParity ⊠ SU2Irrep ⊠ SU2Irrep))
    dims = TensorKit.SectorDict{I,Int}()
    for (i, d1) in desymmetrize(U1Irrep, c[2]), (j, d2) in desymmetrize(U1Irrep, c[3])
        dims[c[1]⊠i⊠j] = get(dims, c[1] ⊠ i ⊠ j, 0) + d1 * d2
    end
    return dims
end
function desymmetrize(I::Type{FermionParity}, c::(FermionParity ⊠ SU2Irrep ⊠ SU2Irrep))
    dims = TensorKit.SectorDict{I,Int}()
    for (i, d1) in desymmetrize(Trivial, c[2]), (j, d2) in desymmetrize(Trivial, c[3])
        dims[c[1]] = get(dims, c[1], 0) + d1 * d2
    end
    return dims
end


function summarize_mps_spaces(psi, H)
    Vmps = left_virtualspace(psi, 1)
    Vmpo = TensorKit.oplus(left_virtualspace(H, 1)...)
    return Dict("virtual_mps" => Vmps, "virtual_mpo" => Vmpo, "physical" => physicalspace(psi, 1))
end

# Benchmarks
# ----------
const _repetitions = 5

function instantiate_mps_spaces(::Type{V}, d::Dict) where {V<:ElementarySpace}
    keynames = ("virtual_mps", "virtual_mpo", "physical")
    issetequal(keys(d), keynames) || throw(ArgumentError("invalid dict"))
    Vmps, Vmpo, Vphys = untomlify.(V, getindex.((d,), keynames))
    return Vmps, Vmpo, Vphys
end

function init_mps_tensors(T, Vmps, Vmpo, Vphys)
    A = randn(T, Vmps ⊗ Vphys ← Vmps)
    M = randn(T, Vmpo ⊗ Vphys ← Vphys ⊗ Vmpo)
    FL = randn(T, Vmps ← Vmps ⊗ Vmpo)
    FR = randn(T, Vmpo ⊗ Vmps' ← Vmps')
    return A, M, FL, FR
end

function AC_benchmark(A, M, FL, FR; repetitions::Integer=_repetitions)
    for _ in 1:repetitions
        @tensor A[-1 -2; -3] ≔ A[1 2; 4] * FL[-1; 1 3] * M[3 -2; 2 5] * FR[5 -3; 4]
    end
    return A
end

function AC2_benchmark(A, M, FL, FR; repetitions::Integer=_repetitions)
    @tensor AC2[-1 -2; -3 -4] ≔ A[-1 -2; 1] * A[1 -4; -3]
    for _ in 1:repetitions
        @tensor AC2[-1 -2; -3 -4] ≔ AC2[1 3; 7 5] * FL[-1; 1 2] * M[2 -2; 3 4] * M[4 -4; 5 6] * FR[6 -3; 7]
    end
    A1, S, A2 = tsvd!(AC2; trunc=truncspace(space(A, 1)))
    return A1 * S, repartition(A2, 2, 1)
end

function generate_AC_benchmark(T, Vmps, Vmpo, Vphys)
    local init
    let T = T, Vphys = Vphys, Vmps = Vmps, Vmpo = Vmpo
        init() = init_mps_tensors(T, Vmps, Vmpo, Vphys)
    end
    @benchmarkable AC_benchmark(A, M, FL, FR) setup = ((A, M, FL, FR) = $init())
end
function generate_AC2_benchmark(T, Vmps, Vmpo, Vphys)
    local init
    let T = T, Vphys = Vphys, Vmps = Vmps, Vmpo = Vmpo
        init() = init_mps_tensors(T, Vmps, Vmpo, Vphys)
    end
    @benchmarkable AC2_benchmark(A, M, FL, FR) setup = ((A, M, FL, FR) = $init())
end

const spacetypes = Dict{String,Type}(
    "Trivial" => ComplexSpace, "U1" => U1Space, "U1xU1" => Rep[U₁×U₁], "SU2" => SU2Space,
    "SU3" => Vect[SU3Irrep], "fZ2" => Vect[fℤ₂], "fZ2xU1xU1" => Vect[fℤ₂⊠U1Irrep⊠U1Irrep],
    "fZ2xU1xSU2" => Vect[fℤ₂⊠U1Irrep⊠SU2Irrep], "fZ2xSU2xSU2" => Vect[fℤ₂⊠SU2Irrep⊠SU2Irrep]
)

function generate_AC_benchmark!(suite, kind, T, params; maxdims=Dict())
    V = spacetypes[kind]

    Vmps, Vmpo, Vphys = instantiate_mps_spaces(V, params)
    D = dim(Vmps)
    D > get(maxdims, kind, typemax(Int)) && return suite

    suite[D] = generate_AC_benchmark(T, Vmps, Vmpo, Vphys)
    return suite
end
function generate_AC2_benchmark!(suite, kind, T, params; maxdims=Dict())
    V = spacetypes[kind]

    Vmps, Vmpo, Vphys = instantiate_mps_spaces(V, params)
    D = dim(Vmps)
    D > get(maxdims, kind, typemax(Int)) && return suite

    suite[D] = generate_AC2_benchmark(T, Vmps, Vmpo, Vphys)
    return suite
end

end
