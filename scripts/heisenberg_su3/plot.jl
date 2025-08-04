using TensorKitBenchmarks
using CairoMakie
using TOML
using LaTeXStrings

blasthreads = 1
filename_in = joinpath(@__DIR__, "heisenberg_sun_results")
filename = filename_in * "_b$blasthreads"

estimator = minimum

results = only(BenchmarkTools.load(filename * ".json"))

kinds = ["Trivial", "U1xU1", "SU3"]


fig = let f = Figure(; size=(800, 600), title="SU(3) Heisenberg")
    xscale = log10
    yscale = log10
    xlabel = L"D"
    ylabel = L"t[s]"
    xlabelsize = ylabelsize = 20
    
    ax1 = Axis(f[1, 1]; xlabel, xscale, ylabel, yscale, title="single-site benchmark", xlabelsize, ylabelsize)
    for kind in kinds
        group = results["AC"][kind]
        Ds = Int[]
        ts = Float64[]
        for (D, res) in group
            push!(Ds, Base.parse(Int, D))
            push!(ts, estimator(res).time / 1e9)
        end
        I = sortperm(Ds)
        Ds = Ds[I]
        ts = ts[I]
        
        scatterlines!(ax1, Ds, ts; label=kind)
    end
    
    ax2 = Axis(f[1, 2]; xlabel, xscale, xlabelsize, ylabel, yscale, ylabelsize, title="two-site benchmark")
    for kind in kinds
        group = results["AC2"][kind]
        Ds = Int[]
        ts = Float64[]
        for (D, res) in group
            push!(Ds, Base.parse(Int, D))
            push!(ts, estimator(res).time / 1e9)
        end
        I = sortperm(Ds)
        Ds = Ds[I]
        ts = ts[I]
        
        scatterlines!(ax2, Ds, ts; label=kind)
    end

    Legend(f[2, :], ax1; orientation=:horizontal)

    save(filename * ".pdf", f)
    f
end
