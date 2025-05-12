D = dim(left_virtualspace(psi, 1))
while D < 4096
    Dtrim = max(round(Int, D * factor), 10)
    psi, = changebonds(psi, H, OptimalExpand(; trscheme=truncdim(Dtrim)))
    psi, = find_groundstate(psi, H, VUMPS(; tol=1e-8, maxiter=10))
    D = dim(left_virtualspace(psi, 1))
end
psi, = find_groundstate(psi, H, VUMPS(; tol=1e-10, maxiter=20))
psi = changebonds(psi, SvdCut(; trscheme=truncdim(4096)))

entanglementplot(psi, 1)

Vl = left_virtualspace(psi, 1)
samples1 = mapreduce(vcat, sectors(Vl)) do c
    return fill(dim(c), dim(Vl, c))
end
weights1 = mapreduce(vcat, sectors(Vl)) do c
    return fill(float(dim(c)), dim(Vl, c))
end
Vr = right_virtualspace(psi, 1)
samples2 = mapreduce(vcat, sectors(Vr)) do c
    return fill(dim(c), dim(Vr, c))
end
weights2 = mapreduce(vcat, sectors(Vr)) do c
    return fill(float(dim(c)), dim(Vr, c))
end
samples = vcat(samples1, samples2)
weights = vcat(weights1, weights2)


dist_invgauss = Distributions.fit_mle(InverseGaussian, samples, weights)
dist_poisson = Distributions.fit_mle(Poisson, samples, weights)

xs = 1:9
ys_invgauss = pdf.(dist_invgauss, xs)
ys_poisson = pdf.(dist_poisson, xs)

ys_actual = map(xs) do x
    c = SU2Irrep((x - 1) / 2)
    if iseven(x)
        return dim(Vr, c) * dim(c) / dim(Vr) / 2
    else
        return dim(Vl, c) * dim(c) / dim(Vl) / 2
    end
end


let f = Figure()
    ax = Axis(f[1, 1])
    scatterlines!(ax, xs, ys_invgauss, label="InvGauss")
    scatterlines!(ax, xs, ys_poisson, label="Poisson")
    scatterlines!(ax, xs, ys_actual, label="Actual")
    Legend(f[1, 2], ax)
    f
end


# lets settle for the Poisson distribution:
spectrum1 = entanglement_spectrum(psi, 1)
spectrum2 = entanglement_spectrum(psi, 2)
spectra = merge(spectrum1, spectrum2)
svals = logrange(1e-1, 1e-6, 10)
dists = map(svals) do sval
    samples = mapreduce(vcat, collect(keys(spectra))) do c
        return fill(dim(c), count(>(sval), spectra[c]))
    end
    weights = mapreduce(vcat, collect(keys(spectra))) do c
        return fill(float(dim(c)), count(>(sval), spectra[c]))
    end
    dist = Distributions.fit_mle(Poisson, samples, weights)
    return dist
end
λs = map(Base.Fix2(getproperty, :λ), dists)

f = CairoMakie.plot((log10.(svals)), (λs))

model(x, p) = p[1] .* (x .^ (-p[2])) .+ p[3]
p0 = [1.0, 1.0, 1.0]
fit = curve_fit(model, svals, λs, p0)
fit.param
fig, ax = f
CairoMakie.plot!(ax, log10.(svals), model(svals, fit.param), label="fit")

fig

# not great, try again:

Ds = logrange(10, 4096, 10)
dists = map(Ds) do D
    psi_D = changebonds(psi, SvdCut(; trscheme=truncdim(round(Int, D))))
    spectrum1 = entanglement_spectrum(psi_D, 1)
    spectrum2 = entanglement_spectrum(psi_D, 2)
    spectra = merge(spectrum1, spectrum2)
    samples = mapreduce(vcat, collect(keys(spectra))) do c
        return fill(dim(c), length(spectra[c]))
    end
    weights = mapreduce(vcat, collect(keys(spectra))) do c
        return fill(float(dim(c)), length(spectra[c]))
    end
    dist = Distributions.fit_mle(Poisson, samples, weights)
    return dist
end
λs = map(Base.Fix2(getproperty, :λ), dists)

f = CairoMakie.plot(log10.(Ds), map(Base.Fix2(getproperty, :λ), dists))
model(x, p) = p[1] .* (x .^ (p[2])) .+ p[3]
p0 = [1.0, 1.0, 1.0]
fit = curve_fit(model, Ds, λs, p0)
fig, ax = f
CairoMakie.plot!(ax, log10.(Ds), model(Ds, fit.param), label="fit")
fig

CairoMakie.plot(Ds, λs)

## fit.param = [411, 0.0007, -408]


Ds = logrange(10, 4096, 10)
p = [411, 0.0007, -408]
model(x, p) = p[1] .* (x .^ (p[2])) .+ p[3]
model(Ds, p)
