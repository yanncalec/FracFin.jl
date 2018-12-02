########## Simulation methods (sampler) for stochastic process ##########

"""
Abstract sampler for a stochastic process.

# Members
- proc: process to be sampled
- grid: sampling grid adapted to proc
"""
abstract type AbstractRandomFieldSampler end

abstract type Sampler{T<:TimeStyle, P<:StochasticProcess{T}, G<:AbstractVector{<:T}} <: AbstractRandomFieldSampler end

const ContinuousTimeGrid = AbstractVector{<:ContinuousTime}
const DiscreteTimeGrid = AbstractVector{<:DiscreteTime}

const ContinuousTimeSampler{P} = Sampler{ContinuousTime, P, ContinuousTimeGrid}
const DiscreteTimeSampler{P} = Sampler{DiscreteTime, P, DiscreteTimeGrid}


"""
Random sampling function for an initialized sampler.
"""
rand!(x::Vector{<:AbstractFloat}, s::Sampler) = throw(NotImplementedError())
rand(s::Sampler) = rand!(zeros(Float64, size(s)), s)


"""
Random sampling function with on-the-fly implementation.
"""
rand_otf!(x::Vector{<:AbstractFloat}, p::StochasticProcess{T}, g::AbstractVector{<:T}) where T = throw(NotImplementedError())
rand_otf(p::StochasticProcess{T}, g::AbstractVector{<:T}) where T = rand_otf!(Vector{<:AbstractFloat}(length(g)), p, g)

length(s::Sampler) = length(s.grid)
size(s::Sampler) = size(s.grid,)

# The convert function is implicitly called by `new()` in the constructor of a sampler
# convert(::Type{S}, s::Sampler) where {S<:Sampler} = S(s.proc, s.grid)


"""
Initialize a sampler for fGn.

# Args
- H: hurst exponent
- N: number of points of the regular sampling grid
- name: name of the sampling method: {"CHOLESKY", "CIRCULANT", "HOSKING", "MIDPOINT", "WAVELET"}

# Returns
- an object of sampler
"""
function init_sampler_fGn(H::Real, N::Integer, name::String)
    @assert 0. < H < 1.

    name = uppercase(name)
    fGn = FractionalGaussianNoise(H)  # use unit step of increment
    fIt = FractionalIntegrated(H-1/2)

    Rgrid = (1:N)/N  # sampling grid on [0,1]
    Zgrid = 1:N  # sampling grid 1,2...

    # the scaling factor \delta for arbitrary N comes from the following reasoning: to transform a fBm defined on [0,s] to [0,t], just apply the scaling factor (t/s)^H on the original fBm.
    # δ = Tmax * Rgrid.step  # scaling factor

    if name == "CHOLESKY"
        sampler = CholeskySampler(fGn, Rgrid)
    elseif name == "CIRCULANT"
        sampler = CircSampler(fGn, Zgrid)
    elseif name == "HOSKING"
        sampler = HoskingSampler(fGn, Zgrid)
    elseif name == "MIDPOINT"
        sampler = CRMDSampler(fGn, Zgrid)
    elseif name == "WAVELET"
        sampler = WaveletSampler(fIt, Zgrid, psflag=false)
    else
        error("Unknown method $(name).")
    end

    return sampler
end


"""
Generate a sample trajectory of fGn on the interval [0, `Tmax`] by applying an initialized sampler.
"""
function rand_fGn(sampler::Sampler, Tmax::Real=1.)
    T = typeof(sampler)
    N = length(sampler)
    H = ss_exponent(sampler.proc)
    δ = Tmax / N  # sampling step

    if T <: CholeskySampler
        X = rand(sampler)
    elseif T <: Union{CircSampler, HoskingSampler, CRMDSampler, WaveletSampler}
        X = δ^H * rand(sampler)
    end

    return X
end


"""
Initialize a sampler for fBm.

# Args
- H: hurst exponent
- N: number of points of the regular sampling grid
- name: name of the sampling method: {"CHOLESKY", "CIRCULANT", "HOSKING", "MIDPOINT", "WAVELET"}

# Returns
- an object of sampler
"""
function init_sampler_fBm(H::Real, N::Integer, name::String)
    @assert 0. < H < 1.

    name = uppercase(name)
    fBm = FractionalBrownianMotion(H)
    fGn = FractionalGaussianNoise(H)  # use unit step of increment
    fIt = FractionalIntegrated(H-1/2)

    Rgrid = (1:N)/N  # sampling grid on [0,1]
    Zgrid = 1:N  # sampling grid 1,2...

    # the scaling factor \delta for arbitrary N comes from the following reasoning: to transform a fBm defined on [0,s] to [0,t], just apply the scaling factor (t/s)^H on the original fBm.
    # δ = Tmax * Rgrid.step  # scaling factor

    if name == "CHOLESKY"
        sampler = CholeskySampler(fBm, Rgrid)
    elseif name == "CIRCULANT"
        sampler = CircSampler(fGn, Zgrid)
    elseif name == "HOSKING"
        sampler = HoskingSampler(fGn, Zgrid)
    elseif name == "MIDPOINT"
        sampler = CRMDSampler(fGn, Zgrid)
    elseif name == "WAVELET"
        sampler = WaveletSampler(fIt, Zgrid, psflag=true)
    else
        error("Unknown method $(name).")
    end

    return sampler
end


"""
Generate a sample trajectory of fBm on the interval [0, `Tmax`] by applying an initialized sampler.
"""
function rand_fBm(sampler::Sampler, Tmax::Real=1.)
    T = typeof(sampler)
    N = length(sampler)
    H = ss_exponent(sampler.proc)
    δ = Tmax / N  # sampling step

    if T <: CholeskySampler
        X = rand(sampler)
    elseif T <: Union{CircSampler, HoskingSampler, CRMDSampler}
        X = δ^H * cumsum(rand(sampler))
    elseif T <: WaveletSampler
        # B_H = 2^(-J*H) * rand(sampler) is a fBm trajectory on [0,1] of length 2^J. Taking the first N samples and rescaling by (2^J/N)^H * B_H[1:N] gives a fBm trajectory on [0,1] of length N. Finally multiplying by Tmax^H it gives a trajectory on [0, Tmax], which gives the (same) scaling factor (Tmax/N)^H
        X = δ^H * rand(sampler)
    end

    # X .-= X[1]  # force starting from 0
    return X
end

include("Sampler_FBM.jl")
include("Sampler_MFBM.jl")