########## Simulation methods (sampler) for stochastic process ##########

"""
Abstract sampler for a stochastic process.

# Members
- proc: process to be sampled
- grid: sampling grid adapted to proc
"""
abstract type AbstractRandomFieldSampler end

abstract type Sampler{T<:TimeStyle, P<:StochasticProcess{<:T}, G<:AbstractVector{<:T}} <: AbstractRandomFieldSampler end

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
rand_otf(p::StochasticProcess{T}, g::AbstractVector{<:T}) where T<:TimeStyle = rand_otf!(zeros(Float64, length(g)), p, g)

length(s::Sampler) = length(s.grid)
size(s::Sampler) = size(s.grid,)

# The convert function is implicitly called by `new()` in the constructor of a sampler
# convert(::Type{S}, s::Sampler) where {S<:Sampler} = S(s.proc, s.grid)


"""
Initialize a sampler for standard fBm (or related) process.

# Args
- H: hurst exponent
- N: number of points of the regular sampling grid
- name: name of the sampling method: {:cholesky, :circulant, :hosking, :midpoint, :wav elet}
- proc: {:fBm, :fGn, :fIt}

# Returns
- sampler object

# Notes
- Only Cholesky method can work with continuous time grid on. We take `(1:N)/N` as the continuous time grid and `1:N` as the discrete time grid.
"""
function init_sampler_fBm(H::Real, N::Integer, name::Symbol)
    @assert 0. < H < 1.

    fBm = FractionalBrownianMotion(H)
    fGn = FractionalGaussianNoise(H, 1)
    fIt = FractionalIntegrated(H-1/2)  # fIt(H-1/2) is an approximation of fGn(H)

    sampler = if name == :cholesky
        CholeskySampler(fBm, 1:N)
    elseif name == :circulant
        CircSampler(fGn, 1:N)
    elseif name == :hosking
        HoskingSampler(fGn, 1:N)
    elseif name == :midpoint
        CRMDSampler(fGn, 1:N)
    elseif name == :wavelet
        WaveletSampler(fIt, 1:N, psflag=true)  # psflag=true for the built-in partial sum, such that the sample trajectory is an approximation fBm. Otherwise it will be an approximation of fGn.
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
    # the scaling factor δ for arbitrary N comes from the following reasoning: to transform a fBm defined on [0,s] to [0,t], just apply the scaling factor (t/s)^H on the original fBm.
    δ = Tmax / N  # sampling step

    X = if T <: CholeskySampler
        δ^H * rand(sampler)
    elseif T <: Union{CircSampler, HoskingSampler, CRMDSampler}
        δ^H * cumsum(rand(sampler))
    elseif T <: WaveletSampler
        # B_H = 2^(-J*H) * rand(sampler) is a fBm trajectory on [0,1] of length 2^J. Taking the first N samples and rescaling as (2^J/N)^H * B_H[1:N] gives a fBm trajectory on [0,1] of length N. Finally multiplying by Tmax^H it gives a trajectory on [0, Tmax], which gives the (same) scaling factor (Tmax/N)^H

        δ^H * rand(sampler)  # No cumsum here since psflag=true in the initialization of sampler
    end

    # X .-= X[1]  # force starting from 0
    return X
end


include("Sampler_FBM.jl")
# include("Sampler_MFBM.jl")