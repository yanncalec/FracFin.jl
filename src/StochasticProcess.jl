########## Definitions of various stochastic process ##########

abstract type AbstractRandomField end
abstract type AbstractRealRandomField <: AbstractRandomField end


"""
Generic zero-mean real-valued stochastic process.
"""
abstract type StochasticProcess{T<:TimeStyle} <: AbstractRealRandomField end
const ContinuousTimeStochasticProcess = StochasticProcess{ContinuousTime}
const DiscreteTimeStochasticProcess = StochasticProcess{DiscreteTime}


abstract type StationaryProcess{T} <: StochasticProcess{T} end
const ContinuousTimeStationaryProcess = StationaryProcess{ContinuousTime}
const DiscreteTimeStationaryProcess = StationaryProcess{DiscreteTime}


abstract type SelfSimilarProcess <: ContinuousTimeStochasticProcess end
abstract type SSSIProcess <: SelfSimilarProcess end  #


@doc raw"""
Stationary process resulting from filtration (without DC) of another process.

The causal version is defined as:
    \sum_{n=0}^{N-1} a[n+1] X(t-nδ)

and the anti-causal version:
    \sum_{n=0}^{N-1} a[n+1] X(t+nδ)

# Note
- Under the stationarity assumption the causality has no effect on the auto-covariance function.
- The aimed concrete type of the abstract type `IncrementProcess` is `FractionalGaussianNoise` (fGn) which is stationary. However due to the lack of multiple inheritance in Julia it is very hard to make `IncrementProcess` a subtype of `StationaryProcess`. Possible solutions to this problem include 1) SimpleTrait.jl,  2) copy functions defined for `StationaryProcess`,  3) force `FilteredProcess` to be a subtype of `StationaryProcess`. We adopt the solution 3) here.
"""
abstract type FilteredProcess{T<:TimeStyle, P<:StochasticProcess{>:T}} <: StationaryProcess{T} end

"""
Differential process `X(t±δ) - X(t)`
"""
abstract type DifferentialProcess{T<:TimeStyle, P<:StochasticProcess{>:T}} <: FilteredProcess{T, P} end  # Process as first order finite difference of another process.

"""
Discrete time differential process `X((n±l)δ) - X(nδ)` with `l` being the lag.
"""
abstract type IncrementProcess{P<:StochasticProcess} <: DifferentialProcess{DiscreteTime, P} end



#### Process specific functions ####
"""
Return the self-similar exponent of the process.
"""
ss_exponent(X::SelfSimilarProcess) = throw(NotImplementedError())

"""
Return the filter of the process.
"""
filter(X::FilteredProcess)::AbstractVector = throw(NotImplementedError())

"""
Test whether the filtration is causal:
"""
iscausal(X::FilteredProcess) = throw(NotImplementedError())

"""
Return the parent process of the filtered process.
"""
parent_process(X::FilteredProcess) = throw(NotImplementedError())

"""
Return the step `δ` of the filtered process.
"""
step(X::FilteredProcess)::Real = throw(NotImplementedError())

"""
Filter of differential process, defined as
- the causal case: X(t) - X(t-lδ)
- the anti-causal case: X(t+lδ) - X(t)
with `l = lag(X)`.
"""
function filter(X::DifferentialProcess)
    filt = vcat(1, zeros(lag(X)-1), -1)
    return causal(X) ? filt : reverse(filt)
end

"""
Return the lag of differential process.
"""
lag(X::DifferentialProcess) = throw(NotImplementedError())


#### Generic identifiers ####
"""
Test whether a process is time discrete (a time series).
"""
iscontinuoustime(X::StochasticProcess{ContinuousTime}) = true
iscontinuoustime(X::StochasticProcess{DiscreteTime}) = false

"""
Test whether a process is multivariate.
"""
ismultivariate(X::StochasticProcess) = false

"""
Test whether a process has stationary increments.
"""
isincrementstationary(X::StochasticProcess) = false
isincrementstationary(X::SSSIProcess) = true
isincrementstationary(X::StationaryProcess) = true

"""
Test whether a process is stationary.
"""
isstationary(X::StochasticProcess) = false
isstationary(X::StationaryProcess) = true
# isstationary(X::FilteredProcess{T, P}) where {T, P<:StationaryProcess} = true
# isstationary(X::IncrementProcess{P}) where {P<:SSSIProcess} = true


"""
Determine whether a grid has the constant step.
"""
function isregulargrid(G::AbstractVector)
    return if length(G) == 2
        true
    elseif length(G) > 2  # second order difference should be close to 0
        isapprox(maximum(abs.(diff(diff(G)))), 0.0; atol=1e-10)
    else
        false
    end
end



#### Statistics for stochastic process ####
"""
Auto-covariance function of a stochastic process.
"""
autocov(X::StochasticProcess{T}, t::T, s::T) where T<:TimeStyle = throw(NotImplementedError())

autocov(X::StationaryProcess{T}, t::T) where T<:TimeStyle = throw(NotImplementedError())
autocov(X::StationaryProcess{T}, t::T, s::T) where T<:TimeStyle = autocov(X, t-s)


"""
Compute the auto-covariance matrix of a stochastic process on a sampling grid.
"""
function autocov!(C::Matrix{<:AbstractFloat}, X::StochasticProcess{T}, G::AbstractVector{<:T}) where T<:TimeStyle
    @assert size(C, 1) == size(C, 2) == length(G)

    # construct the covariance matrix (a symmetric matrix)
    N = size(C, 1)  # dimension of the auto-covariance matrix
    for c = 1:N, r = 1:c
        C[r,c] = autocov(X, G[r], G[c])
    end
    for c = 1:N, r = (c+1):N
        C[r,c] = C[c,r]
    end
    return Symmetric(C)
end

function autocov!(C::Matrix{<:AbstractFloat}, X::StochasticProcess{T}, G1::AbstractVector{<:T}, G2::AbstractVector{<:T}) where T<:TimeStyle
    @assert size(C, 1) == length(G1) && size(C, 2) == length(G2)

    # construct the covariance matrix (a symmetric matrix)
    N,M = size(C)  # dimension of the auto-covariance matrix
    for c = 1:M, r = 1:N
        C[r,c] = autocov(X, G1[r], G2[c])
    end
    return C
end


"""
Compute the auto-covarince sequence of a stationary process on a regular grid.
"""
function autocov!(C::AbstractVector{<:AbstractFloat}, X::StationaryProcess{T}, G::AbstractVector{<:T}) where T<:TimeStyle
    # check dimension
    @assert length(C) == length(G)
    @assert isregulargrid(G)

    # construct the auto-covariance kernel
    for n = 1:length(C)
        C[n] = autocov(X, G[n]-G[1])
    end
    return C
end


"""
    covseq(X::StationaryProcess, G::RegularGrid)

Return the auto-covariance sequence of a stationary process on a regular grid.
"""
covseq(X::StationaryProcess{T}, G::AbstractVector{<:T}) where T<:TimeStyle = autocov!(zeros(length(G)), X, G)


function autocov!(C::Matrix{<:AbstractFloat}, X::StationaryProcess{T}, G::AbstractVector{<:T}) where T<:TimeStyle
    # check dimension
    @assert size(C, 1) == size(C, 2) == length(G)

    # construct the covariance matrix (a Toeplitz matrix)
    N = size(C, 1)
    return covmat!(C, covseq(X,G))
end


"""
Construct the covariance matrix from the covariance sequence.
"""
function covmat!(C::Matrix{T}, S::AbstractVector{T}) where {T<:Real}
    for c = 1:length(S), r = 1:length(S)
        C[r,c] = S[abs(r-c)+1]
    end
    return C
end

covmat(S::AbstractVector{T}) where {T<:Real} = covmat!(zeros(T, length(S), length(S)), S)


"""
Return the auto-covarince matrix of a stochastic process on a sampling grid.
"""
covmat(X::StochasticProcess, G::AbstractVector) = autocov!(zeros(length(G), length(G)), X, G)


"""
Return the auto-covarince matrix of a stochastic process on two sampling grids.

The `(i,j)`-th coefficient in the matrix is `autocov(G1[i], G2[j])`.
"""
covmat(X::StochasticProcess, G1::AbstractVector, G2::AbstractVector) = autocov!(zeros(length(G1), length(G2)), X, G1, G2)

"""
Return the auto-covarince matrix of a stochastic process on an integer sampling grid.
"""
covmat(X::StochasticProcess, N::Integer) = covmat(X, 1:N)
covmat(X::StochasticProcess, N::Integer, M::Integer) = covmat(X, 1:N, 1:M)


"""
Return the partial correlation function of a discrete time stationary process.
"""
partcorr(X::DiscreteTimeStationaryProcess, n::DiscreteTime) = throw(NotImplementedError())


"""
Compute the partial correlation sequence of a discrete time stationary process on a regular integer sampling grid.
"""
function partcorr!(C::Vector{<:AbstractFloat}, X::DiscreteTimeStationaryProcess, G::AbstractVector{<:DiscreteTime})
    # check dimension
    @assert length(C) == length(G)
    @assert isregulargrid(G)

    for n = 1:length(C)
        C[n] = partcorr(X, G[n]-G[1]+1)
    end
    return C
end


"""
Return the partial correlation function of a time discrete stationary process.

# Args
- method: if set to `:LevinsonDurbin` it will use the Levinson-Durbin method which needs only the autocovariance expression of the process.
"""
function partcorr(X::DiscreteTimeStationaryProcess, G::AbstractVector{<:DiscreteTime}, method::Symbol=:None)
    if method == :LevinsonDurbin # use Levinson-Durbin
        cseq = covseq(X, G)
        pseq, sseq, rseq = LevinsonDurbin(cseq)
        return rseq
    else
        return partcorr!(zeros(length(G)), X, G)
    end
end


# function autocov(X::FilteredProcess{T, P}, t::T, s::T) where {T<:TimeStyle, P<:StochasticProcess}
#     proc = parent_process(X)
#     δ = step(X)

#     return autocov(proc, (t+δ), (s+δ)) - autocov(proc, (t+δ), (s)) - autocov(proc, (t), (s+δ)) + autocov(proc, (t), (s))
# end


# """
# Return the auto-covariance function of a process of increments. The computation is done via the auto-covariance of the parent process.
# """
# function autocov(X::IncrementProcess{P}, t::Integer, s::Integer) where P<:StochasticProcess
#     proc = parent_process(X)
#     δ = step(X)

#     return autocov(proc, (t+δ), (s+δ)) - autocov(proc, (t+δ), (s)) - autocov(proc, (t), (s+δ)) + autocov(proc, (t), (s))
# end


#### Statistical inference on stochastic process ####

"""
Conditional mean and covariance of a Gaussian process `P` on the position `Gx` given the value `Y` on the postion `Gy`.
"""
function cond_mean_cov(P::StochasticProcess{T}, Gx::AbstractVector{T}, Gy::AbstractVector{T}, Y::AbstractVector{<:Real}) where T<:TimeStyle
    @assert length(Gy) == length(Y)

    Σxx = covmat(P, Gx)
    Σxy = covmat(P, Gx, Gy)
    Σyy = covmat(P, Gy)
    return  Σxy * (Σyy\Y), Σxx - Σxy * inv(Σyy) * Σxy'
end

cond_mean_cov(P::StochasticProcess, gx::ContinuousTime, Gy::AbstractVector, Y::AbstractVector) = cond_mean_cov(P, [gx], Gy, Y)


include("FBM.jl")  # Fractional Brownian Motion related
include("MFBM.jl")  # Multi-Fractional Brownian Motion related
include("FARIMA.jl")  # Fractional ARIMA related