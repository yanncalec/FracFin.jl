########## Definitions of various stochastic process ##########

"""
Real Gaussian stochastic process.
"""
abstract type StochasticProcess{T<:TimeStyle}<:Sampleable{Univariate, Continuous} end


"""
Process with integer time index (time-series).
"""
const DiscreteTimeStochasticProcess = StochasticProcess{DiscreteTime}


"""
Process with real time index.
"""
const ContinuousTimeStochasticProcess = StochasticProcess{ContinuousTime}


"""
Stationary process.
"""
abstract type StationaryProcess{T}<:StochasticProcess{T} end

const DiscreteTimeStationaryProcess = StationaryProcess{DiscreteTime}
const ContinuousTimeStationaryProcess = StationaryProcess{ContinuousTime}


"""
Self-similar process.

# Members
- ss_exponent()
"""
abstract type SelfSimilarProcess<:ContinuousTimeStochasticProcess end


"""
    ss_exponent(X::SelfSimilarProcess)

Return the self-similar exponent of the process.
"""
ss_exponent(X::SelfSimilarProcess) = throw(NotImplementedError("ss_exponent(::$(typeof(X)))"))


"""
Non-stationary process of increments.
"""
abstract type NonStationaryIncrementProcess{T<:TimeStyle, P<:StochasticProcess}<:StochasticProcess{T} end


"""
Stationary process of increments.
"""
abstract type StationaryIncrementProcess{T<:TimeStyle, P<:StochasticProcess}<:StationaryProcess{T} end


"""
Process of increments including both non-stationary and stationary cases.

# Members
- parent_process
- step
"""
const IncrementProcess{T<:TimeStyle, P<:StochasticProcess} = Union{NonStationaryIncrementProcess{T, P}, StationaryIncrementProcess{T, P}}


"""
Self-similar process with stationary increments (SSSI).
"""
abstract type SSSIProcess<:SelfSimilarProcess end

abstract type IncrementSSSIProcess{T<:TimeStyle, P<:SSSIProcess}<:StationaryIncrementProcess{T, P} end

"""
Process of increments of a SSSI process.
"""
abstract type IncrementSSSIProcess{T<:TimeStyle, P<:SSSIProcess}<:StationaryIncrementProcess{T, P} end

const DiscreteTimeIncrementSSSIProcess{P<:SSSIProcess} = IncrementSSSIProcess{DiscreteTime, P}
const ContinuousTimeIncrementSSSIProcess{P<:SSSIProcess} = IncrementSSSIProcess{ContinuousTime, P}


"""
    step(X::IncrementProcess)

Return the step of increment of the process.
"""
step(X::IncrementProcess) = throw(NotImplementedError("step(::$(typeof(X)))"))


"""
    ss_exponent(X::IncrementSSSIProcess)

Return the self-similar exponent of the parent process.
"""
ss_exponent(X::IncrementSSSIProcess) = throw(NotImplementedError("ss_exponent(::$(typeof(X)))"))


"""
    autocov(X::StochasticProcess{T}, i::T, j::T) where T

Auto-covariance function of a stochastic process.
"""
autocov(X::StochasticProcess{T}, i::T, j::T) where T = throw(NotImplementedError("autocov(::$(typeof(X)), ::$(T), ::$(T))"))


"""
    autocov!(C::Matrix{Float64}, X::StochasticProcess, G::SamplingGrid)

Compute the auto-covariance matrix of a stochastic process on a sampling grid.
"""
function autocov!(C::Matrix{Float64}, X::StochasticProcess{T}, G::SamplingGrid{<:T}) where T
    # check dimension
    # @assert size(C, 1) == size(C, 2) == length(G)
    # check grid
    # @assert any(diff(G) .> 0)  # grid points must be strictly increasing

    # construct the covariance matrix (a symmetric matrix)
    N = size(C, 1)  # dimension of the auto-covariance matrix
    for c = 1:N, r = 1:c
        C[r,c] = autocov(X, G[c], G[r])
    end
    for c = 1:N, r = (c+1):N
        C[r,c] = C[c,r]
    end
    return C
end


"""
    autocov(X::StochasticProcess, G::SamplingGrid)

Return the auto-covarince matrix of a stochastic process on a sampling grid.
"""
autocov(X::StochasticProcess{T}, G::SamplingGrid{<:T}) where T = autocov!(Matrix{Float64}(undef, length(G),length(G)), X, G)
# autocov(X::StationaryProcess, G::SamplingGrid) = autocov!(Matrix{Float64}(length(G), length(G)), X, G)


"""
    covmat(X::StochasticProcess, G::SamplingGrid)

Alias to `autocov(X::StochasticProcess, G::SamplingGrid)`.
"""
covmat(X::StochasticProcess{T}, G::SamplingGrid{<:T}) where T = autocov(X, G)

covmat(X::DiscreteTimeStochasticProcess, N::Integer) = covmat(X, DiscreteTimeRegularGrid(1:N))


"""
    autocov(X::StationaryProcess, i::Real)

Auto-covarince function of a stationary stochastic process.
"""
autocov(X::StationaryProcess{T}, i::T) where T = throw(NotImplementedError("autocov(::$(typeof(X)), ::Real)"))
autocov(X::StationaryProcess{T}, i::T, j::T) where T = autocov(X, i-j)


"""
    autocov!(C::Vector{Float64}, X::StationaryProcess, G::RegularGrid)

Compute the auto-covarince sequence of a stationary process on a regular grid.
"""
function autocov!(C::Vector{Float64}, X::StationaryProcess{T}, G::RegularGrid{<:T}) where T
    # check dimension
    # @assert length(C) == length(G)

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
covseq(X::StationaryProcess{T}, G::RegularGrid{<:T}) where T = autocov!(zeros(length(G)), X, G)


function autocov!(C::Matrix{Float64}, X::StationaryProcess{T}, G::RegularGrid{<:T}) where T
    # check dimension
    # @assert size(C, 1) == size(C, 2) == length(G)

    # construct the covariance matrix (a Toeplitz matrix)
    N = size(C, 1)
    knl = covseq(X, G)  # autocovariance sequence
    for c = 1:N, r = 1:N
        C[r,c] = knl[abs(r-c)+1]
    end
    return C
end


# function autocov!(C::Matrix{Float64}, X::StationaryProcess, G::Vector{Float64})
#     try
#         idx = convert(Vector{Int64}, G)
#         all(diff(idx) .== 1) || throw(InexactError("Failed to convert the irregular sampling grid."))
#         return autocov!(C, X, idx[1]:idx[end])
#     catch
#         return invoke(autocov!, Tuple{Matrix{Float64}, StochasticProcess, SamplingGrid}, C, X, G)
#     end
# end


"""
    partcorr(X::DiscreteTimeStationaryProcess, n::DiscreteTime)

Return the partial correlation function of a stationary process.
"""
partcorr(X::DiscreteTimeStationaryProcess, n::DiscreteTime) = throw(NotImplementedError("partcorr(::$(typeof(X)), ::$(typeof(n)))"))


"""
    partcorr!(C::Vector{Float64}, X::DiscreteTimeStationaryProcess, G::DiscreteTimeRegularGrid)    

Compute the partial correlation sequence of a stationary process for a range of index.
"""
function partcorr!(C::Vector{Float64}, X::DiscreteTimeStationaryProcess, G::DiscreteTimeRegularGrid)
    # check dimension
    # @assert length(C) == length(G)

    for n = 1:length(C)
        C[n] = partcorr(X, G[n]-G[1]+1)
    end
    return C
end


"""
    partcorr(X::DiscreteTimeStationaryProcess, G::DiscreteTimeRegularGrid, ld::Bool=false)

Return the partial correlation function of a stationary process for a range of index.
If `ld==true` use the Levinson-Durbin method which needs only the autocovariance function of the process.
"""
function partcorr(X::DiscreteTimeStationaryProcess, G::DiscreteTimeRegularGrid, ld::Bool=false)
    if ld  # use Levinson-Durbin
        cseq = covseq(X, G)
        pseq, sseq, rseq = LevinsonDurbin(cseq)
        return rseq
    else
        return partcorr!(Vector{Float64}(length(G)), X, G)
    end
end


"""
    autocov(X::IncrementProcess{T, P}, t::T, s::T) where {T, P}

Return the auto-covariance function of a process of increments. The computation is done via the auto-covariance of the parent process.
"""
function autocov(X::IncrementProcess{T, P}, t::T, s::T) where {T, P}
    proc = X.parent_process
    δ = X.step

    return autocov(proc, (t+δ), (s+δ)) - autocov(proc, (t+δ), (s)) - autocov(proc, (t), (s+δ)) + autocov(proc, (t), (s))
end


#### Fractional Brownian Motion ####
"""
Fractional Brownian motion.

# Members
* hurst: the Hurst exponent
"""
struct FractionalBrownianMotion<:SSSIProcess
    hurst::Float64

    function FractionalBrownianMotion(hurst::Float64)
        0. < hurst < 1. || error("Hurst exponent must be bounded in 0 and 1.")
        new(hurst)
    end
end

ss_exponent(X::FractionalBrownianMotion) = X.hurst


"""
    autocov(X::FractionalBrownianMotion, t::ContinuousTime, s::ContinuousTime)

Return the autocovariance function of fBm:
    1/2 * (|t|^{2H} + |s|^{2H} - |t-s|^{2H})
"""
function autocov(X::FractionalBrownianMotion, t::ContinuousTime, s::ContinuousTime)
    twoh::Float64 = 2*X.hurst
    return 0.5 * (abs(t)^twoh + abs(s)^twoh - abs(t-s)^twoh)
end


# Moving average kernels of fBm
"""
K_+ kernel
"""
function Kplus(x::Real,t::Real,H::Real)
    # @assert t>0
    p::Float64 = H-1/2
    v::Float64 = 0
    if x<0
        v = (t-x)^p - (-x)^p
    elseif 0<=x<t
        v = (t-x)^p
    else
        v = 0
    end
    return v
end


"""
K_- kernel
"""
function Kminus(x::Real,t::Real,H::Real)
    p::Float64 = H-1/2
    v::Float64 = 0
    if x<=0
        v = 0
    elseif 0<x<=t
        v = -(x)^p
    else
        v = (x-t)^p - (x)^p
    end
    return v
end


"""
K_+ + K_- kernel
"""
Kppm(x, t, H) = Kplus(x,t,H) + Kminus(x,t,H)


"""
K_+ - K_- kernel
"""
Kpmm(x, t, H) = Kplus(x,t,H) - Kminus(x,t,H)


"""
Fractional Gaussian noise.

fGn is the (discrete-time) increment process of a fBm.
"""
struct FractionalGaussianNoise<:DiscreteTimeIncrementSSSIProcess{FractionalBrownianMotion}
    parent_process::FractionalBrownianMotion
    step::Float64

    function FractionalGaussianNoise(hurst::Float64, step::Float64=1.)
        step > 0 || error("Step must be > 0.")
        new(FractionalBrownianMotion(hurst), step)
    end
end

step(X::FractionalGaussianNoise) = X.step

ss_exponent(X::FractionalGaussianNoise) = X.parent_process.hurst


"""
    autocov(X::FractionalGaussianNoise, l::DiscreteTime)

Return the autocovariance function of fGn:
    1/2 δ^{2H} (|i-j+1|^{2H} + |i-j-1|^2H - 2|i-j|^{2H})
where δ is the step of increment.
"""
function autocov(X::FractionalGaussianNoise, l::DiscreteTime)
    twoh::Float64 = 2*X.parent_process.hurst
    return 0.5 * X.step^twoh * (abs(l+1)^twoh + abs(l-1)^twoh - 2*abs(l)^twoh)
end


# """
# Fractional Wavelet noise.

# fWn is the (discrete-time) process resulting from the filtering of a fBm by a wavelet.
# """
# struct FractionalWaveletNoise<:DiscreteTimeSSSIProcess{FractionalBrownianMotion}
#     parent_process::FractionalBrownianMotion
#     step::Float64

#     function FractionalGaussianNoise(hurst::Float64, step::Float64=1.)
#         step > 0 || error("Step must be > 0.")
#         new(FractionalBrownianMotion(hurst), step)
#     end
# end

# step(X::FractionalGaussianNoise) = X.step

# ss_exponent(X::FractionalGaussianNoise) = X.parent_process.hurst


# """
#     autocov(X::FractionalGaussianNoise, l::DiscreteTime)

# Return the autocovariance function of fGn:
#     1/2 δ^{2H} (|i-j+1|^{2H} + |i-j-1|^2H - 2|i-j|^{2H})
# where δ is the step of increment.
# """
# function autocov(X::FractionalGaussianNoise, l::DiscreteTime)
#     twoh::Float64 = 2*X.parent_process.hurst
#     return 0.5 * X.step^twoh * (abs(l+1)^twoh + abs(l-1)^twoh - 2*abs(l)^twoh)
# end


#### Fractional Integrated Process ####
"""
Fractional integrated process.

This is a stationary process defined by ∇^d X(t) = ε(t), where ∇^d, d in (-1/2, 1/2) is the fractional differential operator, and ε(t) are i.i.d. standard Gaussian variables.
"""
struct FractionalIntegrated<:DiscreteTimeStationaryProcess
    d::Float64

    function FractionalIntegrated(d::Float64)
        abs(d) < 0.5 || error("Order of fractional differential must be in the range (-0.5, 0.5).")
        # new(d, all(ar.==0) ? Float64[] : ar, all(ma.==0) ? Float64[] : ma)
        new(d)
    end
end

ss_exponent(X::FractionalIntegrated) = X.d + 1/2

partcorr(X::FractionalIntegrated, k::DiscreteTime) = X.d/(k-X.d)


function autocov(X::FractionalIntegrated, n::DiscreteTime)
    return n > 0 ? (n-1+X.d) / (n-X.d) * autocov(X, n-1) : gamma(1-2*X.d) / gamma(1-X.d)^2
end


"""
    autocov!(C::Vector{Float64}, X::FractionalIntegrated, G::DiscreteTimeRegularGrid)

Note: The covariance of a fractional integrated process is computed recursively, so we overload `autocov!` for reason of efficiency.
"""
function autocov!(C::Vector{Float64}, X::FractionalIntegrated, G::DiscreteTimeRegularGrid)
    # check dimension
    # @assert length(C) == length(G)

    V = zeros(G[end])
    V[1] = gamma(1-2*X.d) / gamma(1-X.d)^2  # cov(0)
    for n = 1:length(V)-1
        V[n+1] = V[n] * (n-1+X.d) / (n-X.d)
    end
    for n = 1:length(C)
        C[n] = V[G[n]]
    end
    return C
end


#### FARIMA Process ####
"""
Fractional Auto-regression Integrated Moving Average (FARIMA) process.
"""
struct FARIMA<:DiscreteTimeStochasticProcess
    d::Float64
    ar::Vector{Float64}
    ma::Vector{Float64}

    FARIMA(d::Float64, ar::Vector{Float64}, ma::Vector{Float64}) = new(d, ar, ma)

    # function FARIMA(d::Float64, ar::Vector{Float64}, ma::Vector{Float64})
    #     # (typeof(P)==Int64 && P==length(ar)) || error("Inconsistent parameters for auto-regression.")
    #     # (typeof(Q)==Int64 && Q==length(ma)) || error("Inconsistent parameters for moving average.")
    #     # abs(d) < 0.5 || error("Order of fractional differential must be in the range (-0.5, 0.5).")
    #     # new(d, all(ar.==0) ? Float64[] : ar, all(ma.==0) ? Float64[] : ma)
    #     new(d, ar, ma)
    # end
end

FARIMA(d::Float64) = FARIMA(d, Float64[], Float64[])

# parameters(X::FARIMA{P,Q}) where {P,Q} = Dict('p'=>P, 'd'=>X.d, 'q'=>Q)
# parameters(X::FARIMA{P,Q}) where {P,Q} = (P, X.d, Q)

# FARIMA(d::Float64, ar::Vector{Float64}, ma::Vector{Float64}) = FARIMA{length(ar), length(ma)}(d, ar, ma)


############################

function fBm_covmat(G1::AbstractVector{<:Real}, G2::AbstractVector{<:Real}, H::Real)
    Σ = zeros(length(G1),length(G2))
    for c=1:length(G2), r=1:length(G1)
        Σ[r,c] = (abs(G1[r])^(2H) + abs(G2[c])^(2H) - abs(G1[r]-G2[c])^(2H))/2
    end
    return Σ
end

fBm_covmat(G::AbstractVector, H) = Matrix(Symmetric(fBm_covmat(G, G, H)))


function fBm_cond_mean(Gx::AbstractVector{<:Real}, Gy::AbstractVector{<:Real}, Y::AbstractVector{<:Real}, H::Real)
    @assert length(Gy) == length(Y)
    Σxy = fBm_covmat(Gx, Gy, H)
    Σyy = fBm_covmat(Gy, Gy, H)
    return Σxy * (Σyy\Y)
end


function fBm_cond_mean(gx::Real, Gy::AbstractVector{<:Real}, Y::AbstractVector{<:Real}, H::Real)
    return fBm_cond_mean([gx], Gy, Y, H)
end


function fBm_cond_cov(Gx::AbstractVector{<:Real}, Gy::AbstractVector{<:Real}, Y::AbstractVector{<:Real}, H::Real)
    Σxx = fBm_covmat(Gx, Gx, H)
    Σxy = fBm_covmat(Gx, Gy, H)
    Σyy = fBm_covmat(Gy, Gy, H)
    return Σxx - Σxy * inv(Σyy) * Σxy'
end


"""
Compute the covariance matrix of a fWn at some time lag.

# Args
- F: array of band pass filters (no DC component)
- d: time lag
- H: Hurst exponent
"""
function fWn_covmat_lag(F::AbstractVector{<:AbstractVector{T}}, d::Int, H::Real) where {T<:Real}
    L = maximum([length(f) for f in F])  # maximum length of filters
    # M = [abs(d+(n-m))^(2H) for n=0:L-1, m=0:L-1]  # matrix comprehension is ~ 10x slower
    M = zeros(L,L)
    for n=1:L, m=1:L
        M[n,m] = abs(d+(n-m))^(2H)
    end
    Σ = -1/2 * [f' * view(M, 1:length(f), 1:length(g)) * g for f in F, g in F]
end


"""
Compute the covariance matrix of a time-concatenated fWn.
"""
function fWn_covmat(F::AbstractVector{<:AbstractVector{T}}, lmax::Int, H::Real) where {T<:Real}
    J = length(F)
    Σ = zeros(((lmax+1)*J, (lmax+1)*J))
    Σs = [fWn_covmat_lag(F, d, H) for d = 0:lmax]

    for r = 0:lmax
        for c = 0:lmax
            Σ[(r*J+1):(r*J+J), (c*J+1):(c*J+J)] = (c>=r) ? Σs[c-r+1] : transpose(Σs[r-c+1])
        end
    end

    return Matrix(Symmetric(Σ))  #  forcing symmetry
end



# Auxiliary functions for the covariance of the  multifractional field
_D(h1,h2) = (h1==h2) ? 1. : sqrt(gamma(2*h1+1)*sin(pi*h1)*gamma(2*h2+1)*sin(pi*h2)) / (gamma(h1+h2+1)*sin(pi*(h1+h2)/2))
# or equivalently
# _F(h) = lgamma(2h+1) + log(sin(π*h))
# _D(h1,h2) = exp((_F(h1)+_F(h2))/2 - _F((h1+h2)/2))

# h actually corresponds to 2h here:
_gn(t,s,h) = (abs(t)^(h) + abs(s)^(h) - abs(t-s)^(h))/2  # non-stationary
_gs(t,h) = (abs(t+1)^(h) + abs(t-1)^(h) - 2*abs(t)^(h))/2  # stationary


"""
Covariance of multifractional Brownian motion.
"""
function mBm_cov(t1::Real, t2::Real, h1::Real, h2::Real)
    _D(h1, h2) * _gn(t1, t2, h1+h2)
end


function mBm_covmat(G1::AbstractVector{<:Real}, H1::AbstractVector{<:Real}, G2::AbstractVector{<:Real}, H2::AbstractVector{<:Real})
    @assert length(G1) == length(H1)
    @assert all(0 .< H1 .< 1)
    @assert length(G2) == length(H2)
    @assert all(0 .< H2 .< 1)

    N1, N2 = length(G1), length(G2)
    Σ = zeros(N1,N2)
    
    for c=1:N2, r=1:N1
        Σ[r,c] = mBm_cov(G1[r], G2[c], H1[r], H2[c])
    end
    return Σ
end

mBm_covmat(G::AbstractVector{<:Real}, H::AbstractVector{<:Real}) = mBm_covmat(G, H, G, H)

mBm_covmat(G::AbstractVector{<:Real}, H::Real) = mBm_covmat(G, H)




"""
(Approximate) Covariance of multifractional Gaussian noise.
"""
function mGn_cov(t1::Real, t2::Real, h1::Real, h2::Real)
    _D(h1, h2) * _gs(t1-t2, h1+h2)
end


function mGn_covmat(G1::AbstractVector{<:Real}, H1::AbstractVector{<:Real}, G2::AbstractVector{<:Real}, H2::AbstractVector{<:Real})
    @assert length(G1) == length(H1)
    @assert all(0 .< H1 .< 1)
    @assert length(G2) == length(H2)
    @assert all(0 .< H2 .< 1)

    N1, N2 = length(G1), length(G2)
    Σ = zeros(N1,N2)
    
    for c=1:N2, r=1:N1
        Σ[r,c] = mGn_cov(G1[r], G2[c], H1[r], H2[c])
    end
    return Σ
end


mGn_covmat(G::AbstractVector{<:Real}, H::AbstractVector{<:Real}) = mGn_covmat(G, H, G, H)

mGn_covmat(G::AbstractVector{<:Real}, H::Real) = mGn_covmat(G, H)


