
"""
Abstract real Gaussian stochastic process.
"""
abstract type StochasticProcess{T<:TimeStyle}<:Sampleable{Univariate, Continuous} end

const DiscreteTimeStochasticProcess = StochasticProcess{DiscreteTime}
const ContinuousTimeStochasticProcess = StochasticProcess{ContinuousTime}

"""
Abstract stationary process.
"""
abstract type StationaryProcess{T}<:StochasticProcess{T} end

const DiscreteTimeStationaryProcess = StationaryProcess{DiscreteTime}
const ContinuousTimeStationaryProcess = StationaryProcess{ContinuousTime}

doc"""
Abstract self-similar process.

A process $(X(t))_t$ is self-similar with exponent $H\geq 0$ if $X(ct)$ and $c^H X(t)$ have the probability distribution for any $t$ and $c>0$, moreover we require $X(0)=0$ which implies that $X$ is continuous at $0$.

## Fields
* `ss_exponent`: self-similar exponent $H$
"""
abstract type SelfSimilarProcess<:ContinuousTimeStochasticProcess end

"""
    ss_exponent(X::SelfSimilarProcess)

Return the self-similar exponent of the process.
"""
ss_exponent(X::SelfSimilarProcess) = throw(NotImplementedError("ss_exponent(::$(typeof(X)))"))

doc"""
Stationary increments process of some self-similar process.

The increments of a process $(B_H(t))_{t\geq 0}$ with step $\delta>0$ is the discrete time process $(X^\delta_H(n))_{n=0,1\ldots}$ defined by

$X^\delta_H(n) = B_H((n+1)\delta) - B_H(n\delta)$

We always suppose $\delta=1$. In fact the following relation holds between the autocovariance function of $X^\delta_H$ and $X^1_H$:

$\gamma_{X^{\delta}_H}(n) = \delta^{2H} \gamma_{X^1_H}(n)$

Hence the step $\delta$ can be singled out in computations involving only the auto-covariance function.

## Fields
* `parent_process`: an object of the parent self-similar process

## Example
A fractional Brownian motion (fBm) is self-similar process with stationary increments. Its increments process is called fractional Gaussian noise (fGn).
"""
abstract type IncrementProcess{T<:SelfSimilarProcess}<:DiscreteTimeStationaryProcess end

# doc"""
#     step(X::IncrementProcess)

# Return the step of increment $\delta$ of the process.
# """
# step(X::IncrementProcess) = throw(NotImplementedError("incr_step(::$(typeof(X)))"))

"""
    ss_exponent(X::IncrementProcess)

Return the self-similar exponent of the parent process.
"""
ss_exponent(X::IncrementProcess) = throw(NotImplementedError("ss_exponent(::$(typeof(X)))"))

doc"""
    autocov(X::StochasticProcess{T}, i::T, j::T) where T

Auto-covariance function $\gamma_X(i,j)$ of a stochastic process $X$.
"""
autocov(X::StochasticProcess{T}, i::T, j::T) where T = throw(NotImplementedError("autocov(::$(typeof(X)), ::$(T), ::$(T))"))

"""
    autocov!(C::Matrix{Float64}, X::StochasticProcess, G::SamplingGrid)

Compute the auto-covariance matrix of a stochastic process on a sampling grid.
"""
function autocov!(C::Matrix{Float64}, X::StochasticProcess{T}, G::SamplingGrid{<:T}) where T
    # check dimension
    @assert size(C, 1) == size(C, 2) <= length(G)
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
autocov(X::StochasticProcess{T}, G::SamplingGrid{<:T}) where T = autocov!(Matrix{Float64}(length(G),length(G)), X, G)
# autocov(X::StationaryProcess, G::SamplingGrid) = autocov!(Matrix{Float64}(length(G), length(G)), X, G)

"""
    covmat(X::StochasticProcess, G::SamplingGrid)

Alias to `autocov(X::StochasticProcess, G::SamplingGrid)`.
"""
covmat(X::StochasticProcess{T}, G::SamplingGrid{<:T}) where T = autocov(X, G)

doc"""
    autocov(X::StationaryProcess, i::Real)

Auto-covarince function $\gamma_X(i)$ of a stationary stochastic process $X$.
"""
autocov(X::StationaryProcess{T}, i::T) where T = throw(NotImplementedError("autocov(::$(typeof(X)), ::Real)"))
autocov(X::StationaryProcess{T}, i::T, j::T) where T = autocov(X, i-j)

"""
    autocov!(C::Vector{Float64}, X::StationaryProcess, G::RegularGrid)

Compute the auto-covarince sequence of a stationary process on a regular grid.
"""
function autocov!(C::Vector{Float64}, X::StationaryProcess{T}, G::RegularGrid{<:T}) where T
    # check dimension
    @assert length(C) <= length(G)

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
    @assert size(C, 1) == size(C, 2) <= length(G)

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
    partcorr(X::StationaryProcess, n::Int64)

Return the partial correlation function of a stationary process.
"""
partcorr(X::DiscreteTimeStationaryProcess, n::DiscreteTime) = throw(NotImplementedError("partcorr(::$(typeof(X)), ::$(typeof(n)))"))

"""
    partcorr!(C::Vector{Float64}, X::StationaryProcess, G::UnitRange)

Compute the partial correlation sequence of a stationary process for a range of index.
"""
function partcorr!(C::Vector{Float64}, X::DiscreteTimeStationaryProcess, G::DiscreteTimeRegularGrid)
    # check dimension
    @assert length(C) <= length(G)

    for n = 1:length(C)
        C[n] = partcorr(X, G[n]-G[1]+1)
    end
    return C
end

"""
    partcorr(X::StationaryProcess, G::UnitRange, ld::Bool=false)

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

doc"""
    autocov(X::IncrementProcess, i::Int64, j::Int64)

Return the auto-covariance function of a process of increments. The computation is done via the auto-covariance of the parent process.
"""
function autocov(X::IncrementProcess, i::DiscreteTime, j::DiscreteTime)
    P = X.parent_process

    return autocov(P, (i+1), (j+1)) - autocov(P, (i+1), (j)) -
    autocov(P, (i), (j+1)) + autocov(P, (i), (j))
end

# autocov(X::IncrementProcess, i::DiscreteTime) = autocov(X, i, 0)

doc"""
Fractional Brownian motion.

## Fields
* hurst: the Hurst exponent
"""
struct FractionalBrownianMotion<:SelfSimilarProcess
    hurst::Float64

    function FractionalBrownianMotion(hurst::Float64)
        0. < hurst < 1. || error("Hurst exponent must be bounded in 0 and 1.")
        new(hurst)
    end
end

ss_exponent(X::FractionalBrownianMotion) = X.hurst

doc"""
    autocov(X::FractionalBrownianMotion, i::Real, j::Real)

Return the autocovariance function of fBm:
    $\gamma_{B_H}(i,j) = \frac 1 2 (|i|^{2H} + |j|^2H - |i-j|^{2H})
"""
function autocov(X::FractionalBrownianMotion, i::ContinuousTime, j::ContinuousTime)
    twoh::Float64 = 2*X.hurst
    return 0.5 * (abs(i)^twoh + abs(j)^twoh - abs(i-j)^twoh)
end

doc"""
Fractional Gaussian noise.
"""
struct FractionalGaussianNoise<:IncrementProcess{FractionalBrownianMotion}
    hurst::Float64
    parent_process::FractionalBrownianMotion

    function FractionalGaussianNoise(hurst::Float64)
        new(hurst, FractionalBrownianMotion(hurst))
    end
end

FractionalGaussianNoise(P::FractionalBrownianMotion) = FractionalGaussianNoise(P.hurst)

ss_exponent(X::FractionalGaussianNoise) = X.hurst

doc"""
    autocov(X::FractionalGaussianNoise, l::Int64)

Return the autocovariance function of fGn:
    $\gamma_{X^1_H}(i,j) = \frac 1 2 (|i-j+1|^{2H} + |i-j-1|^2H - 2|i-j|^{2H})
"""
function autocov(X::FractionalGaussianNoise, l::DiscreteTime)
    twoh::Float64 = 2*X.hurst
    return 0.5 * (abs(l+1)^twoh + abs(l-1)^twoh - 2*abs(l)^twoh)
end

# power_law_exponent(X::FractionalGaussianNoise) = 2*X.hurst

doc"""
Fractional integrated process.

This is a stationary process defined by

$\nabla^d X(t) = \varepsilon(t)$

where $\nabla^d, d\in(-1/2, 1/2)$ is the fractional differential operator, and $\varepsilon(t)\sim \mathcal{N}(0,1)$ are i.i.d. Gaussian variables.
"""
struct FractionalIntegrated<:DiscreteTimeStationaryProcess
    d::Float64

    function FractionalIntegrated(d::Float64)
        abs(d) < 0.5 || error("Order of fractional differential must be in the range (-0.5, 0.5).")
        # new(d, all(ar.==0) ? Float64[] : ar, all(ma.==0) ? Float64[] : ma)
        new(d)
    end
end

partcorr(X::FractionalIntegrated, k::DiscreteTime) = X.d/(k-X.d)

function autocov(X::FractionalIntegrated, n::DiscreteTime)
    return n > 0 ? (n-1+X.d) / (n-X.d) * autocov(X, n-1) : gamma(1-2*X.d) / gamma(1-X.d)^2
end

"""
    autocov!(C::Vector{Float64}, X::FractionalIntegrated, G::RegularGrid)

# Notes
* The covariance of a fractional integrated process is computed recursively, so we overload `autocov!` for reason of efficiency.
* `G::RegularGrid` is not used in the computation.
"""
function autocov!(C::Vector{Float64}, X::FractionalIntegrated, G::DiscreteTimeRegularGrid)
    # check dimension
    @assert length(C) <= length(G)

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



doc"""
    LevinsonDurbin(cseq::Vector{Float64})

Diagonalization of a symmetric positive definite Toeplitz matrix using Levinson-Durbin (LD) method.

# Arguments
- `cseq::Vector{Float64})`: covariance sequence of a stationary process

# Outputs
- `pseq::Vector{Vector{Float64}}`: linear prediction coefficients
- `sseq`: variances of residual
- `rseq`: partial correlation coefficients

# Explanation
`pseq` forms the lower triangular matrix diagonalizing the covariance matrix $\Gamma$, and `sseq` forms the resulting diagonal matrix. `rseq[n]` is just `pseq[n][n]`.
"""
function LevinsonDurbin(cseq::Vector{Float64})
    N = length(cseq)

    if N > 1
        # check that cseq is a validate covariance sequence
        @assert cseq[1] > 0
        @assert all(abs.(cseq[2:end]) .<= cseq[1])
        # @assert all(diff(abs.(cseq)) .<= 0)

        # initialization
        # pseq: linear prediction coefficients
        pseq = Vector{Vector{Float64}}(N-1); pseq[1] = [cseq[2]/cseq[1]]
        # sseq: variances of residual
        sseq = zeros(N); sseq[1] = cseq[1]; sseq[2] = (1-pseq[1][1]^2) * sseq[1]
        # rseq: partial correlation coefficients
        rseq = zeros(N-1); rseq[1] = pseq[1][1]

        # recursive construction of the prediction coefficients and variances
        for n=2:N-1
            pseq[n] = zeros(n)
            pseq[n][n] = (cseq[n+1] - cseq[2:n]' * pseq[n-1][end:-1:1]) / sseq[n]
            pseq[n][1:n-1] = pseq[n-1] - pseq[n][n] * pseq[n-1][end:-1:1]
            sseq[n+1] = (1 - pseq[n][n]^2) * sseq[n]
            rseq[n] = pseq[n][n]
        end
    else
        pseq = Vector{Float64}[]
        sseq = copy(cseq)
        rseq = Float64[]
    end
    return pseq, sseq, rseq
end

LevinsonDurbin(p::StationaryProcess{T}, g::RegularGrid{<:T}) where T = LevinsonDurbin(covseq(p, g))
