"""
Stochastic process
"""
abstract type AbstractStochasticProcess{T<:Number}<:Sampleable{Univariate, Continuous} end
abstract type StochasticProcess<:AbstractStochasticProcess{Real} end
abstract type StationaryStochasticProcess<:StochasticProcess end

"""
Sampling grid for the stochastic process
"""
const SamplingGrid = Union{UnitRange{Int64}, Vector{Float64}}  # Regular grid or arbitrary collection of points

# virtual function of auto-covariance, to be implemented
autocov(X::StochasticProcess, i::Real, j::Real) = throw(NotImplementedError("autocov(::$(typeof(X)), ::Real, ::Real)"))

function autocov!(C::Matrix{Float64}, X::StochasticProcess, G::SamplingGrid)
    # check dimension
    @assert size(C, 1) == size(C, 2) == length(G)

    # construct the covariance matrix (a symmetric matrix)
    N = length(G)  # dimension of the auto-covariance matrix
    for c = 1:N, r = 1:c
        C[r,c] = autocov(X, G[c], G[r])
    end
    for c = 1:N, r = (c+1):N
        C[r,c] = C[c,r]
    end
    return C
end

autocov(X::StochasticProcess, G::SamplingGrid) = autocov!(Matrix{Float64}(length(G),length(G)), X, G)

autocov(X::StationaryStochasticProcess, i::Real) = throw(NotImplementedError("autocov(::$(typeof(X)), ::Real)"))
autocov(X::StationaryStochasticProcess, i::Real, j::Real) = autocov(X, i-j)

function autocov!(C::Vector{Float64}, X::StationaryStochasticProcess, G::UnitRange{Int64})
    # check dimension
    @assert length(C) == length(G)

    # construct the auto-covariance kernel
    for n = 1:length(C)
        C[n] = autocov(X, G[n]-G[1])
    end
    return C
end

function autocov!(C::Matrix{Float64}, X::StationaryStochasticProcess, G::UnitRange{Int64})
    # check dimension
    @assert size(C, 1) == size(C, 2) == length(G)

    # construct the covariance matrix (a Toeplitz matrix)
    N = length(G)
    knl = autocov!(zeros(N), X, G)  # autocovariance sequence
    for c = 1:N, r = 1:N
        C[r,c] = knl[abs(r-c)+1]
    end
    return C
end

# function autocov!(C::Matrix{Float64}, X::StationaryStochasticProcess, G::Vector{Float64})
#     try
#         idx = convert(Vector{Int64}, G)
#         all(diff(idx) .== 1) || throw(InexactError("Failed to convert the irregular sampling grid."))
#         return autocov!(C, X, idx[1]:idx[end])
#     catch
#         return invoke(autocov!, Tuple{Matrix{Float64}, StochasticProcess, SamplingGrid}, C, X, G)
#     end
# end

autocov(X::StationaryStochasticProcess, G::SamplingGrid) = autocov!(Matrix{Float64}(length(G), length(G)), X, G)


# partial correlation function
partcorr(X::StationaryStochasticProcess, n::Int64) = throw(NotImplementedError("partcorr(::$(typeof(X)), ::Int64)"))

function partcorr!(C::Vector{Float64}, X::StationaryStochasticProcess, G::UnitRange{Int64})
    # check dimension
    @assert length(C) == length(G)

    for n = 1:length(G)
        C[n] = partcorr(X, G[n]-G[1]+1)
    end
    return C
end

partcorr(X::StationaryStochasticProcess, G::UnitRange{Int64}) = partcorr!(Vector{Float64}(length(G)), X, G)


# Fractional Brownian motion
struct FractionalBrownianMotion<:StochasticProcess
    hurst::Float64

    function FractionalBrownianMotion(hurst::Float64)
        0. < hurst < 1. || error("Hurst exponent must be bounded in 0 and 1.")
        new(hurst)
    end
end

function autocov(X::FractionalBrownianMotion, i::Real, j::Real)
    twoh::Float64 = 2*X.hurst
    return 0.5 * (abs(i)^twoh + abs(j)^twoh - abs(i-j)^twoh)
end


# Fractional Gaussian noise
struct FractionalGaussianNoise<:StationaryStochasticProcess
    hurst::Float64

    function FractionalGaussianNoise(hurst::Float64)
        0. < hurst < 1. || error("Hurst exponent must be bounded in 0 and 1.")
        new(hurst)
    end
end

function autocov(X::FractionalGaussianNoise, l::Real)
    twoh::Float64 = 2*X.hurst
    return 0.5 * (abs(l+1)^twoh + abs(l-1)^twoh - 2*abs(l)^twoh)
end


# convert(::Type{FractionalGaussianNoise}, p::FractionalBrownianMotion) = FractionalGaussianNoise(p.hurst)
# convert(::Type{FractionalBrownianMotion}, p::FractionalGaussianNoise) = FractionalBrownianMotion(p.hurst)


# Auto-regression Fractional Integrated Moving Average (ARFIMA) process
struct ARFIMA{P, Q}<:StationaryStochasticProcess
    d::Float64
    ar::Vector{Float64}
    ma::Vector{Float64}

    function ARFIMA{P,Q}(d::Float64, ar::Vector{Float64}, ma::Vector{Float64}) where {P,Q}
        (typeof(P)==Int64 && P==length(ar)) || error("Inconsistent parameters for auto-regression.")
        (typeof(Q)==Int64 && Q==length(ma)) || error("Inconsistent parameters for moving average.")
        abs(d) < 0.5 || error("Fractional difference must be in the range (-0.5, 0.5).")
        # new(d, all(ar.==0) ? Float64[] : ar, all(ma.==0) ? Float64[] : ma)
        new(d, ar, ma)
    end
end

# parameters(X::ARFIMA{P,Q}) where {P,Q} = Dict('p'=>P, 'd'=>X.d, 'q'=>Q)
# parameters(X::ARFIMA{P,Q}) where {P,Q} = (P, X.d, Q)

ARFIMA(d::Float64) = ARFIMA{0,0}(d, Float64[], Float64[])

ARFIMA(d::Float64, ar::Vector{Float64}, ma::Vector{Float64}) = ARFIMA{length(ar), length(ma)}(d, ar, ma)

partcorr(X::ARFIMA{0,0}, k::Int64) = X.d/(k-X.d)

function autocov(X::ARFIMA{0,0}, n::Int64)
    return n > 0 ? (n-1+X.d) / (n-X.d) * autocov(X, n-1) : gamma(1-2*X.d) / gamma(1-X.d)^2
end

function autocov!(C::Vector{Float64}, X::ARFIMA{0,0}, G::UnitRange{Int64})
    # check dimension
    @assert length(C) == length(G)

    C[1] = gamma(1-2*X.d) / gamma(1-X.d)^2  # cov(0)
    for n = 1:length(C)-1
        C[n+1] = C[n] * (n-1+X.d) / (n-X.d)
    end
    return C
end

