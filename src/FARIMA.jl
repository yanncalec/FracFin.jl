######## FARIMA Process #########
"""
Fractional Auto-regression Integrated Moving Average (FARIMA) process.
"""
struct FARIMA <: DiscreteTimeStochasticProcess
    d::Real
    ar::Vector{Float64}
    ma::Vector{Float64}

    FARIMA(d::Real, ar::Vector{<:AbstractFloat}, ma::Vector{<:AbstractFloat}) = new(d, ar, ma)

    # function FARIMA(d::Real, ar::Vector{Float64}, ma::Vector{Float64})
    #     # (typeof(P)==Int64 && P==length(ar)) || error("Inconsistent parameters for auto-regression.")
    #     # (typeof(Q)==Int64 && Q==length(ma)) || error("Inconsistent parameters for moving average.")
    #     # abs(d) < 0.5 || error("Order of fractional differential must be in the range (-0.5, 0.5).")
    #     # new(d, all(ar.==0) ? Float64[] : ar, all(ma.==0) ? Float64[] : ma)
    #     new(d, ar, ma)
    # end
end

FARIMA(d::Real) = FARIMA(d, Float64[], Float64[])

# parameters(X::FARIMA{P,Q}) where {P,Q} = Dict('p'=>P, 'd'=>X.d, 'q'=>Q)
# parameters(X::FARIMA{P,Q}) where {P,Q} = (P, X.d, Q)

# FARIMA(d::Float64, ar::Vector{Float64}, ma::Vector{Float64}) = FARIMA{length(ar), length(ma)}(d, ar, ma)


######## Fractional Integrated Process ########
"""
Fractional integrated process.

This is a stationary process defined by ∇^d X(t) = ε(t), where ∇^d, d in (-1/2, 1/2) is the fractional differential operator, and ε(t) are i.i.d. standard Gaussian variables.
"""
struct FractionalIntegrated <: DiscreteTimeStationaryProcess
    d::Real

    function FractionalIntegrated(d::Real)
        abs(d) < 0.5 || error("Order of fractional differential must be in the range (-0.5, 0.5).")
        # new(d, all(ar.==0) ? Float64[] : ar, all(ma.==0) ? Float64[] : ma)
        new(d)
    end
end

ss_exponent(X::FractionalIntegrated) = X.d + 1/2

partcorr(X::FractionalIntegrated, k::Integer) = X.d/(k-X.d)

function autocov(X::FractionalIntegrated, n::Integer)
    return n > 0 ? (n-1+X.d) / (n-X.d) * autocov(X, n-1) : gamma(1-2*X.d) / gamma(1-X.d)^2
end


"""
Note: The covariance of a fractional integrated process is computed recursively, so we overload `autocov!` for reason of efficiency.
"""
function autocov!(C::Vector{<:AbstractFloat}, X::FractionalIntegrated, G::AbstractVector{<:Integer})
    # check dimension
    # @assert length(C) == length(G)
    # @assert isregulargrid(G) && eltype(G)<:Int

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


