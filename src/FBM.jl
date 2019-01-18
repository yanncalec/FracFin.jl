######## Fractional Brownian Motion ########
"""
Fractional Brownian motion.

# Members
- hurst: the Hurst exponent
"""
struct FractionalBrownianMotion <: SSSIProcess
    hurst::Real

    function FractionalBrownianMotion(hurst::Real)
        0. < hurst < 1. || error("Hurst exponent must be bounded in 0 and 1.")
        new(hurst)
    end
end

ss_exponent(X::FractionalBrownianMotion) = X.hurst

@doc raw""" fBm_autocov(t,s,H)

Autocovariance function of standard fBm:
    $\frac 1 2 (|t|^{2H} + |s|^{2H} - |t-s|^{2H})$
"""
fBm_autocov = (t::Real,s::Real,H::Real) -> 1/2 * (abs(t)^(2H) + abs(s)^(2H) - abs(t-s)^(2H))

autocov(X::FractionalBrownianMotion, t::Real, s::Real) = fBm_autocov(t,s,X.hurst)

"""
    fBm_covmat(G1::AbstractVector{<:Real}, G2::AbstractVector{<:Real}, H::Real)

Compute the covariance matrix of fBm with hurst `H` between two grids `G1` and `G2`.
This gives the same result as `covmat(FractionalBrownianMotion(H), G1, G2)`.
"""
function fBm_covmat(G1::AbstractVector{<:Real}, G2::AbstractVector{<:Real}, H::Real)
    Σ = zeros(length(G1),length(G2))
    for (c,s) in enumerate(G2), (r,t) in enumerate(G1)
        Σ[r,c] = fBm_autocov(t, s, H)
    end
    return Σ
end

fBm_covmat(G, H) = Matrix(Symmetric(fBm_covmat(G, G, H)))

# Moving average kernels of fBm
"""
K_+ kernel
"""
function Kplus(x::Real, t::Real, H::Real)
    # @assert t>0
    p::Real = H-1/2
    v::Real = 0
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
function Kminus(x::Real, t::Real, H::Real)
    p::Real = H-1/2
    v::Real = 0
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
Kppm(x, t, H) = Kplus(x, t, H) + Kminus(x, t, H)


"""
K_+ - K_- kernel
"""
Kpmm(x, t, H) = Kplus(x, t, H) - Kminus(x, t, H)


######## Fractional Gaussian Noise ########
@doc raw"""
Fractional Gaussian noise (fGn) is the discrete time version of the continuous time differential process of a fBm: $ B(t+\delta) - B(t) $.  It is defined as $ B(n+l) - B(n) $, where `l` is the lag.

# Note
- The anti-causal convention adopted by the definition here has no impact since the process is stationary.
"""
struct FractionalGaussianNoise <: IncrementProcess{FractionalBrownianMotion}
    parent_process::FractionalBrownianMotion
    lag::Integer

    function FractionalGaussianNoise(hurst::Real, lag::Integer=1)
        lag >= 1 || error("Lag must be >= 1.")
        new(FractionalBrownianMotion(hurst), lag)
    end
end

ss_exponent(X::FractionalGaussianNoise) = X.parent_process.hurst
lag(X::FractionalGaussianNoise) = X.lag
step(X::FractionalGaussianNoise) = 1

@doc raw"""
    fGn_autocov(t::Real,H::Real,δ::Real)

Autocovariance function of stardard (continuous time) fGn: $ 1/2 (|t+δ|^{2H} + |t-δ|^{2H} - 2|t|^{2H}) $
"""
fGn_autocov = (t::Real,H::Real,δ::Real) -> 1/2 * (abs(t+δ)^(2H) + abs(t-δ)^(2H) - 2*abs(t)^(2H))

function autocov(X::FractionalGaussianNoise, n::Integer)
    return fGn_autocov(n, ss_exponent(X), lag(X))
end


"""
    fGn_covmat(G1::AbstractVector{<:Real}, G2::AbstractVector{<:Real}, H::Real, δ::Real)

Compute the covariance matrix of standard (continuous time) fractional Gaussian noise of Hurst exponent `H` and time lag `δ` between the grid `G1` and `G2`.

# Notes
The special case of discrete regular grids with integer lag `d` is equivalent to `covmat(FractionalGaussianNoise(H, d), 1:N, 1:M)`.
"""
function fGn_covmat(G1::AbstractVector{<:Real}, G2::AbstractVector{<:Real}, H::Real, δ::Real)
    Σ = zeros(length(G1),length(G2))
    for (c,s) in enumerate(G2), (r,t) in enumerate(G1)
        Σ[r,c] = fGn_autocov(t-s, H, δ)
    end
    return Σ
end

fGn_covmat(G::AbstractVector{<:Real}, H::Real, δ::Real) = Matrix(Symmetric(fGn_covmat(G, G, H, δ)))

fGn_covmat(N::Integer, H::Real, d::Integer) = fGn_covmat(1:N, H, d)


######## Fractional Wavelet noise (fWn) ########

@doc raw"""
Fractional Wavelet noise (fWn).

fWn is the filtered process of fBm by some high-pass banc filters, e.g. wavelet filters.

$$
W(t) = \sum_{n=0}^{p-1} B(t-n\delta) \psi[n]
$$
"""
struct FractionalWaveletNoise <: FilteredProcess{DiscreteTime, FractionalBrownianMotion}
    parent_process::FractionalBrownianMotion
    filter::AbstractVector{<:Real}
    coeff::AbstractVector{<:Real}  # coefficients for the summation of |t - d*l|^(2H)

    function FractionalWaveletNoise(hurst::Real, filter::AbstractVector{<:Real})
        @assert isapprox(sum(filter), 0; atol=1e-8) "Filter must not contain DC."
        new(FractionalBrownianMotion(hurst), filter, fWn_cov_coeff(filter))
    end
end

ss_exponent(X::FractionalWaveletNoise) = X.parent_process.hurst
filter(X::FractionalWaveletNoise) = X.filter
step(X::FractionalWaveletNoise) = 1

function fWn_cov_coeff(u::AbstractVector{<:Real}, v::AbstractVector{<:Real})
    m = length(u)
    n = length(v)

    w = zeros(promote_type(eltype(u),eltype(v)), m+n-1)
    for j=1:m, k=1:n
        w[j-k+n] += u[j] * v[k]  # corresponding to the range 1-n:m-1
    end
    return -w/2  # factor -1/2 comes from the covariance of fBm
end

fWn_cov_coeff(u::AbstractVector{<:Real}) = fWn_cov_coeff(u, u)

"""
Autocovariance function of stardard (continuous time) fWn.
"""
function fWn_autocov(X::FractionalWaveletNoise, t::Real, δ::Real)
    n = length(X.filter)
    return sum(X.coeff .* abs.(t.-δ*(1-n:n-1)).^(2*ss_exponent(X)))
end

function autocov(X::FractionalWaveletNoise, n::Integer)
    fWn_autocov(X, n, 1)
end


function fWn_covmat(X::FractionalWaveletNoise, G1::AbstractVector{<:Real}, G2::AbstractVector{<:Real}, δ::Real)
    Σ = zeros(length(G1),length(G2))
    for (c,s) in enumerate(G2), (r,t) in enumerate(G1)
        Σ[r,c] = fWn_autocov(X, t-s, δ)
    end
    return Σ
end

fWn_covmat(X::FractionalWaveletNoise, G::AbstractVector{<:Real}, δ::Real) = Matrix(Symmetric(fWn_covmat(X, G, G, δ)))


const MultiScaleFractionalWaveletNoise = Vector{FractionalWaveletNoise}
const msfWn = MultiScaleFractionalWaveletNoise

# fWn_cov_coeff(F::AbstractVector{<:AbstractVector}) = [(fWn_cov_coeff(u,v), (1-length(v):length(u)-1)) for u in F, v in F]

# struct FractionalWaveletNoise <: FilteredProcess{ContinuousTime, FractionalBrownianMotion}
#     parent_process::FractionalBrownianMotion
#     filters::AbstractVector{<:AbstractVector{<:Real}}
#     coeff::AbstractMatrix  # pair coefficients for the summation of |t - d*l|^(2H)

#     function FractionalWaveletNoise(hurst::Real, filters::AbstractVector)
#         @assert all(isapprox(sum(f), 0; atol=1e-8) for f in filters) "Filters must not contain DC."
#         new(FractionalBrownianMotion(hurst), filters, fWn_cov_coeff(filters))
#     end
# end

# function fWn_autocov(X::FractionalWaveletNoise, t::Real, δ::Real)
#     H = ss_exponent(X)
#     Σ = zeros(Real, size(X.coeff))
#     for r=1:length(X.filters), c=1:length(X.filters)
#         α, k = X.coeff[r,c]
#         Σ[r,c] = sum(α .* abs.(t-δ*k)^(2H))
#     end
#     return Σ
# end


# function fWn_covmat(X::FractionalWaveletNoise, G::Integer)
#     J = length(X.filters)
#     Σ = zeros(((lmax+1)*J, (lmax+1)*J))
#     Σs = [fWn_autocov(X, d, 1) for d = 0:lmax]

#     for r = 0:lmax
#         for c = 0:lmax
#             Σ[(r*J+1):(r*J+J), (c*J+1):(c*J+J)] = (c>=r) ? Σs[c-r+1] : transpose(Σs[r-c+1])
#         end
#     end

#     return Matrix(Symmetric(Σ))  #  forcing symmetry
# end


########################################################
# """
# Compute the covariance matrix of a fWn at some time lag.

# # Args
# - F: array of band pass filters (no DC component)
# - d: time lag
# - H: Hurst exponent
# """
# function fWn_covmat_lag(F::AbstractVector{<:AbstractVector{T}}, d::DiscreteTime, H::Real) where {T<:Real}
#     L = maximum([length(f) for f in F])  # maximum length of filters
#     # M = [abs(d+(n-m))^(2H) for n=0:L-1, m=0:L-1]  # matrix comprehension is ~ 10x slower
#     M = zeros(L,L)
#     for n=1:L, m=1:L
#         M[n,m] = abs(d+(n-m))^(2H)
#     end
#     Σ = -1/2 * [f' * view(M, 1:length(f), 1:length(g)) * g for f in F, g in F]
# end


# """
# Compute the covariance matrix of a time-concatenated fWn.
# """
# function fWn_covmat(F::AbstractVector{<:AbstractVector{T}}, lmax::Int, H::Real) where {T<:Real}
#     J = length(F)
#     Σ = zeros(((lmax+1)*J, (lmax+1)*J))
#     Σs = [fWn_covmat_lag(F, d, H) for d = 0:lmax]

#     for r = 0:lmax
#         for c = 0:lmax
#             Σ[(r*J+1):(r*J+J), (c*J+1):(c*J+J)] = (c>=r) ? Σs[c-r+1] : transpose(Σs[r-c+1])
#         end
#     end

#     return Matrix(Symmetric(Σ))  #  forcing symmetry
# end
