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
A fractional Wavelet noise (fWn) is discrete time version of the high-pass filtered process of fBm:
$$
W(t) = \sum_{n=0}^{p-1} B(t-n\delta) \psi[n]
$$

It is defined as
$$
W[k] = \sum_{n=0}^{p-1} B(k-n) \psi[n]
$$

# Notes
- fGn is a special case of fWn.
```julia
d = 1  # lag for fGn
fGn = FracFin.FractionalGaussianNoise(0.7, d)

filter = vcat(1,zeros(d-1),-1)
fWn = FracFin.FractionalWaveletNoise(0.7, filter)

G = FracFin.covmat(fGn, 10)
W = FracFin.covmat(fWn, 10)
maximum(abs.(G-W))  # close to 0
```
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


"""
# Notes
- `fWn_cov_coeff(u,v)` is the reverse of `fWn_cov_coeff(v,u)`.
"""
function fWn_cov_coeff(u::AbstractVector{<:Real}, v::AbstractVector{<:Real})
    m = length(u)
    n = length(v)

    w = zeros(promote_type(eltype(u),eltype(v)), m+n-1)
    for j=1:m, k=1:n
        w[j-k+n] += u[j] * v[k]  # corresponding to the range 1-n:m-1
    end

    # another implementation is the following:
    # # W = v .* u'
    # # w = [sum(diag(W,d)) for d=1-n:m-1]
    # however this is way much slower.

    return -w/2  # factor -1/2 comes from the covariance of fBm
end

fWn_cov_coeff(u::AbstractVector{<:Real}) = fWn_cov_coeff(u, u)

"""
Autocovariance function of stardard (continuous time) fWn.
"""
function fWn_autocov(X::FractionalWaveletNoise, t::Real, δ::Real)
    n = length(X.filter)
    return sum(X.coeff .* abs.(t .- δ*(1-n:n-1)).^(2*ss_exponent(X)))
end

autocov(X::FractionalWaveletNoise, n::Integer) = fWn_autocov(X, n, 1)


"""
    covmat(X::FractionalWaveletNoise, G1::AbstractVector{<:Real}, G2::AbstractVector{<:Real}, δ::Real=1)

Compute the covariance matrix between the grid `G1` and `G2` of a standard (continuous time) fWn `X` with step `δ`.

# Notes
- Like `fGn_covmat()`, this function is equivalent to `covmat()` in case of regular grids.
"""
function covmat(X::FractionalWaveletNoise, G1::AbstractVector{<:Real}, G2::AbstractVector{<:Real}, δ::Real=1)
    Σ = zeros(length(G1),length(G2))
    for (c,s) in enumerate(G2), (r,t) in enumerate(G1)
        Σ[r,c] = autocov(X, t-s, δ)
    end

    return Σ
end

covmat(X::FractionalWaveletNoise, G::AbstractVector{<:Real}, δ::Real) = Matrix(Symmetric(covmat(X, G, G, δ)))

# fWn_covmat(G1::AbstractVector{<:Real}, G2::AbstractVector{<:Real}, f::AbstractVector{<:Real}, H::Real, δ::Real) = covmat(FractionalWaveletNoise(H,f), G1, G2, δ)

function fWn_covmat(G1::AbstractVector{<:Real}, G2::AbstractVector{<:Real}, filter::AbstractVector{<:Real}, H::Real, δ::Real)
    coeff = fWn_cov_coeff(filter)
    n = length(filter)
    Σ = zeros(length(G1),length(G2))
    for (c,s) in enumerate(G2), (r,t) in enumerate(G1)
        Σ[r,c] = sum(coeff .* abs.((t-s) .- δ*(1-n:n-1)).^(2H))
    end

    return Σ
end

fWn_covmat(G::AbstractVector{<:Real}, f::AbstractVector{<:Real}, H::Real, δ::Real) = Matrix(Symmetric(fWn_covmat(G, G, f, H, δ)))


######## fWn bank (fWnb) ########

"""
A fractional Wavelet noise bank (fWnb) is a fBm filtered by a bank of filters, in other words, a collection of fWn.

# Notes
- This process is multivariate hence can not be properly handled in the scalar framework as fWn.
"""
struct FractionalWaveletNoiseBank <: AbstractRandomField # <: FilteredProcess{DiscreteTime, FractionalBrownianMotion}
    parent_process::FractionalBrownianMotion
    filters::AbstractVector
    coeffs::AbstractMatrix  # 2-tuple: pair coefficients for the summation of |t - d*l|^(2H) and the range of index

    function FractionalWaveletNoiseBank(hurst::Real, filters::AbstractVector{<:AbstractVector{<:Real}})
        @assert all(isapprox(sum(f), 0; atol=1e-8) for f in filters) "Filters must not contain DC."

        coeffs = [(fWn_cov_coeff(u,v), (1-length(v):length(u)-1)) for v in filters, u in filters]
        new(FractionalBrownianMotion(hurst), filters, coeffs)
    end
end

ss_exponent(X::FractionalWaveletNoiseBank) = X.parent_process.hurst
filter(X::FractionalWaveletNoiseBank) = X.filters
step(X::FractionalWaveletNoiseBank) = 1

# fWnb_cov_coeff(F::AbstractVector{<:AbstractVector}) = [(fWn_cov_coeff(u,v), (1-length(v):length(u)-1)) for v in F, u in F]

"""
Covariance of fWnb. This covariance is a matrix since fWnb is multivariate process.

# Args
- X: a fWnb process
- t:

# Notes
- filter at [r,c] is the reverse of that at [c,r]
- range of index at [r,c] is reverse of that at [c,r] times -1
By consequent the final matrix Σ(t) = Σ'(-t).
"""
function autocov(X::FractionalWaveletNoiseBank, t::Real, δ::Real)
    H = ss_exponent(X)
    Σ = zeros(Real, size(X.coeffs))
    for c=1:length(X.filters), r=1:length(X.filters)
        # X.coeffs[r,c] is (filter, range of index)
        Σ[r,c] = sum(X.coeffs[r,c][1] .* abs.(t .- δ*X.coeffs[r,c][2]).^(2H))
    end
    return Σ
end


"""
Compute the covariance matrix of standard (continuous time) fractional Wavelet noise of Hurst exponent `H` and time lag `δ` between the grid `G1` and `G2`.
"""
function covmat(X::FractionalWaveletNoiseBank, G1::AbstractVector{<:Real}, G2::AbstractVector{<:Real}, δ::Real)
    J = length(X.filters)
    Σ = zeros(length(G1)*J, length(G2)*J)
    for (c,s) in enumerate(G2), (r,t) in enumerate(G1)
        Σ[(r*J+1):(r*J+J), (c*J+1):(c*J+J)] = autocov(X, t-s, δ)
    end
    return Σ
end

function covmat(X::FractionalWaveletNoiseBank, G::AbstractVector{<:Real}, δ::Real)
    J = length(X.filters)
    Σ = zeros(length(G)*J, length(G)*J)
    for (c,s) in enumerate(G), (r,t) in enumerate(G)
        Σ[(r*J+1):(r*J+J), (c*J+1):(c*J+J)] = (r>=c) ? autocov(X, t-s, δ) : transpose(Σ[c,r])
    end

    return Matrix(Symmetric(Σ))
end

function covmat(X::FractionalWaveletNoiseBank, G::AbstractVector{<:Integer})
    if isregulargrid(G)
        J = length(X.filters)
        l = length(G)

        Σs = [autocov(X, G[n]-G[1], 1) for n=1:length(G)]
        Σ = zeros(l*J, l*J)
        for c=0:l-1,r=0:l-1
            Σ[(r*J+1):(r*J+J), (c*J+1):(c*J+J)] = (c>=r) ? Σs[c-r+1] : transpose(Σs[r-c+1])
        end

        return Matrix(Symmetric(Σ))  #  forcing symmetry
    else
        return covmat(X, G, 1)
    end
end

covmat(X::FractionalWaveletNoiseBank, l::Integer) = covmat(X, 1:l)


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
