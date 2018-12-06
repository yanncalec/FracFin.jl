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

"""
Autocovariance function of standard fBm:
    1/2 * (|t|^{2H} + |s|^{2H} - |t-s|^{2H})
"""
fBm_autocov = (t::Real,s::Real,H::Real) -> 1/2 * (abs(t)^(2H) + abs(s)^(2H) - abs(t-s)^(2H))

autocov(X::FractionalBrownianMotion, t::Real, s::Real) = fBm_autocov(t,s,X.hurst)

"""
Covariance matrix of fBm.

This is equivalent to `covmat(FractionalBrownianMotion(H), G1, G2)`.
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
"""
Fractional Gaussian noise (fGn) is the discrete time version of the continuous time differential process of a fBm: `B(t+δ) - B(t)`.  It is defined as `B(n+l) - B(n)`, where `l` is the lag.

# Note
- The definition here is the anticausal version.
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

"""
Autocovariance function of stardard (continuous time) fGn:
    1/2 (|t+δ|^{2H} + |t-δ|^2H - 2|t|^{2H})
"""
fGn_autocov = (t::Real,H::Real,δ::Real) -> 1/2 * (abs(t+δ)^(2H) + abs(t-δ)^(2H) - 2*abs(t)^(2H))

function autocov(X::FractionalGaussianNoise, n::Integer)
    return fGn_autocov(n, ss_exponent(X), lag(X))
end


"""
Covariance matrix of fGn.

For discrete regular grid this is equivalent to `covmat(FractionalGaussianNoise(H), 1:N, 1:M)`.
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
"""
Fractional Wavelet noise.

fWn is the process resulting from the filtering of a fBm by a wavelet.
"""
struct FractionalWaveletNoise <: FilteredProcess{ContinuousTime, FractionalBrownianMotion}
    parent_process::FractionalBrownianMotion
    filter::AbstractVector

    function FractionalWaveletNoise(hurst::Real, filter::AbstractVector{<:Real})
        new(FractionalBrownianMotion(hurst), filter)
    end
end

ss_exponent(X::FractionalWaveletNoise) = X.parent_process.hurst

filter(X::FractionalWaveletNoise) = X.filter

fWn_autocov = () -> NaN