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
Return the autocovariance function of fBm:
    1/2 * (|t|^{2H} + |s|^{2H} - |t-s|^{2H})
"""
function autocov(X::FractionalBrownianMotion, t::Real, s::Real)
    twoh::Real = 2*X.hurst
    return 0.5 * (abs(t)^twoh + abs(s)^twoh - abs(t-s)^twoh)
end


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
Fractional Gaussian noise (fGn) is the (discrete-time) increment process of a fBm.
"""
struct FractionalGaussianNoise <: IncrementProcess{FractionalBrownianMotion}
    parent_process::FractionalBrownianMotion
    step::Real

    function FractionalGaussianNoise(hurst::Real, step::Real=1.)
        step > 0 || error("Step must be > 0.")
        new(FractionalBrownianMotion(hurst), step)
    end
end

step(X::FractionalGaussianNoise) = X.step

ss_exponent(X::FractionalGaussianNoise) = X.parent_process.hurst

filter(X::FractionalGaussianNoise) = [-1, 1]

"""
Return the autocovariance function of fGn:
    1/2 δ^{2H} (|i-j+1|^{2H} + |i-j-1|^2H - 2|i-j|^{2H})
where δ is the step of increment.
"""
function autocov(X::FractionalGaussianNoise, l::DiscreteTime)
    twoh::Real = 2*X.parent_process.hurst
    return 0.5 * X.step^twoh * (abs(l+1)^twoh + abs(l-1)^twoh - 2*abs(l)^twoh)
end

######## Fractional Wavelet noise (fWn) ########
"""
Fractional Wavelet noise.

fWn is the process resulting from the filtering of a fBm by a wavelet.
"""
struct FractionalWaveletNoise <: FilteredProcess{ContinuousTime, FractionalBrownianMotion}
    parent_process::FractionalBrownianMotion
    filter::AbstractVector{<:Real}

    function FractionalWaveletNoise(hurst::Real, filter::AbstractVector{<:Real})
        new(FractionalBrownianMotion(hurst), filter)
    end
end

ss_exponent(X::FractionalWaveletNoise) = X.parent_process.hurst

filter(X::FractionalWaveletNoise) = X.filter

