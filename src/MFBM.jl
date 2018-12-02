######## Multiple fBm ########
"""
Multiple fBm defined on [0, Tmax]

# Members
- hurstfunc: the time varying Hurst function
- hurstvec: the time varying Hurst vector
- Tmax: the ending time
"""
struct MultipleFractionalBrownianMotion <: ContinuousTimeStochasticProcess
    hurstfunc::Function  # hurst(t) is the hurst exponent at time t
    hurstvec::AbstractVector{<:Real}
    Tmax::Real

    function FractionalBrownianMotion(hurstfunc::Function)
        new(hurstfunc, Float64[], +Inf)
    end

    function FractionalBrownianMotion(hurstvec::AbstractVector{<:Real}, Tmax::Real=1.)
        all(0. .< hurstvec .< 1.) || error("Hurst exponent must be bounded in 0 and 1.")
        @assert Tmax > 0.

        N = length(hurstvec)
        hurstfunc = t -> (t<0 || t>Tmax) ? NaN : hurst[max(1, round(Int, (t/Tmax)*N))]
        new(hurstfunc, hurstvec, Tmax)
    end
end


# Auxiliary functions for the covariance of the  multifractional field
_D(h1, h2) = (h1==h2) ? 1. : sqrt(gamma(2*h1+1)*sin(pi*h1)*gamma(2*h2+1)*sin(pi*h2)) / (gamma(h1+h2+1)*sin(pi*(h1+h2)/2))
# or equivalently
# _F(h) = lgamma(2h+1) + log(sin(π*h))
# _D(h1,h2) = exp((_F(h1)+_F(h2))/2 - _F((h1+h2)/2))

# h actually corresponds to 2h here:
_gn(t, s, h) = (abs(t)^(h) + abs(s)^(h) - abs(t-s)^(h))/2  # non-stationary
_gs(t, h) = (abs(t+1)^(h) + abs(t-1)^(h) - 2*abs(t)^(h))/2  # stationary

_mfBm_autocov(t1, t2, h1, h2) = _D(h1, h2) * _gn(t1, t2, h1+h2)

function autocov(P::MultipleFractionalBrownianMotion, t::Real, s::Real)
    return _mfBm_autocov(t, s, P.hurst(t), P.hurst(s))
end

function autocov(P::MultipleFractionalBrownianMotion, t::Int, s::Int)
    N = length(P.hurstvec)
    return _mfBm_autocov(t/N * P.Tmax, s/N * P.Tmax, P.hurstvec[t], P.hurstvec[s])
end



struct MultipleFractionalGaussianNoise <: IncrementProcess{MultipleFractionalBrownianMotion}
    parent_process::MultipleFractionalBrownianMotion
    step::Real

    hurstfunc::Function  # hurst(t) is the hurst exponent at time t
    hurstvec::AbstractVector{<:Real}
    Tmax::Real

    function MultipleFractionalGaussianNoise(hurstfunc::Function, step::Real=1.)
        0. < step || error("Invalid value for step.")
        new(MultipleFractionalBrownianMotion(hurstfunc), step)
    end

    function MultipleFractionalGaussianNoise(hurstvec::Real, Tmax::Real, step::Real=1.)
        0. < step < Tmax || error("Invalid value for step.")
        new(MultipleFractionalBrownianMotion(hurstvec, Tmax), step)
    end
end


"""
(Approximate) Covariance of multifractional Gaussian noise.
"""
_mfGn_autocov(t1, t2, h1, h2) = _D(h1, h2) * _gs(t1-t2, h1+h2)

function autocov(P::MultipleFractionalGaussianNoise, t::Real, s::Real)
    return _mfGn_autocov(t, s, P.hurst(t), P.hurst(s))
end

function autocov(P::MultipleFractionalGaussianNoise, t::Int, s::Int)
    N = length(P.hurstvec)
    return _mfGn_autocov(t/N * P.Tmax, s/N * P.Tmax, P.hurstvec[t], P.hurstvec[s])
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


####### TODO : fWn ########
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
