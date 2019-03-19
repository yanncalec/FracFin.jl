########## Collections of some old functions (Not maintained and not exposed to main package) ##########

#### fBm-MLE ####

"""
    fBm_log_likelihood_H(X, H, ψ, G)

Log-likelihood of a fBm model with the optimal volatility.
"""
function fBm_log_likelihood_H(X::AbstractVecOrMat{<:Real}, H::Real, G::AbstractVector{<:Integer}=Int[])
    proc = FractionalBrownianMotion(H)

    Σ::AbstractMatrix = if length(G)>0
        @assert length(G) == size(X,1)
        @assert any(diff(G) .> 0)  # grid points are in increasing order
        covmat(proc, G)
    else
        covmat(proc, size(X,1))
    end

    return log_likelihood_H(Σ, X-X[1])  # relative to the first point
end


"""
Maximum likelihood estimation of Hurst exponent and volatility for fBm.

# Args
- X: sample vector or matrix. For matrix each column is a sample.
- ψ: wavelet filter used for computing `X`.
- G: integer time grid of `X`, by default the regular grid `1:size(X,1)` is used.
- method: `:optim` for optimization based or `:table` for lookup table based procedure
- ε: search hurst in the range [ε, 1-ε]

# Notes
- The MLE is known for its sensitivivity to mis-specification of model. In particular the fGn-MLE is sensitive to NaN value and outliers.
"""
function fBm_MLE_estim(X::AbstractVecOrMat{<:Real}, G::AbstractVector{<:Integer}=Int[]; method::Symbol=:optim, ε::Real=1e-2)
    # @assert 0. < ε < 1.
    if length(G)>0
        @assert length(G) == size(X,1)
        @assert minimum(abs.(diff(sort(G)))) > 0  # all elements are distinct
    end

    func = h -> -fBm_log_likelihood_H(X, h, G)

    opm = nothing
    hurst = nothing

    if method == :optim
        # Gradient-free constrained optimization
        opm = Optim.optimize(func, ε, 1-ε, Optim.Brent())
        # # Gradient-based optimization
        # optimizer = Optim.GradientDescent()  # e.g. Optim.BFGS(), Optim.GradientDescent()
        # opm = Optim.optimize(func, ε, 1-ε, [0.5], Optim.Fminbox(optimizer))
        hurst = Optim.minimizer(opm)[1]
    elseif method == :table
        Hs = collect(ε:ε:1-ε)
        hurst = Hs[argmin([func(h) for h in Hs])]
    else
        throw("Unknown method: ", method)
    end

    proc = FractionalBrownianMotion(hurst)
    Σ = covmat(proc, length(G)>0 ? G : size(X,1))
    σ = sqrt(xiAx(Σ, X) / length(X))
    L = log_likelihood_H(Σ, X)
    # # or equivalently
    # L = log_likelihood(σ^2*Σ, X)

    return hurst, σ, L, opm
end


"""
Accelerated fBm-MLE by dividing a large vector of samples into smaller ones.

The MLE method can be expensive on data of large dimensions due to the inversion of covariance matrix. This function accelerates the MLE method by dividing a large vector `X` into smaller vectors of size `s` downsampled by a factor `l`. The smaller vectors are treated by MLE as i.i.d. samples.

# Args
- X: same as in `fWn_MLE_estim()`
- s: sub window size
- l: length of decorrelation
"""
function fBm_MLE_estim(X::AbstractVector{<:Real}, s::Integer, l::Integer; kwargs...)
    t, V = rolling_vectorize(X, s, 1, l; mode=:causal)
    return fBm_MLE_estim(V; kwargs...)  # The regular grid is implicitely used here.
end



#### fGn-MLE ####
# A special case of fWn-MLE which deserves it own implementation.

function fGn_log_likelihood_H(X::AbstractVecOrMat{<:Real}, H::Real, d::Int, G::AbstractVector{<:Integer}=Int[])
    proc = FractionalGaussianNoise(H, d)

    Σ::AbstractMatrix = if length(G)>0
        @assert length(G) == size(X,1)
        covmat(proc, G)
    else
        covmat(proc, size(X,1))
    end

    return log_likelihood_H(Σ, X)
end


function fGn_MLE_estim(X::AbstractVecOrMat{<:Real}, d::Integer, G::AbstractVector{<:Integer}=Int[]; method::Symbol=:optim, ε::Real=1e-2)
    # @assert 0. < ε < 1.
    if length(G)>0
        @assert length(G) == size(X,1)
        # @assert minimum(abs.(diff(sort(G)))) > 0  # all elements are distinct
    end

    func = h -> -fGn_log_likelihood_H(X, h, d, G)

    opm = nothing
    hurst = nothing

    if method == :optim
        # Gradient-free constrained optimization
        opm = Optim.optimize(func, ε, 1-ε, Optim.Brent())
        # # Gradient-based optimization
        # optimizer = Optim.GradientDescent()  # e.g. Optim.BFGS(), Optim.GradientDescent()
        # opm = Optim.optimize(func, ε, 1-ε, [0.5], Optim.Fminbox(optimizer))
        hurst = Optim.minimizer(opm)[1]
    elseif method == :table
        Hs = collect(ε:ε:1-ε)
        hurst = Hs[argmin([func(h) for h in Hs])]
    else
        error("Unknown method: $method")
    end

    proc = FractionalGaussianNoise(hurst, d)
    Σ = covmat(proc, length(G)>0 ? G : size(X,1))
    σ = sqrt(xiAx(Σ, X) / length(X))
    L = log_likelihood_H(Σ, X)
    # or equivalently
    # L = log_likelihood(σ^2*Σ, X)

    return hurst, σ, L, opm
end


"""
Accelerated fGn-MLE by dividing a large vector of samples into smaller ones.

The MLE method can be expensive on data of large dimensions due to the inversion of covariance matrix. This function accelerates the MLE method by dividing a large vector `X` into smaller vectors of size `s` downsampled by a factor `l`. The smaller vectors are treated by MLE as i.i.d. samples.

# Args
- X, d: same as in `fGn_MLE_estim()`
- s,l: sub window size, length of decorrelation
"""
function fGn_MLE_estim(X::AbstractVector{<:Real}, d::Integer, s::Integer, l::Integer; kwargs...)
    t, V = rolling_vectorize(X, s, 1, l; mode=:causal)
    return fGn_MLE_estim(V, d; kwargs...)
end



# function bspline_scalogram_estim(X::AbstractMatrix{<:Real}, sclrng::AbstractVector{<:Integer}, v::Integer, pows::AbstractVector{<:Real}; kwargs...)

#     bspline_scalogram_estim()
# end


# """
# B-Spline scalogram estimator with a matrix of DCWT coefficients as input. Each column in `W` is a vector of DCWT coefficients.
# """
# function fBm_bspline_scalogram_estim(W::AbstractMatrix{T}, sclrng::AbstractVector{Int}, v::Int; dims::Int=1, mode::Symbol=:center) where {T<:Real}
#     return fBm_bspline_scalogram_estim(var(W,dims), sclrng, v; mode=mode)
# end

# """
# B-Spline scalogram estimator with an array of DCWT coefficients as input. Each row in `W` corresponds to a scale.
# """
# function fBm_bspline_scalogram_estim(W::AbstractVector{T}, sclrng::AbstractVector{Int}, v::Int; mode::Symbol=:center) where {T<:AbstractVector{<:Real}}
#     return fBm_bspline_scalogram_estim([var(w) for w in W], sclrng, v; mode=mode)
# end


"""
Generalized B-Spline scalogram estimator for Hurst exponent and volatility.

# Args
- Σ: covariance matrix of wavelet coefficients.
- sclrng: scale of wavelet transform. Each number in `sclrng` corresponds to one row in the matrix X
- v: vanishing moments
- r: rational ratio defining a line in the covariance matrix, e.g. r=1 corresponds to the main diagonal.
"""
function gen_bspline_scalogram_estim(Σ::AbstractMatrix{T}, sclrng::AbstractVector{Int}, v::Int, r::Rational=1//1; mode::Symbol=:center) where {T<:Real}
    @assert issymmetric(Σ)
    @assert size(Σ,1) == length(sclrng)
    @assert r >= 1
    if r > 1
        all(diff(sclrng/sclrng[1]) .== 1) || error("Imcompatible scales: the ratio between the k-th and the 1st scale must be k")
    end

    p,q,N = r.num, r.den, length(sclrng)
    @assert N>=2p

    # Σ = cov(X, X, dims=2, corrected=true)  # covariance matrix

    yr = [log(abs(Σ[q*j, p*j])) for j in 1:N if p*j<=N]
    xr = [log(sclrng[q*j] * sclrng[p*j]) for j in 1:N if p*j<=N]

    df = DataFrames.DataFrame(xvar=xr, yvar=yr)
    ols = GLM.lm(@GLM.formula(yvar~xvar), df)
    coef = GLM.coef(ols)

    hurst = coef[2]-1/2
    Aρ = Aρ_bspline(0, r, hurst, v, mode)
    σ = exp((coef[1] - log(abs(Aρ)))/2)
    return (hurst, σ), ols

    # Ar = hcat(xr, ones(length(xr)))  # design matrix
    # H0, η = Ar \ yr  # estimation of H and β
    # hurst = H0-1/2
    # Aρ = Aρ_bspline(0, r, hurst, v, mode)
    # σ = ℯ^((η - log(abs(Aρ)))/2)
    # return hurst, σ
end
# const fBm_gen_bspline_scalogram_estim = gen_bspline_scalogram_estim


"""
Compute the covariance matrix of a fWn at some time lag.

# Args
- F: array of band pass filters (no DC component)
- d: time lag
- H: Hurst exponent
"""
function fWn_covmat_lag(F::AbstractVector{<:AbstractVector{T}}, d::DiscreteTime, H::Real) where {T<:Real}
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
