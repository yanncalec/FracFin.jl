
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
