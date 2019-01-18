######### Estimators for fBm and related processes #########


###### Power law estimator ######

@doc raw"""
    powlaw_estim(X::AbstractMatrix{<:Real}, lags::AbstractVector{<:Integer}; p::Real=2., method::Symbol=:optim)

Power-law estimator for Hurst exponent and volatility.

# Args
- X: Matrix of fGn, each row is a fGn of some time lag and each column is an observation.
- lags: time lags (increment step) used to compute each component of `X`
- p: power of the moment
- methods: method of estimation: {:optim, :lm, :glm}

# Returns
- hurst, σ: estimation of Hurst and volatility, as well as an object of optimizer
- (xp, yp): vectors of regression

# Notes
- `X` is computed from fBm by taking finite differences. The second dimension corresponds to time. Example, let `W` be a fBm sample path then the following command computes `X`:
```julia
julia> lags = 2:10
julia> X = transpose(lagdiff(W, lags, :causal))
```
- `p=1` is robust against quantization error.
- Y = (y_k)_k, with y_k := 𝐄[|Δ_{kδ} B^{H}(t)|^p], X = (x_k)_k, with x_k := p * log(k)
"""
function powlaw_estim(X::AbstractMatrix{<:Real}, lags::AbstractVector{<:Integer}; p::Real=2., method::Symbol=:optim)
    @assert length(lags) == size(X,1) > 1
    @assert all(lags .>= 1)
    # @assert p > 0. && kt > 0

    idx = findall(vec(.!any(isnan.(X), dims=1)))
    # println(length(idx))
    # println(idx)
    X = view(X,:,idx)  # remove columns containing NaN

    xp = p * log.(lags)
    μX = mean(X, dims=2)
    yp = vec(log.(mean((abs.(X.-μX)).^p, dims=2)))

    hurst, η = NaN, NaN

    # old version with weights
    # # polynomial order of the weight for samples, if 0 the uniform weight is used
    # kt::Integer = 0  # non-zero value puts more weight on most recent samples (i.e. those at large column numbers).
    # wt = StatsBase.weights(causal_weight(size(X,2), kt))
    # μX = mean(X, wt, 2)
    # yp = vec(log.(mean((abs.(X.-μX)).^p, wt, 2)))
    # # yp = vec(log.(mean((abs.(X)).^p, wt, 2)))  # <- this gives lower SNR
    # xp = p * log.(lags)
    # # weight for scales
    # ks::Integer = 0  # hard-coded: polynomial order of weight for scales, if 0 the uniform weight is used
    # ws = StatsBase.weights(poly_weight(length(yp), ks))
    # yc = yp .- mean(yp, ws)
    # xc = xp .- mean(xp, ws)
    # func = h -> 1/2 * sum(ws .* (yc - h*xc).^2)
    # # func = h -> 1/2 * sum(ws .* abs.(yc - h*xc))

    # estimation of H and η
    if method==:optim
        yc = yp .- mean(yp)
        xc = xp .- mean(xp)
        func = h -> 1/2 * sum((yc - h*xc).^2)
        # Gradient-free constrained optimization
        ɛ = 1e-2  # search hurst in the interval [ɛ, 1-ɛ]
        opm = Optim.optimize(func, ε, 1-ε, Optim.Brent())
        # # Gradient-based optimization
        # optimizer = Optim.GradientDescent()  # e.g. Optim.BFGS(), Optim.GradientDescent()
        # opm = Optim.optimize(func, ε, 1-ε, [0.5], Optim.Fminbox(optimizer))
        hurst = Optim.minimizer(opm)[1]
        η = mean(yp - hurst*xp)
    elseif method==:lm  # linear model by hand
        # by manual inversion
        Ap = hcat(xp, ones(length(xp))) # design matrix
        hurst, η = Ap \ yp
    elseif method==:glm  # using GLM package, same as lm
        dg = DataFrames.DataFrame(xvar=xp, yvar=yp)
        opm = GLM.lm(@GLM.formula(yvar~xvar), dg)
        η, hurst = GLM.coef(opm)
    else
        error("Unknown method $(method).")
    end

    cp = 2^(p/2) * gamma((p+1)/2)/sqrt(pi)  # constant depending on p
    σ = exp((η-log(cp))/p)

    return hurst, σ, (xp, yp)
end

"""
    powlaw_estim(X::AbstractVector{<:Real}, lags::AbstractVector{<:Integer}; kwargs...)

# Args
- X: sample path of fBm
- lags:
"""
function powlaw_estim(X::AbstractVector{<:Real}, lags::AbstractVector{<:Integer}; kwargs...)
    dX = transpose(lagdiff(X, lags, :causal))  # take transpose s.t. each column is an observation
    return powlaw_estim(dX, lags; kwargs...)
end


####### Generalized scalogram #######

"""
B-Spline scalogram estimator for Hurst exponent and volatility.

# Args
- S: vector of scalogram, ie, variance of the wavelet coefficients per scale.
- sclrng: scale of wavelet transform. Each number in `sclrng` corresponds to one row in the matrix X
- v: vanishing moments
- p: power by which the scalogram is computed
"""
function bspline_scalogram_estim(S::AbstractVector{T}, sclrng::AbstractVector{Int}, v::Int; p::Real=2., mode::Symbol=:center) where {T<:Real}
    @assert length(S) == length(sclrng)

    C = 2^(p/2) * gamma((p+1)/2)/sqrt(pi)

    # res = IRLS(log.(S), p*log.(sclrng), p)
    # hurst::Float64 = res[1][1]-1/2
    # β::Float64 = res[1][2][1]  # returned value is a scalar in a vector form
    # ols::Float64 = NaN

    df = DataFrames.DataFrame(xvar=log.(sclrng.^p), yvar=log.(S))
    ols = GLM.lm(@GLM.formula(yvar~xvar), df)
    coef = GLM.coef(ols)
    β::Float64 = coef[1]
    hurst::Float64 = coef[2]-1/2

    σ::Float64 = try
        Aρ = Aρ_bspline(0, 1, hurst, v, mode)
        exp((β - log(C) - log(abs(Aρ))*p/2)/p)
    catch
        NaN
    end

    return (hurst, σ), ols

    # Ar = hcat(xr, ones(length(xr)))  # design matrix
    # H0, η = Ar \ yr  # estimation of H and β
    # hurst = H0-1/2
    # A = Aρ_bspline(0, r, hurst, v, mode)
    # σ = exp((η - log(abs(A)))/2)
    # return hurst, σ
end

const fBm_bspline_scalogram_estim = bspline_scalogram_estim


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
const fBm_gen_bspline_scalogram_estim = gen_bspline_scalogram_estim


###### MLE ######

"""
Safe evaluation of the inverse quadratic form
    trace(X' * inv(A) * X)
where the matrix `A` is symmetric and positive definite.
"""
function xiAx(A::AbstractMatrix{<:Real}, X::AbstractVecOrMat{<:Real}, ε::Real=0)
    # Sanity check
    @assert issymmetric(A)
    @assert size(X, 1) == size(A, 1)

    # a simple version would be:
    # return tr(X' * pinv(A) * X)

    # SVD is equivalent to eigen decomposition on covariance matrix
    # U, S, V = svd(A)
    S, U = eigen(A)  # so that U * Diagonal(S) * inv(U) == A, in particular, U' == inv(U)
    idx = (S .> ε)  # shrinkage of small eigen values for stability

    if length(idx) > 0
        return sum((U[:,idx]'*X).^2 ./ S[idx])
    else
        error("Invalide covariance matrix.")
    end
end


"""
    log_likelihood_H(A, X)

Safe evaluation of the log-likelihood of a fBm model with the implicite σ (optimal in the MLE sense).

The value of log-likelihood (up to some additive constant) is
    -1/2 * (N*log(X'*inv(A)*X) + logdet(A))

# Args
- A: covariance matrix
- X: sample vector or matrix. For matrix each column is a sample.

# Notes
- This function is common to all MLEs with the covariance matrix of form `σ²A(h)`, where `{σ, h}` are unknown parameters. This kind of MLE can be carried out in `h` uniquely and `σ` is obtained from `h`.
"""
function log_likelihood_H(A::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}, ε::Real=0)
    # Sanity check
    @assert issymmetric(A)
    @assert size(X, 1) == size(A, 1)

    N = size(X,2) # number of i.i.d. samples in data

    # U, S, V = svd(A)
    S, U = eigen(A)  # so that U * Diagonal(S) * inv(U) == A, in particular, U' == inv(U)
    idx = (S .> ε)

    val = -1/2 * (length(X)*log(sum((U[:,idx]'*X).^2 ./ S[idx])) + N*sum(log.(S[idx])))  # non-constant part of log-likelihood

    return val - length(X)*log(2π*exp(1)/length(X))/2  # with the constant part
end

log_likelihood_H(A::AbstractMatrix, X::AbstractVector, args...) = log_likelihood_H(A, reshape(X,:,1), args...)


"""
Log-likelihood of a general Gaussian vector.

# Args
- A: covariance matrix
- X: sample matrix, each column is one sample.
"""
function log_likelihood(A::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real})
    @assert issymmetric(A)
    @assert size(X, 1) == size(A, 1)

    N = size(X,2) # number of i.i.d. samples in data
    return -1/2 * (N*logdet(A) + xiAx(A,X) + length(X)*log(2π))
end

log_likelihood(A::AbstractMatrix, X::AbstractVector) = log_likelihood(A, reshape(X,:,1))


#### fWn-MLE ####

function fWn_log_likelihood_H(X::AbstractVecOrMat{<:Real}, H::Real, ψ::AbstractVector{<:Real}, G::AbstractVector{<:Integer}=Int[])
    proc = FractionalWaveletNoise(H, ψ)

    Σ::AbstractMatrix = if length(G)>0
        @assert length(G) == size(X,1)
        covmat(proc, G)
    else
        covmat(proc, size(X,1))
    end

    return log_likelihood_H(Σ, X)
end

"""
Maximum likelihood estimation of Hurst exponent and volatility for fractional Wavelet noise.

# Args
- X: sample vector or matrix.
- ψ: wavelet filter used for computing `X`.
- G: integer time grid of `X`, by default the regular grid `1:size(X,1)` is used.
- method, ε: see `fWn_MLE_estim()`.
"""
function fWn_MLE_estim(X::AbstractVecOrMat{<:Real}, ψ::AbstractVector{<:Real}, G::AbstractVector{<:Integer}=Int[]; method::Symbol=:optim, ε::Real=1e-2)
    # @assert 0. < ε < 1.
    if length(G)>0
        @assert length(G) == size(X,1)
        # @assert minimum(abs.(diff(sort(G)))) > 0  # all elements are distinct
    end

    func = h -> -fWn_log_likelihood_H(X, h, ψ, G)

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

    proc = FractionalWaveletNoise(hurst, ψ)
    Σ = covmat(proc, length(G)>0 ? G : size(X,1))
    σ = sqrt(xiAx(Σ, X) / length(X))
    L = log_likelihood_H(Σ, X)

    return hurst, σ, L, opm
end


"""
Accelerated fWn-MLE by dividing a large vector of samples into smaller ones.

The MLE method can be expensive on data of large dimensions due to the inversion of covariance matrix. This function accelerates the MLE method by dividing a large vector `X` into smaller vectors of size `s` downsampled by a factor `l`. The smaller vectors are treated by MLE as i.i.d. samples.

# Args
- X, ψ: same as in `fWn_MLE_estim()`
- s,l: sub window size, length of decorrelation

# Notes
- This function works only with regular grid.
"""
function fWn_MLE_estim(X::AbstractVector{<:Real}, ψ::AbstractVector{<:Real}, s::Integer, l::Integer; kwargs...)
    t, V = rolling_vectorize(X, s, 1, l; mode=:causal)
    return fWn_MLE_estim(V, ψ; kwargs...)  # regular grid is implicitely used here.
end


#### fGn-MLE ####
# A special case of fWn-MLE which deserves it own implementation.

fGn_filter = d -> vcat(1, zeros(d-1), -1)

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

"""
Maximum likelihood estimation of Hurst exponent and volatility for fractional Gaussian noise.

# Args
- X: sample vector or matrix.
- d: time lag of the finite difference operator used for computing `X`.
- G: integer time grid of `X`, by default the regular grid `1:size(X,1)` is used.
- method, ε: see `fWn_MLE_estim()`.

# Notes
- The MLE is known for its sensitivivity to mis-specification of model. In particular the fGn-MLE is sensitive to NaN value and outliers.
"""
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


#### fWnb-MLE ####

"""
Safe evaluation of the log-likelihood of a fWm model with the implicite σ (optimal in the MLE sense).
"""
function fWnb_log_likelihood_H(X::AbstractVecOrMat{<:Real}, F::AbstractVector{<:AbstractVector{<:Real}}, H::Real, G::AbstractVector{<:Integer}=Int[])
    proc = FractionalWaveletNoiseBank(H, F)

    # covariance matrix of fWnb
    Σ::AbstractMatrix = if length(G)>0
        @assert size(X,1) == length(F) * length(G)
        covmat(proc, G)
    else
        covmat(proc, size(X,1)÷length(F))  # max time lag
    end
    # println(size(Σ))
    # println(size(X))
    # println(typeof(Σ))
    return log_likelihood_H(Σ, X)
end


"""
Maximum likelihood estimation of Hurst exponent and volatility for fractional wavelet noise bank (fWnb).

# Args
- X: sample vector or matrix
- F: array of filters
- G: integer time grid of `X`, by default the regular grid is used.
- method: :optim for optimization based or :table for look-up table based solution.
- ε: this defines the bounded constraint [ε, 1-ε], and for method==:table this is also the step of search for Hurst exponent.

# Returns
- hurst, σ: estimation
- L: log-likelihood of estimation
- opm: object of optimizer, for method==:optim only

# Notes
- The fWnb process is multivariate. An observation at time `t` is a `d`-dimensional vector, where `d` equals to the number of filters used in fWnb. A vector `X` is the concatenation of observations made on some time grid `G`, while a matrix `X` is a collection of i.i.d. sample vectors. Hence the row dimension of `X` must be `length(F) * length(G)`, if `G` is ever provided.
"""
function fWnb_MLE_estim(X::AbstractVecOrMat{<:Real}, F::AbstractVector{<:AbstractVector{<:Real}}, G::AbstractVector{<:Integer}=Int[]; method::Symbol=:optim, ε::Real=1e-2)
    # @assert 0. < ε < 1.
    if length(G)>0
        @assert size(X,1) == length(F) * length(G)
        @assert minimum(abs.(diff(sort(G)))) > 0  # all elements are distinct
    else
        @assert size(X,1) % length(F) == 0
    end

    func = h -> -fWnb_log_likelihood_H(X, F, h, G)

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

    proc = FractionalWaveletNoiseBank(hurst, F)
    Σ = covmat(proc, length(G)>0 ? G : size(X,1)÷length(F))
    σ = sqrt(xiAx(Σ, X) / length(X))
    L = log_likelihood_H(Σ, X)

    return hurst, σ, L, opm
end


function fWnb_MLE_estim(X::AbstractVector{<:Real}, F::AbstractVector{<:AbstractVector{<:Real}}, s::Integer, l::Integer; kwargs...)
    t, V = rolling_vectorize(X, s, 1, l; mode=:causal)
    return fWnb_MLE_estim(V, F; kwargs...)  # regular grid is implicitely used here.
end

# function fWn_swt_MLE_estim(X::AbstractVecOrMat{T}, wvl::String, level::Int; method::Symbol=:optim, ε::Real=1e-2) where {T<:Real}
#     F = [_intscale_bspline_filter(s, v)/sqrt(s) for s in sclrng]  # extra 1/sqrt(s) factor due to the implementation of DCWT
#     return fWn_MLE_estim(X, F; method=method, ε=ε)
# end


"""
fWn-MLE based on B-Spline wavelet transform.

# Args
- X: DCWT coefficients, each column corresponding to a vector of coefficients. See `cwt_bspline()`.
- sclrng: integer scales of DCWT
- v: vanishing moments of B-Spline wavelet
"""
function fWn_bspline_MLE_estim(X::AbstractVecOrMat{T}, sclrng::AbstractVector{Int}, v::Int; method::Symbol=:optim, ε::Real=1e-2) where {T<:Real}
    F = [_intscale_bspline_filter(s, v)/sqrt(s) for s in sclrng]  # extra 1/sqrt(s) factor due to the implementation of DCWT
    return fWn_MLE_estim(X, F; method=method, ε=ε)
end
# const fBm_bspline_MLE_estim = fWn_bspline_MLE_estim



function fGn_MLE_estim2(X::AbstractVecOrMat{<:Real}, d::Integer, G::AbstractVector{<:Integer}=Int[]; kwargs...)
    fWn_MLE_estim(X, fGn_filter(d), G; kwargs...)
end

function fGn_MLE_estim2(X::AbstractVector{<:Real}, d::Integer, s::Integer, l::Integer; kwargs...)
    fWn_MLE_estim(X, fGn_filter(d), s, l; kwargs...)
end

function fGn_MLE_estim3(X::AbstractVecOrMat{<:Real}, d::Integer, G::AbstractVector{<:Integer}=Int[]; kwargs...)
    fWnb_MLE_estim(X, [fGn_filter(d)], G; kwargs...)
end

function fGn_MLE_estim3(X::AbstractVector{<:Real}, d::Integer, s::Integer, l::Integer; kwargs...)
    fWnb_MLE_estim(X, [fGn_filter(d)], s, l; kwargs...)
end


"""TODO
Multiscale fGn-MLE
"""
function ms_fGn_MLE_estim(X::AbstractVector{T}, lags::AbstractVector{Int}, w::Int) where {T<:Real}
    Hs = zeros(length(lags))
    Σs = zeros(length(lags))

    for (n,lag) in enumerate(lags)  # time lag for finite difference
        # vectorization with window size w
        dXo = rolling_vectorize(X[lag+1:end]-X[1:end-lag], w, 1, 1)
        # rolling mean with window size 2lag, then down-sample at step lag
        dX = rolling_mean(dXo, 2lag, lag; boundary=:hard)

        (hurst_estim, σ_estim), obj = fGn_MLE_estim(squeezedims(dX), lag)

        Hs[n] = hurst_estim
        Σs[n] = σ_estim
    end

    return Hs, Σs
end


##### B-Spline DCWT MLE (Not maintained) #####
# Implementation based on DCWT formulation, not working well in practice.

function fBm_bspline_covmat_lag(H::Real, v::Int, l::Int, sclrng::AbstractVector{Int}, mode::Symbol)
    return Amat_bspline(H, v, l, sclrng) .* [sqrt(i*j) for i in sclrng, j in sclrng].^(2H+1)
end


"""
Compute the covariance matrix of B-Spline DCWT coefficients of a pure fBm.

The full covariance matrix of `J`-scale transform and of time-lag `N` is a N*J-by-N*J symmetric matrix.

# Args
- l: maximum time-lag
- sclrng: scale range
- v: vanishing moments of B-Spline wavelet
- H: Hurst exponent
- mode: mode of convolution
"""
function fBm_bspline_covmat(l::Int, sclrng::AbstractVector{Int}, v::Int, H::Real, mode::Symbol)
    J = length(sclrng)
    Σ = zeros(((l+1)*J, (l+1)*J))
    Σs = [fBm_bspline_covmat_lag(H, v, d, sclrng, mode) for d = 0:l]

    for r = 0:l
        for c = 0:l
            Σ[(r*J+1):(r*J+J), (c*J+1):(c*J+J)] = (c>=r) ? Σs[c-r+1] : transpose(Σs[r-c+1])
        end
    end

    return Matrix(Symmetric(Σ))  #  forcing symmetry
    # return [(c>=r) ? Σs[c-r+1] : Σs[r-c+1]' for r=0:N-1, c=0:N-1]
end


"""
Evaluate the log-likelihood of B-Spline DCWT coefficients.
"""
function fBm_bspline_log_likelihood_H(X::AbstractVecOrMat{T}, sclrng::AbstractVector{Int}, v::Int, H::Real, mode::Symbol) where {T<:Real}
    @assert 0 < H < 1
    @assert size(X,1) % length(sclrng) == 0

    L = size(X,1) ÷ length(sclrng)  # integer division: \div
    # N = ndims(X)>1 ? size(X,2) : 1

    Σ = fBm_bspline_covmat(L-1, sclrng, v, H, mode)  # full covariance matrix

    # # strangely, the following does not work (logarithm of a negative value)
    # iΣ = pinv(Σ)  # regularization by pseudo-inverse
    # return -1/2 * (J*N*log(trace(X'*iΣ*X)) + logdet(Σ))

    return log_likelihood_H(Σ, X)
end


"""
B-Spline wavelet-MLE estimator.
"""
function fBm_bspline_DCWT_MLE_estim(X::AbstractVecOrMat{T}, sclrng::AbstractVector{Int}, v::Int, mode::Symbol; method::Symbol=:optim, ε::Real=1e-2) where {T<:Real}
    @assert size(X,1) % length(sclrng) == 0
    # number of wavelet coefficient vectors concatenated into one column of X
    L = size(X,1) ÷ length(sclrng)  # integer division: \div
    # N = ndims(X)>1 ? size(X,2) : 1

    func = x -> -fBm_bspline_log_likelihood_H(X, sclrng, v, x, mode)

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

    Σ = fBm_bspline_covmat(L-1, sclrng, v, hurst, mode)
    σ = sqrt(xiAx(Σ, X) / length(X))

    return (hurst, σ), opm
end
