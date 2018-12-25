######### Estimators for fBm and related processes #########

###### Power law ######

"""
Power-law estimator for Hurst exponent and volatility.

Y = (y_k)_k, with y_k := 𝐄[|Δ_{kδ} B^{H}(t)|^p]
X = (x_k)_k, with x_k := p * log(k)

# Args
- X: Matrix of fGn, each row is one component (fGn at some time lag)
- lags: integer time lags (increment step) used to compute each component of `X`
- p: power of the moment
- kt: polynomial order of the weight for samples, if 0 the uniform weight is used

# Returns
- hurst, σ, obj: estimation of Hurst and volatility, as well as an object of optimizer.

# Notes
- `X` is computed from a sample path of fBm by taking finite difference of different time lags. The second dimension corresponds to time. Example, let `B` be a fBm sample path then the following command gives `X`:
```julia
julia> lags = 2:10
julia> X = transpose(lagdiff(W, lags, :causal))
- `p=1` is robust against quantization error
- The weight (i.e. `kt>0`) for samples is more important for the most recent ones (i.e. those at large column numbers).
```
"""
function powlaw_estim(X::AbstractMatrix{<:Real}, lags::AbstractVector{<:Integer}, p::Real=2.; kt::Integer=0)
    @assert length(lags) == size(X,1) > 1
    @assert all(lags .>= 1)
    # @assert p > 0. && kt > 0

    cp = 2^(p/2) * gamma((p+1)/2)/sqrt(pi)  # constant depending on p
    
    # observation and explanatory vectors
    wt = StatsBase.weights(causal_weight(size(X,2), kt))
    μX = mean(X, wt, 2)
    yp = vec(log.(mean((abs.(X.-μX)).^p, wt, 2)))
    # yp = vec(log.(mean((abs.(X)).^p, wt, 2)))  # <- this gives lower SNR
    xp = p * log.(lags)
    
    # Estimation method 1: optimization
    # weight for scales
    ks::Integer = 0  # polynomial order of the weight for scales, if 0 the uniform weight is used
    ws = StatsBase.weights(poly_weight(length(yp), ks))    
    yc = yp .- mean(yp, ws)
    xc = xp .- mean(xp, ws)
    func = h -> 1/2 * sum(ws .* (yc - h*xc).^2)
    # func = h -> 1/2 * sum(ws .* abs.(yc - h*xc))

    # estimation of H and η
    # Gradient-free constrained optimization
    ɛ::Real = 1e-2  # search hurst in the interval [ɛ, 1-ɛ]
    opm = Optim.optimize(func, ε, 1-ε, Optim.Brent())
    # # Gradient-based optimization
    # optimizer = Optim.GradientDescent()  # e.g. Optim.BFGS(), Optim.GradientDescent()
    # opm = Optim.optimize(func, ε, 1-ε, [0.5], Optim.Fminbox(optimizer))
    hurst = Optim.minimizer(opm)[1]
    η = mean(yp - hurst*xp, ws)

    # # Estimation method 2: linear regression
    # # by manual inversion
    # Ap = hcat(xp, ones(length(xp))) # design matrix
    # hurst, β = Ap \ yp
    # # or by GLM
    # dg = DataFrames.DataFrame(xvar=xp, yvar=yp)
    # opm = GLM.lm(@GLM.formula(yvar~xvar), dg)
    # η, hurst = GLM.coef(opm)

    σ = exp((η-log(cp))/p)

    return hurst, σ, opm
end

const fBm_powlaw_estim = powlaw_estim


"""
# Args
- s: index of working scale
- d: average downsampling factor
- k: steps of prediction in the future

# Notes
- This function is intended to be used with `rolling_apply` where `X` is some observation on a time window. - The time arrow is on the second dimension (i.e. horizontal) from left to right.
"""
function powlaw_estim_predict(X::AbstractMatrix{<:Real}, lags::AbstractVector{<:Integer}, p::Real, s::Integer, d::Integer, k::Integer; kwargs...)
    # @assert k>0 && d>0

    # println("l=$l, d=$d")
    H, σ, opm = FracFin.powlaw_estim(X, lags, p; kwargs...)

    # convention of time arrow: from left to right
    # tidx = reverse(size(X,2):-d:1)
    tidx = findall(reverse(logtrain(size(X,2), size(X,2)÷d)))    
    xm = X[s,tidx]

    # xm = transpose(vec2mat(X[s,:], d, keep=:tail))
    # tidx = d*(1:size(xm,1))

    μc, Σc = cond_mean_cov(FractionalGaussianNoise(H, lags[s]), tidx[end]+1:tidx[end]+k, tidx, xm)
    return H, σ, μc, σ^2 * Σc  # <- adding mean(X[l,tidx]) to μc won't help
    
end


"""
EXPERIMENTAL: prediction by recursive conditional mean

No visible improvements over classical conditional mean for long range prediction.
"""
function powlaw_estim_predict(X::AbstractMatrix{<:Real}, lags::AbstractVector{<:Integer}, p::Real, s::Integer, k::Integer; kwargs...)
    # @assert k>0 && d>0

    H, σ, opm = FracFin.powlaw_estim(X, lags, p; kwargs...)
    l = size(X,2)÷4
    Cv = cond_mean_coeff(FractionalGaussianNoise(H, lags[s]), k, l)
    μc = Cv * X[s, end-l+1:end]
    
    return H, σ, μc
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
    @assert issymmetric(A)
    @assert size(X, 1) == size(A, 1)

    # a simple version would be:  
    # return tr(X' * pinv(A) * X)

    # U, S, V = svd(A)
    S, U = eigen(A)  # so that U * Diagonal(S) * inv(U) == A, in particular, U' == inv(U)
    idx = (S .> ε)
    return sum((U[:,idx]'*X).^2 ./ S[idx])
end


"""
Safe evaluation of the log-likelihood of a fBm model with the implicit optimal volatility (in the MLE sense).

The value of log-likelihood (up to some additif constant) is
    -1/2 * (N*log(X'*inv(A)*X) + logdet(A))

# Args
- A: covariance matrix
- X: vector of matrix of observation. For matrix case the columns should be i.i.d. observations.

# Notes
- This function is common to all MLEs with the covariance matrix of form `σ²A(h)`, where `{σ, h}` are unknown parameters. This kind of MLE can be carried out in `h` uniquely and `σ` is obtained from `h`.
"""
function log_likelihood_H(A::AbstractMatrix{<:Real}, X::AbstractVecOrMat{<:Real}, ε::Real=0)
    @assert issymmetric(A)
    @assert size(X, 1) == size(A, 1)

    N = ndims(X)>1 ? size(X,2) : 1  # number of i.i.d. samples in data
    # d = size(X,1)  # such that N*d == length(X)

    # U, S, V = svd(A)
    S, U = eigen(A)  # so that U * Diagonal(S) * inv(U) == A, in particular, U' == inv(U)
    idx = (S .> ε)

    val = -1/2 * (length(X)*log(sum((U[:,idx]'*X).^2 ./ S[idx])) + N*sum(log.(S[idx])))  # non-constant part of log-likelihood
    
    return val - length(X)*log(2π*exp(1)/length(X))/2  # with the constant part
end


"""
Log-likelihood of a general Gaussian vector.

# Args
- A: covariance matrix
- X: sample vector or matrix, each column is one observation.
"""
function log_likelihood(A::AbstractMatrix{<:Real}, X::AbstractVecOrMat{<:Real})
    @assert issymmetric(A)
    @assert size(X, 1) == size(A, 1)

    N = ndims(X)>1 ? size(X,2) : 1  # number of observations
    # d = size(X,1), # dimension of the vector, such that N*d == length(X)

    return -1/2 * (N*logdet(A) + xiAx(A,X) + length(X)*log(2π))
end


#### fWn-MLE ####
# A fWn is the filtration of a fBm time series by a bank of high pass filters, eg, multiscale wavelet filters.

"""
H-dependent log-likelihood of fraction wavelet noise (fWn) with optimal σ.
"""
function fWn_log_likelihood_H(X::AbstractVecOrMat{T}, F::AbstractVector{<:AbstractVector{T}}, H::Real) where {T<:Real}
    @assert 0 < H < 1
    @assert size(X,1) % length(F) == 0

    Σ = Matrix(Symmetric(fWn_covmat(F, size(X,1)÷length(F)-1, H)))
    return log_likelihood_H(Σ, X)
end


"""
General fWn-MLE of Hurst exponent and volatility.

# Args
- X: transformed coefficients, each column is a vector of coefficient; or concatenation of vectors.
- F: array of filters, each corresponding to a row in X
- method: :optim for optimization based or :table for look-up table based solution.
- ε: this defines the bounded constraint [ε, 1-ε], and for method==:table this is also the step of search for Hurst exponent.

# Returns
- (hurst, σ): estimation
- L: log-likelihood of estimation
- opm: object of optimizer, for method==:optim only

# Notes
- X can also be the concatenation of vectors at at consecutive instants.
"""
function fWn_MLE_estim(X::AbstractVecOrMat{T}, F::AbstractVector{<:AbstractVector{T}}; method::Symbol=:optim, ε::Real=1e-2) where {T<:Real}
    @assert 0. < ε < 1.
    @assert size(X,1) % length(F) == 0

    func = h -> -fWn_log_likelihood_H(X, F, h)

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

    Σ = Matrix(Symmetric(fWn_covmat(F, size(X,1)÷length(F)-1, hurst)))
    σ = sqrt(xiAx(Σ, X) / length(X))
    L = log_likelihood_H(Σ, X)

    return (hurst, σ), L, opm
end


"""
Doc
"""
function fWn_swt_MLE_estim(X::AbstractVecOrMat{T}, wvl::String, level::Int; method::Symbol=:optim, ε::Real=1e-2) where {T<:Real}
    F = [_intscale_bspline_filter(s, v)/sqrt(s) for s in sclrng]  # extra 1/sqrt(s) factor due to the implementation of DCWT
    return fWn_MLE_estim(X, F; method=method, ε=ε)
end


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


#### fGn-MLE ####
# A special case of fWn-MLE which deserves it own implementation.

function fGn_log_likelihood_H(X::AbstractVecOrMat{T}, H::Real, d::Int) where {T<:Real}
    Σ = covmat(FractionalGaussianNoise(H, d), size(X,1))
    return log_likelihood_H(Σ, X)
end


"""
Maximum likelihood estimation of Hurst exponent and volatility using fractional Gaussian noise model.

# Args
- X: observation vector or matrix of a fGn process. For matrix input each column is an i.i.d. observation.
- d: time-lag of the finite difference operator used for computing `X`.
- method, ε: see `fWn_MLE_estim()`.

# Notes
- This method may be expensive for data of large dimensions, see docs in `MLE_prepare_data()`.
"""
function fGn_MLE_estim(X::AbstractVecOrMat{T}, d::Int; method::Symbol=:optim, ε::Real=1e-2) where {T<:Real}
    @assert 0. < ε < 1.
    func = h -> -fGn_log_likelihood_H(X, h, d)

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

    # Σ = Matrix(Symmetric(covmat(FractionalGaussianNoise(hurst, 1.), size(X,1))))
    Σ = Matrix(Symmetric(fGn_covmat(size(X,1), hurst, d)))
    σ = sqrt(xiAx(Σ, X) / length(X))
    L = log_likelihood_H(Σ, X)

    return (hurst, σ), L, opm
end


"""
Multiscale fGn-MLE
"""
function ms_fGn_MLE_estim(X::AbstractVector{T}, lags::AbstractVector{Int}, w::Int) where {T<:Real}
    Hs = zeros(length(lags))
    Σs = zeros(length(lags))

    for (n,lag) in enumerate(lags)  # time lag for finite difference
        # vectorization with window size w
        dXo = rolling_vectorize(X[lag+1:end]-X[1:end-lag], w, 1)
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

