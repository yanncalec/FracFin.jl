# Estimators for fractional processes.

"""
Power-law estimator for Hurst exponent and volatility.

# Args
- X: sample path
- lags: array of the increment step
- pows: array of positive power

# Returns
- Y: a matrix that n-th column is the vector [log(E(|Δ_d(X)|^pows[n])) for d in lags]
- ols: a dictionary of GLM objects, ols['h'] is the OLS for Hurst exponent and ols['σ'] is the OLS for log(σ * h^H)

# Notes
The intercepts in both OLS are supposed to be zero. To extract the estimation of H and σ, use
the function `powlaw_coeff`.
"""
function powlaw_estim(X::Vector{Float64}, lags::AbstractArray{Int}, pows::AbstractArray{T}) where {T<:Real}
    @assert length(lags) > 1 && all(lags .> 1)

    # Define the function for computing the p-th moment of the increment
    moment_incr(X,d,p) = mean((abs.(X[d+1:end] - X[1:end-d])).^p)

    # Estimation of Hurst exponent and β
    H = zeros(Float64, length(pows))
    β = zeros(Float64, length(pows))
    C = zeros(Float64, length(pows))

    for (n,p) in enumerate(pows)
        C[n] = 2^(p/2) * gamma((p+1)/2)/sqrt(pi)

        yp = map(d -> log(moment_incr(X, d, p)), lags)
        xp = p * log.(lags)
        Ap = hcat(xp, ones(length(xp)))  # design matrix
        H[n], β[n] = Ap \ yp  # estimation of H and β

        # dg = DataFrames.DataFrame(xvar=Ap, yvar=yp)
        # ols = GLM.lm(@GLM.formula(yvar ~ xvar), dg)
        # β[n], H[n] = GLM.coef(ols)
    end

    Σ = exp.((β-log.(C))./pows)

    hurst = sum(H) / length(H)
    σ = sum(Σ) / length(Σ)

    return hurst, σ
end

function powlaw_estim(X::Vector{Float64}, lags::AbstractArray{Int}, p::T=2.) where {T<:Real}
    return powlaw_estim(X, lags, [p])
end

##### Generalized scalogram #####

"""
Scalogram estimator for Hurst exponent and volatility.

# Args
- X: matrix of wavelet coefficients. Each rwo corresponds to one scale.
- sclrng: scale of wavelet transform. Each number in `sclrng` corresponds to one row in the matrix X
- v: vanishing moments
- r: rational ratio defining a line in the covariance matrix, e.g. r=1 corresponds to the main diagonal.
"""
function scalogram_estim(X::AbstractMatrix{T}, sclrng::AbstractArray{Int}, v::Int, r::Rational=1//1; mode::Symbol=:center) where {T<:Real}
    @assert size(X,1) == length(sclrng)
    @assert r >= 1
    if r > 1
        all(diff(sclrng/sclrng[1]) .== 1) || error("Imcompatible scales: the ratio between the k-th and the 1st scale must be k")
    end

    p,q,N = r.num, r.den, length(sclrng)
    @assert N>=2p

    Σ = cov(X, X, dims=2, corrected=true)  # covariance matrix

    yr = [log(abs(Σ[q*j, p*j])) for j in 1:N if p*j<=N]
    xr = [log(sclrng[q*j] * sclrng[p*j]) for j in 1:N if p*j<=N]

    df = DataFrames.DataFrame(X=xr, Y=yr)
    ols = GLM.lm(@GLM.formula(Y~X), df)
    coef = GLM.coef(ols)

    hurst = coef[2]-1/2
    # println(hurst)
    C1 = C1rho(0, r, hurst, v, mode)
    σ = ℯ^((coef[1] - log(abs(C1)))/2)
    return (hurst, σ), ols

    # Ar = hcat(xr, ones(length(xr)))  # design matrix
    # H0, η = Ar \ yr  # estimation of H and β
    # hurst = H0-1/2
    # C1 = C1rho(0, r, hurst, v, mode)
    # σ = ℯ^((η - log(abs(C1)))/2)
    # return hurst, σ
end


# function rolling_estim(fun::Function, X::AbstractVector{T}, wsize::Int, p::Int=1) where T
#     offset = wsize-1
#     res = [fun(view(X, (n*p):(n*p+wsize-1))) for n=1:p:N]
#     end

#     # y = fun(view(X, idx:idx+offset))  # test return type of fun
#     res = Vector{typeof(y)}(undef, div(length(data)-offset, p))
#     @inbounds for n=1:p:N
#         push!(res, fun(view(X, (idx*p):(idx*p+offset))))
#         end
#     @inbounds for n in eachindex(res)
#         res[n] = fun(hcat(X[idx*p:idx*p+offset]...))
#     end

#     return res
# end


##### MLE #####

"""
Safe evaluation of the inverse quadratic form
    trace(X' * inv(A) * X)
where the matrix A is symmetric and positive definite.
"""
function xiAx(A::AbstractMatrix{T}, X::AbstractVecOrMat{T}, ε::Real=0) where {T<:Real}
    @assert issymmetric(A)
    @assert size(X, 1) == size(A, 1)

    S, U = eigen(A)  # so that U * Diagonal(S) * inv(U) == A, in particular, U' == inv(U)
    idx = (S .> ε)

    # U, S, V = svd(A)
    # idx = S .> ε
    return sum((U[:,idx]'*X).^2 ./ S[idx])
end

# function xiAx(A::AbstractMatrix{T}, X::AbstractVecOrMat{T}, ε::Real=0) where {T<:Real}
#     @assert issymmetric(A)
#     @assert size(X, 1) == size(A, 1)

#     iA = pinv(A)
#     return tr(X' * iA * X)
# end

"""
Safe evaluation of the log-likelihood of a fBm model with the implicit optimal volatility (in the MLE sense).

The value of log-likelihood (up to some additif constant) is
    -1/2 * (N*log(X'*inv(A)*X) + logdet(A))

# Args
- A: covariance matrix, must be symmetric and positive definite
- X: vector of matrix of observation

# Notes
This function is common to all MLE problems with the covariance matrix of form σ²A(θ), with unknown σ and θ. Such kind of MLE can be done in θ uniquely.
"""
function log_likelihood_H(A::AbstractMatrix{T}, X::AbstractVecOrMat{T}, ε::Real=0) where {T<:Real}
    @assert issymmetric(A)
    @assert size(X, 1) == size(A, 1)

    N = ndims(X)>1 ? size(X,2) : 1
    # d = size(X,1), such that N*d == length(X)

    S, U = eigen(A)  # so that U * Diagonal(S) * inv(U) == A, in particular, U' == inv(U)
    idx = (S .> ε)
    # U, S, V = svd(A)
    # idx = S .> ε
    return -1/2 * (length(X)*log(sum((U[:,idx]'*X).^2 ./ S[idx])) + N*sum(log.(S[idx])))
end

# function log_likelihood_H(A::AbstractMatrix{T}, X::AbstractVecOrMat{T}, ε::Real=0) where {T<:Real}
#     @assert issymmetric(A)
#     @assert size(X, 1) == size(A, 1)

#     N = ndims(X)>1 ? size(X,2) : 1
#     # d = size(X,1), such that N*d == length(X)

#     return -1/2 * (length(X)*log(xiAx(A,X)) + N*logdet(A))
# end


function fGn_log_likelihood_H(X::AbstractVecOrMat{T}, H::Real) where {T<:Real}
    @assert 0 < H < 1
    Σ = Matrix(Symmetric(covmat(FractionalGaussianNoise(H, 1.), size(X,1))))
    return log_likelihood_H(Σ, X)
end


function fGn_MLE_estim(X::AbstractVecOrMat{T}; init::Real=0.5) where {T<:Real}
    func = h -> -fGn_log_likelihood_H(X, h)

    ε = 1e-5
    # optimizer = Optim.GradientDescent()  # e.g. Optim.BFGS(), Optim.GradientDescent()
    # optimizer = Optim.BFGS()
    # opm = Optim.optimize(func, ε, 1-ε, [0.5], Optim.Fminbox(optimizer))
    opm = Optim.optimize(func, ε, 1-ε, Optim.Brent())

    hurst = Optim.minimizer(opm)[1]

    Σ = Matrix(Symmetric(covmat(FractionalGaussianNoise(hurst, 1.), size(X,1))))
    σ = sqrt(xiAx(Σ, X) / length(X))

    return (hurst, σ), opm
end


##### Wavelet-MLE #####

"""
Compute the full covariance matrix of DCWT coefficients of a pure fBm with B-Spline wavelet.

The full covariance matrix of `J`-scale transform and of time-lag `N` is a N*J-by-N*J symmetric matrix.

# Args
- l: maximum time-lag
- sclrng: scale range
- v: vanishing moments of B-Spline wavelet
- H: Hurst exponent
- mode: mode of convolution
"""
function full_bspline_covmat(l::Int, sclrng::AbstractArray{Int}, v::Int, H::Real, mode::Symbol)
    J = length(sclrng)
    A = [sqrt(i*j) for i in sclrng, j in sclrng].^(2H+1)
    Σs = [[C1rho(d/sqrt(i*j), j/i, H, v, mode) for i in sclrng, j in sclrng] .* A for d = 0:l]

    Σ = zeros(((l+1)*J, (l+1)*J))
    for r = 0:l
        for c = 0:l
            Σ[(r*J+1):(r*J+J), (c*J+1):(c*J+J)] = (c>=r) ? Σs[c-r+1] : transpose(Σs[r-c+1])
        end
    end

    return Matrix(Symmetric(Σ))  #  forcing symmetry
    # return [(c>=r) ? Σs[c-r+1] : Σs[r-c+1]' for r=0:N-1, c=0:N-1]
end


"""
Evaluate the log-likelihood of DCWT coefficients of B-Spline wavelet with full covariance matrix.
"""
function full_bspline_log_likelihood_H(X::AbstractVecOrMat{T}, sclrng::AbstractArray{Int}, v::Int, H::Real, mode::Symbol) where {T<:Real}
    @assert 0 < H < 1
    @assert size(X,1) % length(sclrng) == 0

    L = size(X,1) ÷ length(sclrng)  # interger division: \div
    N = ndims(X)>1 ? size(X,2) : 1

    Σ = full_bspline_covmat(L-1, sclrng, v, H, mode)  # full covariance matrix

    # # strangely, the following does not work (logarithm of a negative value)
    # iΣ = pinv(Σ)  # regularization by pseudo-inverse
    # return -1/2 * (J*N*log(trace(X'*iΣ*X)) + logdet(Σ))

    return log_likelihood_H(Σ, X)
end


"""
B-Spline wavelet-MLE estimator with full covariance matrix.
"""
function full_bspline_MLE_estim(X::AbstractVecOrMat{T}, sclrng::AbstractArray{Int}, v::Int; init::Real=0.5, ε::Real=1e-3, mode::Symbol=:center) where {T<:Real}
    @assert size(X,1) % length(sclrng) == 0

    L = size(X,1) ÷ length(sclrng)  # interger division: \div
    N = ndims(X)>1 ? size(X,2) : 1

    func = x -> -full_bspline_log_likelihood_H(X, sclrng, v, x[1], mode)

    # # Gradient based
    # # optimizer = Optim.GradientDescent()  # e.g. Optim.BFGS(), Optim.GradientDescent()
    # optimizer = Optim.BFGS()
    # opm = Optim.optimize(func, [ε], [1-ε], [0.5], Optim.Fminbox(optimizer))

    # Non-gradient based
    optimizer = Optim.Brent()
    # optimizer = Optim.GoldenSection()
    opm = Optim.optimize(func, ε, 1-ε, optimizer)

    hurst = Optim.minimizer(opm)[1]

    Σ = full_bspline_covmat(L-1, sclrng, v, hurst, mode)
    σ = sqrt(xiAx(Σ, X) / length(X))

    return (hurst, σ), opm
end


# function partial_bspline_covmat(sclrng::AbstractArray{Int}, v::Int, H::Real, mode::Symbol)
#     return full_bspline_covmat(0, sclrng, v, H, mode)
# end


# function partial_bspline_log_likelihood_H(X::AbstractVecOrMat{T}, sclrng::AbstractArray{Int}, v::Int, H::Real; mode::Symbol=:center) where {T<:Real}
#     # @assert size(X,1) == length(sclrng)
#     Σ = partial_bspline_covmat(sclrng, v, H, mode)
#     # println(size(Σ))
#     # println(size(X))

#     return log_likelihood_H(Σ, X)
# end


# """
# B-Spline wavelet-MLE estimator with partial covariance matrix.
# """
# function partial_bspline_MLE_estim(X::AbstractVecOrMat{T}, sclrng::AbstractArray{Int}, v::Int; init::Real=0.5, mode::Symbol=:center) where {T<:Real}
#     @assert size(X,1) == length(sclrng)

#     func = h -> -partial_bspline_log_likelihood_H(X, sclrng, v, h; mode=mode)

#     ε = 1e-5
#     # optimizer = Optim.GradientDescent()  # e.g. Optim.BFGS(), Optim.GradientDescent()
#     # optimizer = Optim.BFGS()
#     # opm = Optim.optimize(func, ε, 1-ε, [0.5], Optim.Fminbox(optimizer))
#     opm = Optim.optimize(func, ε, 1-ε, Optim.Brent())

#     hurst = Optim.minimizer(opm)[1]

#     Σ = partial_bspline_covmat(sclrng, v, hurst, mode)
#     σ = sqrt(xiAx(Σ, X) / length(X))

#     return (hurst, σ), opm
# end






function partial_wavelet_log_likelihood_H(X::Matrix{Float64}, sclrng::AbstractArray{Int}, v::Int, H::Real; mode::Symbol=:center)
    N, J = size(X)  # length and dim of X

    A = [sqrt(i*j) for i in sclrng, j in sclrng].^(2H+1)
    Σ = [C1rho(0, j/i, H, v, mode) for i in sclrng, j in sclrng] .* A

    iΣ = pinv(Σ)  # regularization by pseudo-inverse

    return -1/2 * (J*N*log(sum(X' .* (iΣ * X'))) + N*logdet(Σ))
end


function partial_wavelet_MLE_obj(X::Matrix{Float64}, sclrng::AbstractArray{Int}, v::Int, H::Real, σ::Real; mode::Symbol=:center)
    N, d = size(X)  # length and dim of X

    A = [sqrt(i*j) for i in sclrng, j in sclrng].^(2H+1)
    C1 = [C1rho(0, j/i, H, v, mode) for i in sclrng, j in sclrng]

    Σ = σ^2 * C1 .* A
    # Σ += Matrix(1.0I, size(Σ)) * max(1e-10, mean(abs.(Σ))*1e-5)

    # println("H=$(H), σ=$(σ), mean(Σ)=$(mean(abs.(Σ)))")
    # println("logdet(Σ)=$(logdet(Σ))")

    # method 1:
    # iX = Σ \ X'

    # method 2:
    iΣ = pinv(Σ)  # regularization by pseudo-inverse
    iX = iΣ * X'  # regularization by pseudo-inverse

    # # method 3:
    # iX = lsqr(Σ, X')

    return -1/2 * (tr(X*iX) + N*logdet(Σ) + N*d*log(2π))
end


# function wavelet_MLE_obj(X::Matrix{Float64}, sclrng::AbstractArray{Int}, v::Int, H::Real, σ::Real; mode::Symbol=:center)
#     N, d = size(X)  # length and dim of X

#     A = [sqrt(i*j) for i in sclrng, j in sclrng].^(2H+1)
#     C1 = [C1rho(0, j/i, H, v, mode) for i in sclrng, j in sclrng]

#     Σ = σ^2 * C1 .* A
#     # Σ += Matrix(1.0I, size(Σ)) * max(1e-10, mean(abs.(Σ))*1e-5)

#     # println("H=$(H), σ=$(σ), mean(Σ)=$(mean(abs.(Σ)))")
#     # println("logdet(Σ)=$(logdet(Σ))")

#     # method 1:
#     # iX = Σ \ X'

#     # method 2:
#     iΣ = pinv(Σ)  # regularization by pseudo-inverse
#     iX = iΣ * X'  # regularization by pseudo-inverse

#     # # method 3:
#     # iX = lsqr(Σ, X')

#     return -1/2 * (tr(X*iX) + N*logdet(Σ) + N*d*log(2π))
# end


# older version:
# function wavelet_MLE_obj(X::Matrix{Float64}, sclrng::AbstractArray{Int}, v::Int, α::Real, β::Real; cflag::Bool=false, mode::Symbol=:center)
#     N, d = size(X)  # length and dim of X

#     H = cflag ? sigmoid(α) : α
#     σ = cflag ? exp(β) : β

#     A = [sqrt(i*j) for i in sclrng, j in sclrng].^(2H+1)
#     C1 = [C1rho(0, j/i, H, v, mode) for i in sclrng, j in sclrng]

#     Σ = σ^2 * C1 .* A
#     # Σ += Matrix(1.0I, size(Σ)) * max(1e-8, mean(abs.(Σ))*1e-5)

#     # println("H=$(H), σ=$(σ), α=$(α), β=$(β), mean(Σ)=$(mean(abs.(Σ)))")
#     # println("logdet(Σ)=$(logdet(Σ))")

#     # method 1:
#     # iX = Σ \ X'

#     # # method 2:
#     # iΣ = pinv(Σ)  # regularization by pseudo-inverse
#     # iX = iΣ * X'  # regularization by pseudo-inverse

#     # method 3:
#     iX = lsqr(Σ, X')

#     return -1/2 * (tr(X*iX) + N*log(abs(det(Σ))) + N*d*log(2π))
# end


# function grad_wavelet_MLE_obj(X::Matrix{Float64}, sclrng::AbstractArray{Int}, v::Int, α::Real, β::Real; cflag::Bool=false, mode::Symbol=:center)
#     N, d = size(X)  # length and dim of X

#     H = cflag ? sigmoid(α) : α
#     σ = cflag ? exp(β) : β

#     A = [sqrt(i*j) for i in sclrng, j in sclrng].^(2H+1)
#     C1 = [C1rho(0, j/i, H, v, mode) for i in sclrng, j in sclrng]
#     dAda = [log(i*j) for i in sclrng, j in sclrng] .* A
#     dC1da = [diff_C1rho(0, j/i, H, v, mode) for i in sclrng, j in sclrng]

#     if cflag
#         dAda *= diff_sigmoid(α)
#         dC1da *= diff_sigmoid(α)
#     end

#     Σ = σ^2 * C1 .* A
#     # Σ += Matrix(1.0I, size(Σ)) * max(1e-8, mean(abs.(Σ))*1e-5)
#     dΣda = σ^2 * (dC1da .* A + C1 .* dAda)
#     dΣdb = cflag ? 2*Σ : 2σ * C1 .* A

#     # method 1:
#     # iX = Σ \ X'
#     # da = N * tr(Σ \ dΣda) - tr(iX' * dΣda * iX)
#     # db = N * tr(Σ \ dΣdb) - tr(iX' * dΣdb * iX)

#     # method 2:
#     iΣ = pinv(Σ)  # regularization by pseudo-inverse
#     iX = iΣ * X'
#     da = N * tr(iΣ * dΣda) - tr(iX' * dΣda * iX)
#     db = N * tr(iΣ * dΣdb) - tr(iX' * dΣdb * iX)

#     # method 3:
#     # iX = lsqr(Σ, X')
#     # da = N * tr(lsqr(Σ, dΣda)) - tr(iX' * dΣda * iX)
#     # db = N * tr(lsqr(Σ, dΣdb)) - tr(iX' * dΣdb * iX)

#     return  -1/2 * [da, db]
# end


function wavelet_MLE_estim(X::Matrix{Float64}, sclrng::AbstractArray{Int}, v::Int; vars::Symbol=:all, init::Vector{Float64}=[0.5,1.], mode::Symbol=:center)
    @assert size(X,2) == length(sclrng)
    @assert length(init) == 2

    func = x -> ()
    hurst, σ = init
    # println(init)

    if vars == :all
        func = x -> -wavelet_MLE_obj(X, sclrng, v, x[1], x[2]; mode=mode)
    elseif vars == :hurst
        func = x -> -wavelet_MLE_obj(X, sclrng, v, x[1], σ; mode=mode)
    else
        func = x -> -wavelet_MLE_obj(X, sclrng, v, hurst, x[2]; mode=mode)
    end

    ε = 1e-8
    # optimizer = Optim.GradientDescent()  # e.g. Optim.BFGS(), Optim.GradientDescent()
    optimizer = Optim.BFGS()
    opm = Optim.optimize(func, [ε, ε], [1-ε, 1/ε], init, Optim.Fminbox(optimizer))
    # opm = Optim.optimize(func, [ε, ε], [1-ε, 1/ε], init, Optim.Fminbox(optimizer); autodiff=:forward)
    res = Optim.minimizer(opm)

    if vars == :all
        hurst, σ = res[1], res[2]
    elseif vars == :hurst
        hurst = res[1]
    else
        σ = res[2]
    end

    return (hurst, σ), opm
end

