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
function powlaw_estim(X::Vector{Float64}, lags::AbstractArray{Int}, pows::AbstractArray{T}=[2.]) where {T<:Real}
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
        Ap = hcat(xp, ones(xp))  # design matrix
        H[n], β[n] = Ap \ yp  # estimation of H and β

        # dg = DataFrames.DataFrame(xvar=Ap, yvar=yp)
        # ols = GLM.lm(@GLM.formula(yvar ~ xvar), dg)
        # β[n], H[n] = GLM.coef(ols)
    end

    Σ = exp.((β-log.(C))./pows)
    return mean(H), mean(Σ)
end


##### Generalized scalogram #####

"""
    scalogram_estim(C::Matrix{Float64}, sclrng::AbstractArray{Int}, v::Int, ρ::Int=1)

Scalogram estimator for Hurst exponent and volatility.

# Args
- X: matrix of wavelet coefficients. Each column corresponds to one scale.
- sclrng: scale of wavelet transform. Each number in `sclrng` corresponds to one column in the matrix X
- v: vanishing moments
- ρ: integer ratio defining an oblique diagonal (i, ρi), e.g. ρ=1 corresponds to the main diagonal.
"""
function scalogram_estim(X::Matrix{Float64}, sclrng::AbstractArray{Int}, v::Int, ρ::Int=1; mode::Symbol=:center)
    @assert size(X,2) == length(sclrng)

    Σ = cov(X, X, dims=1, corrected=true)  # covariance matrix

    yvar = [Σ[j, ρ*j] for j in 1:size(Σ,1) if ρ*j<=size(Σ,1)]
    df = DataFrames.DataFrame(
        X=log.(sclrng[1:length(yvar)]),
        Y=log.(abs.(yvar))
    )
    ols = GLM.lm(@GLM.formula(Y~X), df)
    coef = GLM.coef(ols)

    hurst_estim = (coef[2]-1)/2
    C2 = C1rho(0, ρ, hurst_estim, v, mode)
    σ_estim = ℯ^((coef[1] - log(abs(C2)) - (hurst_estim+1/2)*log(ρ))/2)
    
    return (hurst_estim, σ_estim), ols
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



##### Wavelet-MLE #####

"""
    wavelet_MLE_obj(X::Matrix{Float64}, sclrng::AbstractArray{Int}, v::Int, α::Real, β::Real; cflag::Bool=true, mode::Symbol=:center)

# Args
- X: each row is an observation
"""

## MLE with constraints by change of variable ##

function wavelet_MLE_obj(X::Matrix{Float64}, sclrng::AbstractArray{Int}, v::Int, α::Real, β::Real; cflag::Bool=true, mode::Symbol=:center)
    N, d = size(X)  # length and dim of X

    H = cflag ? sigmoid(α) : α
    σ = cflag ? exp(β) : β
    
    A = [sqrt(i*j) for i in sclrng, j in sclrng].^(2H+1)
    C1 = [C1rho(0, j/i, H, v, mode) for i in sclrng, j in sclrng]

    Σ = σ^2 * C1 .* A
    Σ += Matrix(1.0I, size(Σ)) * max(1e-8, mean(abs.(Σ))*1e-5)

    println("H=$(H), σ=$(σ), α=$(α), β=$(β), mean(Σ)=$(mean(abs.(Σ)))")
    println("logdet(Σ)=$(logdet(Σ))")
    
    # method 1:
    # iX = Σ \ X'

    # method 2:
    iΣ = pinv(Σ)  # regularization by pseudo-inverse
    iX = iΣ * X'  # regularization by pseudo-inverse

    # method 3:
    # iX = lsqr(Σ, X')
    
    return -1/2 * (tr(X*iX) + N*logdet(Σ) + N*d*log(2π))
end


function grad_wavelet_MLE_obj(X::Matrix{Float64}, sclrng::AbstractArray{Int}, v::Int, α::Real, β::Real; cflag::Bool=true, mode::Symbol=:center)
    N, d = size(X)  # length and dim of X

    H = cflag ? sigmoid(α) : α
    σ = cflag ? exp(β) : β

    A = [sqrt(i*j) for i in sclrng, j in sclrng].^(2H+1)
    C1 = [C1rho(0, j/i, H, v, mode) for i in sclrng, j in sclrng]
    dAda = [log(i*j) for i in sclrng, j in sclrng] .* A
    dC1da = [diff_C1rho(0, j/i, H, v, mode) for i in sclrng, j in sclrng]
    
    if cflag        
        dAda *= diff_sigmoid(α)
        dC1da *= diff_sigmoid(α)
    end

    Σ = σ^2 * C1 .* A
    # Σ += Matrix(1.0I, size(Σ)) * max(1e-8, mean(abs.(Σ))*1e-5)
    dΣda = σ^2 * (dC1da .* A + C1 .* dAda)
    dΣdb = cflag ? 2*Σ : 2σ * C1 .* A

    # method 1:
    # iX = Σ \ X'
    # da = N * tr(Σ \ dΣda) - tr(iX' * dΣda * iX)
    # db = N * tr(Σ \ dΣdb) - tr(iX' * dΣdb * iX)

    # method 2:
    # iΣ = pinv(Σ)  # regularization by pseudo-inverse
    # iX = iΣ * X'
    # da = N * tr(iΣ * dΣda) - tr(iX' * dΣda * iX)
    # db = N * tr(iΣ * dΣdb) - tr(iX' * dΣdb * iX)

    # method 3:
    iX = lsqr(Σ, X')
    da = N * tr(lsqr(Σ, dΣda)) - tr(iX' * dΣda * iX)
    db = N * tr(lsqr(Σ, dΣdb)) - tr(iX' * dΣdb * iX)

    return  -1/2 * [da, db]
end


function wavelet_MLE_estim(X::Matrix{Float64}, sclrng::AbstractArray{Int}, v::Int, d::Int=1, vars::Symbol=:all, init::Vector{Float64}=[0.,0.]; cflag::Bool=true, dflag::Bool=false, mode::Symbol=:center)
    @assert size(X,2) == length(sclrng)
    @assert length(init) == 2
    @assert d >= 1

    func = x -> ()
    grad = x -> ()
    hurst_estim, σ_estim = init

    if vars == :all
        func = x -> -wavelet_MLE_obj(X[1:d:end, :], sclrng, v, x[1], x[2]; cflag=cflag, mode=mode)
        grad = x -> -grad_wavelet_MLE_obj(X[1:d:end,:], sclrng, v, x[1], x[2]; cflag=cflag, mode=mode)
    elseif vars == :hurst
        func = x -> -wavelet_MLE_obj(X[1:d:end, :], sclrng, v, x[1], σ_estim; cflag=cflag, mode=mode)
        grad = x -> -grad_wavelet_MLE_obj(X[1:d:end,:], sclrng, v, x[1], σ_estim; cflag=cflag, mode=mode)
    else
        func = x -> -wavelet_MLE_obj(X[1:d:end, :], sclrng, v, hurst_estim, x[2]; cflag=cflag, mode=mode)
        grad = x -> -grad_wavelet_MLE_obj(X[1:d:end,:], sclrng, v, hurst_estim, x[2]; cflag=cflag, mode=mode)
    end

    # opm = dflag ? Optim.optimize(func, grad, init, Optim.BFGS(); inplace=false) : Optim.optimize(func, init, Optim.BFGS()) #, Optim.Options(f_tol=1e-10, iterations=100))
    
    opm = Optim.optimize(func, [1e-3, 0], [1-1e-3, Inf], init, Optim.Fminbox(Optim.GradientDescent()))
    res = Optim.minimizer(opm)
    
    if vars == :all
        hurst_estim = sigmoid(res[1])
        σ_estim = exp(res[2])
    elseif vars == :hurst
        hurst_estim = sigmoid(res[1])
    else
        σ_estim = exp(res[2])
    end
    
    return (hurst_estim, σ_estim), opm
end
    