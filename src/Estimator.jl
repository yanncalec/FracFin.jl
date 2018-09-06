# Estimators for fractional processes.

"""
Estimate the Hurst exponent and the volatility using power law method.

# Notations
For a continuous-time process X, let Δ_δ(X)(t) = X(t+δ) - X(t) be the increment process of step δ.
The observation we have is a discrete-time process (time series) with the sampling step h.

# Detail of the method
The expectation of the p-th moment of a fBm is
    E(|Δ_δ (X)|^p) = c_p * σ^p * δ^(pH),
with c_p being a constant depending only on p.

## Step 1: Estimation of Hurst exponent
For each p in `pows` we can compute the vector
    Y = [log(E(|Δ_(d*h) (X)|^p) for d in lags]
using the previous formula with δ = d*h. Note that Δ_(d*h) (X) here is just the d-lag difference of X.
Moreover, it holds
    diff(Y) = p * diff(log(lags)) * H
i.e. after derivative the term log(c_p * σ^p * h^(pH)) is removed from Y. Therefore one can use an
OLS to regress
    dY = diff(Y) against dX = p*diff(log(lags))
which gives estimation of the Hurst exponent. Actually a single OLS is applied by concatenating
all dY (and dX) of different p together.

## Step 2: Estimation of volatility
With the estimated H, we compute first the vector
    Z = Y - [p*H * log(d) - log(c_p) for d in lags]
which equals to p * log(σ * h^H), then using again an OLS to regress Z against p gives
log(σ * h^H), from which we can compute the estimation σ if h is provided. Actually a single OLS is
applied in a similar way as above.

# Args
* X: sample path
* lags: array of the increment step
* pows: array of positive power

# Returns
* Y: a matrix that n-th column is the vector [log(E(|Δ_d(X)|^pows[n])) for d in lags]
* ols: a dictionary of GLM objects, ols['h'] is the OLS for Hurst exponent and ols['σ'] is the OLS for log(σ * h^H)

# Notes
The intercepts in both OLS are supposed to be zero. To extract the estimation of H and σ, use
the function `powlaw_coeff`.
"""
function powlaw_estim(X::Vector{Float64}, lags::AbstractArray{Int}, pows::AbstractArray{T}; splstep::Real=1.) where {T<:Real}
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

    Σ = (splstep.^(-H)) .* exp.((β-log.(C))./pows)
    return mean(H), mean(Σ)
end


function powlaw_estim_old(X::Vector{Float64}, lags::AbstractArray{Int}, pows::AbstractArray{T}) where {T<:Real}
    # Define the function for computing the p-th moment of the increment
    moment_incr(X,d,p) = mean((abs.(X[d+1:end] - X[1:end-d])).^p)

    # Estimation of Hurst exponent
    Y = zeros(Float64, (length(lags), length(pows)))
    for (n,p) in enumerate(pows)
        Y[:,n] = map(d -> log(moment_incr(X, d, p)), lags)
    end
    dY = diff(Y, 1)
    df = DataFrames.DataFrame(xvar=(diff(log.(lags)) * pows')[:],
                              yvar=dY[:])
    ols_hurst = GLM.lm(@GLM.formula(yvar ~ xvar), df)
    Hurst = GLM.coef(ols_hurst)[2]  # estimation of hurst exponent

    # Estimation of volatility
    # Constant in the p-th moment of normal distribution, see
    # https://en.wikipedia.org/wiki/Normal_distribution#Moments
    cps = [2^(p/2) * gamma((p+1)/2)/sqrt(pi) for p in pows]
    Z = Y -  Hurst * (log.(lags) * pows') - ones(lags) * log.(cps)'
    dg = DataFrames.DataFrame(xvar=(ones(lags) * pows')[:],
                              yvar=Z[:])
    ols_sigma = GLM.lm(@GLM.formula(yvar ~ xvar), dg)

    return Y, Dict('H'=>ols_hurst, 'σ'=>ols_sigma)
end

"""
Extract the estimates of Hurst and voaltility from the result of `powlaw_estim`.
"""
function powlaw_coeff(ols::Dict, h::Float64)
    H = GLM.coef(ols['H'])[2]
    σ = exp(GLM.coef(ols['σ'])[2] - H * log(h))

    return H, σ
end


##### Generalized scalogram #####

# function scalogram_estim(Cxx::Matrix{Float64}, sclrng::AbstractArray{Int}, ρmax::Int=1)
#     nr, nc = size(Cxx)
#     @assert nr == nc == length(sclrng)
#     @assert ρmax >= 1
#     xvar = Float64[]
#     yvar = Float64[]
#     for ρ=1:ρmax
#         toto = [Cxx[j, ρ*j] for j in 1:nr if ρ*j<=nr]
#         xvar = vcat(xvar, sclrng[1:length(toto)])
#         yvar = vcat(yvar, abs.(toto))
#     end
#     df = DataFrames.DataFrame(
#         xvar=log2.(xvar),
#         yvar=log2.(yvar)
#     )
#     ols_hurst = GLM.lm(@GLM.formula(yvar~xvar), df)
#     hurst_estim = (GLM.coef(ols_hurst)[2]-1)/2    
#     return hurst_estim, ols_hurst
# end

"""
    scalogram_estim(C::Matrix{Float64}, sclrng::AbstractArray{Int}, v::Int, ρ::Int=1)

Scalogram estimator for Hurst exponent and volatility.
"""
function scalogram_estim(C::Matrix{Float64}, sclrng::AbstractArray{Int}, v::Int, ρ::Int=1)
    @assert size(C,1) == size(C,2) == length(sclrng)

    toto = [C[j, ρ*j] for j in 1:size(C,1) if ρ*j<=size(C,1)]
    yvar = abs.(toto)
    df = DataFrames.DataFrame(
        X=log.(sclrng[1:length(toto)]),
        Y=log.(abs.(toto))
    )
    ols = GLM.lm(@GLM.formula(Y~X), df)
    coef = GLM.coef(ols)
    hurst_estim = (coef[2]-1)/2
    C2 = C1rho(0, ρ, hurst_estim, v)
    σ_estim = ℯ^((coef[1] - log(abs(C2)) - (hurst_estim+1/2)*log(ρ))/2)
    
    return (hurst_estim, σ_estim), ols, df
end

##### MLE #####



##### Wavelet-MLE #####

function Wavelet_MLE_obj(X::Matrix{Float64}, sclrng::AbstractArray, vm::Int, H::Real, s::Real)
    N, d = size(X)  # length and dim of X
    A = [sqrt(i*j) for i in sclrng, j in sclrng].^(2H+1)
    C1 = [C1rho(0, j/i, H, vm) for i in sclrng, j in sclrng]
    Σ = s * C1 .* A
    iX = Σ \ X'
    return -1/2 * (trace(X*iX) + N*log(abs(det(Σ))) + N*d*log(2π))
end

function grad_Wavelet_MLE_obj(X::Matrix{Float64}, sclrng::AbstractArray, vm::Int, H::Real, s::Real)
    N, d = size(X)  # length and dim of X
    A = [sqrt(i*j) for i in sclrng, j in sclrng].^(2H+1)
    dAda = [log(i*j) for i in sclrng, j in sclrng] .* A

    C1 = [C1rho(0, j/i, H, vm) for i in sclrng, j in sclrng]
    dC1da = [diff_C1rho(0, j/i, H, vm) for i in sclrng, j in sclrng]

    Σ = s^2 * C1 .* A
    dΣda = s^2 * (dC1da .* A + C1 .* dAda)
    dΣdb = 2s * C1 .* A

    iX = Σ \ X'
    da = N * trace(Σ \ dΣda) - trace(iX' * dΣda * iX)
    db = N * trace(Σ \ dΣdb) - trace(iX' * dΣdb * iX)

    return  -1/2 * [da, db]
end

# MLE with constraints by change of variable
C1sgm(α::Real, ρ::Real, vm::Int) = C1rho(0, ρ, sigmoid(α), vm)
diff_C1sgm(α::Real, ρ::Real, vm::Int) = diff_C1rho(0, ρ, sigmoid(α), vm) * diff_sigmoid(α)

function Wavelet_MLE_obj_sgm(X::Matrix{Float64}, sclrng::AbstractArray, vm::Int, α::Real, β::Real)
    N, d = size(X)  # length and dim of X
    A = [sqrt(i*j) for i in sclrng, j in sclrng].^(2*sigmoid(α)+1)
    C1 = [C1sgm(α, j/i, vm) for i in sclrng, j in sclrng]
    Σ = exp(2β) * C1 .* A
    iX = Σ \ X'
#     iΣ = pinv(Σ)  # regularization by pseudo-inverse
#     iX = iΣ * X'
    return -1/2 * (trace(X*iX) + N*log(abs(det(Σ))) + N*d*log(2π))
end

function grad_Wavelet_MLE_obj_sgm(X::Matrix{Float64}, sclrng::AbstractArray, vm::Int, α::Real, β::Real)
    N, d = size(X)  # length and dim of X
    A = [sqrt(i*j) for i in sclrng, j in sclrng].^(2*sigmoid(α)+1)
    dAda = diff_sigmoid(α) * [log(i*j) for i in sclrng, j in sclrng] .* A

#     C1 = zeros(length(sclrng), length(sclrng))
#     dC1da = zeros(C1)
#     for (r,i) in enumerate(sclrng)
#         for (c,j) in enumerate(sclrng)
#             C1sgm_ij = a -> C1sgm(a, j/i, vm)
#             diff_C1sgm_ij = x -> ForwardDiff.derivative(C1sgm_ij, x)
#             C1[r,c] = C1sgm_ij(α)
#             dC1da[r,c] = diff_C1sgm_ij(α)
#         end
#     end

    C1 = [C1sgm(α, j/i, vm) for i in sclrng, j in sclrng]
    dC1da = [diff_C1sgm(α, j/i, vm) for i in sclrng, j in sclrng]

    Σ = exp(2β) * C1 .* A
#     Σ += eye(Σ) * 1e-8
    dΣda = exp(2β) * (dC1da .* A + C1 .* dAda)
    dΣdb = 2*Σ

#     iΣ = pinv(Σ)  # regularization by pseudo-inverse
#     iX = iΣ * X'
#     da = N * trace(iΣ * dΣda) - trace(iX' * dΣda * iX)
#     db = N * trace(iΣ * dΣdb) - trace(iX' * dΣdb * iX)

    iX = Σ \ X'
    da = N * trace(Σ \ dΣda) - trace(iX' * dΣda * iX)
    db = N * trace(Σ \ dΣdb) - trace(iX' * dΣdb * iX)

    return  -1/2 * [da, db]
end
