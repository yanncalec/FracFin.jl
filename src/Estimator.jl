# Estimators for fractional processes.

"""doc
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

