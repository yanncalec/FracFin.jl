########## General statistics ##########

"""
Auto-Correlation function by RCall.
"""
function acf(X::AbstractVector{T}, lagmax::Int) where {T<:Real}
    res = RCall.rcopy(RCall.rcall(:acf, X, lagmax, plot=false, na_action=:na_pass))
    return res[:acf][2:end]
end

"""
Partial Auto-Correlation function by RCall.
"""
function pacf(X::AbstractVector{T}, lagmax::Int) where {T<:Real}
    res = RCall.rcopy(RCall.rcall(:pacf, X, lagmax, plot=false, na_action=:na_pass))
    return res[:acf][:]
end


"""
Auto-Correlation function of increment process.
"""
function acf_incr(X::AbstractVector{T}, dlags::Union{Int, AbstractVector{Int}}, lagmax::Int; method::Symbol=:acf) where {T<:Real}
    # for single value of dlag: convert to a list
    if typeof(dlags) <: Integer
        dlags = [dlags]
    end
    
    A = if method==:acf
        [acf((X[l+1:end]-X[1:end-l])[100:end-100], lagmax) for l in dlags]
    elseif method==:pacf
        [pacf((X[l+1:end]-X[1:end-l])[100:end-100], lagmax) for l in dlags]            
    else
        error("Invalid method: $(method)")
    end

    # A = []  # output of acf
    # for l in dlags
    #     dX = X[l+1:end]-X[1:end-l]
    #     if method==:acf
    #         push!(A, acf(dX, lagmax))
    #         # push!(A, [cor(dX[t+1:end], dX[1:end-t]) for t in tidx])
    #     elseif method==:pacf
    #         push!(A, pacf(dX, lagmax))
    #     else
    #         error("Invalid method: $(method)")
    #     end
    # end
    return hcat(A...)
end


"""
    cov(X::AbstractVecOrMat, Y::AbstractVecOrMat, w::StatsBase.AbstractWeights)

Reweighted covariance between `X` and `Y`, where each row is an observation.
"""
function cov(X::AbstractVecOrMat, Y::AbstractVecOrMat, w::StatsBase.AbstractWeights)
    # w is always a column vector, by definition of AbstractWeights
    @assert size(X, 1) == size(Y, 1) == length(w)
    # weighted mean
    mX = mean(X, w, 1)
    mY = mean(Y, w, 1)
    return (X .- mX)' * (w .* (Y .- mY)) / sum(w)
end


"""
Robust estimator of variance
"""
function robustvar(X::AbstractArray{T}; dims::Int=1) where {T<:Number}
    lqt = -0.393798799600891  # log of 3/4 quantil of standard normal distribution
    Q = log.((X .- mean(X, dims=dims)).^2)
    return exp.(median(Q, dims=dims) .- 2lqt)
end


"""Principal Component Analysis.

# Args
-X (2d array): each row represents a variable and each column represents an observation
-nc (int): number of components to hold

# Returns
- C, U : coefficients and corresponded principal directions
"""
function pca(X::AbstractMatrix, nc::Int, w::StatsBase.AbstractWeights, center=false)
    U, S, V = svd(cov(X, X, w))
    mX = mean(X, w, 1)
    C = center ? (X .- mX) * U' : X * U'

    return C[:,1:nc], U[:,1:nc], S[1:nc]
end

pca(X::AbstractMatrix, nc::Int, center=false) = pca(X, nc, StatsBase.weights(ones(size(X,1))), center)


######## Linear models ########
"""
Multi-linear regression in the column direction. 
"""
function multi_linear_regression_colwise(Y::AbstractVecOrMat{T}, X::AbstractVecOrMat{T}, w::StatsBase.AbstractWeights) where {T<:Real}
    @assert size(Y,1)==size(X,1)==length(w)
    # println("size(Y)=",size(Y))
    # println("size(X)=",size(X))
    μy = mean(Y, w, 1)[:]  # Julia function (version <= 0.7) `mean` does not take keyword argument `dims=1` if weight is passed.
    μx = mean(X, w, 1)[:]
    Σyx = cov(Y, X, w)   # this calls user defined cov function
    Σxx = cov(X, X, w)

    # Σxx += 1e-5 * Matrix{Float64}(I,size(Σxx))  # perturbation
    # A = Σyx / Σxx  # scalar or matrix, i.e., Σyx * inv(Σxx)

    A = Σyx * pinv(Σxx)  # scalar or matrix, i.e., Σyx * inv(Σxx)

    # A = try
    #     Σyx * pinv(Σxx)  # scalar or matrix, i.e., Σyx * inv(Σxx)
    # catch
    #     zero(Σyx)
    # end

    β = μy - A * μx  # scalar or vector
    E = Y - (A * X' .+ β)'
    Σ = cov(E, E, w)
    return (A, β), E, Σ
end

"""
    linear_regression(Y::AbstractMatrix, X::AbstractMatrix, w::StatsBase.AbstractWeights; dims::Int=1)

Reweighted (multi-)linear regression of data matrix `Y` versus `X` in the given dimension (dimension of observation).

# Returns
- (A, β): explanatory slope (scalar or matrix) and intercept (scalar or vector)
- E: residual
- Σ: covariance of the residual
"""
function linear_regression(Y::AbstractMatrix, X::AbstractMatrix, w::StatsBase.AbstractWeights; dims::Int=1)    
    if dims==1
        return multi_linear_regression_colwise(Y, X, w)
    else
        return multi_linear_regression_colwise(Y', X', w)
    end
end

linear_regression(Y::AbstractMatrix, X::AbstractMatrix; dims::Int=1) =
    linear_regression(Y, X, StatsBase.weights(ones(size(Y,1))); dims=dims)

function linear_regression(Y::AbstractVector, X::AbstractVector, w::StatsBase.AbstractWeights)
    return multi_linear_regression_colwise(Y, X, w)
end

linear_regression(Y::AbstractVector, X::AbstractVector) = linear_regression(Y, X, StatsBase.weights(ones(size(Y,1))))


######## Nonlinear models ########
"""
IRLS
"""
function IRLS(Y::AbstractVecOrMat{T}, X::AbstractVecOrMat{T}, pnorm::Real=2.; maxiter::Int=10^3, tol::Float64=10^-3, vreg::Float64=1e-8) where {T<:Real}
    @assert pnorm > 0

    wfunc = E -> (sqrt.(sum(E.*E, dims=2) .+ vreg).^(pnorm-2))[:]  # function for computing weight vector
    
    (A, β), E, Σ = linear_regression(Y, X)  # initialization
    w0::Vector{Float64} =  wfunc(E) # weight vector
    n::Int = 1
    err::Float64 = 0.

    for n=1:maxiter
        (A, β), E, Σ = linear_regression(Y, X, StatsBase.weights(w0))
        w = wfunc(E)
        err = norm(w - w0) / norm(w0)
        w0 = w
        if  err < tol
            break
        end
    end
    # println(n)
    # println(err)

    return (A, β), w0, E, sum(sqrt.(sum(E.*E, dims=2)).^pnorm)
end


######## Rolling estimators ########
"""
    rolling_estim(estim::Function, X0::AbstractVecOrMat{T}, (w,s,d)::Tuple{Int,Int,Int}, p::Int, trans::Function=(x->vec(x)); mode::Symbol=:causal) where {T<:Real}

Rolling window estimator for 1d or multivariate time series.

# Args
- estim: estimator which accepts either 1d or 2d array
- X0: input, 1d or 2d array with each column being one observation
- (w,s,d): size of rolling window; size of sub-window, length of decorrelation
- p: step of the rolling window
- trans: function of transformation which returns a vector of fixed size or a scalar,  optional
- mode: :causal or :anticausal

# Returns
- array of estimations on the rolling window

# Notes
- The estimator is applied on a rolling window every `p` steps. The rolling window is divided into `n` (possibly overlapping) sub-windows of size `w` at the pace `d`, such that the size of the rolling window equals to `(n-1)*d+w`. For q-variates time series, data on a sub-window is a matrix of dimension `q`-by-`w` which is further transformed by the function `trans` into another vector. The transformed vectors of `n` sub-windows are concatenated into a matrix which is finally passed to the estimator `estim`. As example, for `trans = x -> vec(x)` the data on a rolling window is put into a new matrix of dimension `w*q`-by-`n`, and its columns are the column-concatenantions of data on the sub-window. Moreover, different columns of this new matrix are assumed as i.i.d. observations.
- Boundary is treated in a soft way.
"""
function rolling_estim(estim::Function, X0::AbstractVecOrMat{T}, (w,s,d)::Tuple{Int,Int,Int}, p::Int, trans::Function=(x->vec(x)); mode::Symbol=:causal, nan::Symbol=:ignore) where {T<:Number}    
    X = ndims(X0)>1 ? X0 : reshape(X0, 1, :)  # vec to matrix, create a reference not a copy
    L = size(X,2)
    # println(L); println(w); println(s)
    # @assert w >= s && L >= s
    res = []

    if mode == :causal  # causal
        for t = L:-p:1
            # xv = view(X, :, t:-1:max(1, t-w+1))
            xv = view(X, :, max(1, t-w+1):t)
            xs = if nan == :ignore
                idx = findall(.!any(isnan.(xv), dims=1)[:])  # ignore columns containing nan values
                if length(idx) > 0
                    rolling_apply_hard(trans, view(xv,:,idx), s, d; mode=:causal)
                else
                    []
                end
            else
                rolling_apply_hard(trans, xv, s, d; mode=:causal)
            end

            if length(xs) > 0
                # println(size(xs))
                pushfirst!(res, (t,estim(squeezedims(xs, dims=[1,2]))))
            end
        end
    else  # anticausal
        for t = 1:p:L
            xv = view(X, :, t:min(L, t+w-1))
            xs = if nan == :ignore
                idx = findall(.!any(isnan.(xv), dims=1)[:])  # ignore columns containing nan values
                if length(idx) > 0
                    rolling_apply_hard(trans, view(xv,:,idx), s, d; mode=:anticausal)
                else
                    []
                end
            else
                rolling_apply_hard(trans, xv, s, d; mode=:anticausal)
            end
            
            if length(xs) > 0
                push!(res, (t,estim(squeezedims(xs, dims=[1,2]))))
            end
        end
    end
    return res
end


function rolling_estim(func::Function, X0::AbstractVecOrMat{T}, w::Int, p::Int, trans::Function=(x->x); kwargs...) where {T<:Number}
    return rolling_estim(func, X0, (w,w,1), p, trans; kwargs...)
end


function rolling_regress_predict(regressor::Function, predictor::Function, X0::AbstractVecOrMat{T}, Y0::AbstractVecOrMat{T}, X1::AbstractVecOrMat{T}, Y1::AbstractVecOrMat{T}, (w,s,d)::Tuple{Int,Int,Int}, p::Int, trans::Function=(x->vec(x)); mode::Symbol=:causal) where {T<:Number}
    X = ndims(X0)>1 ? X0 : reshape(X0, 1, :)  # vec to matrix, create a reference not a copy
    Y = ndims(Y0)>1 ? Y0 : reshape(Y0, 1, :)        
    @assert size(X,2) == size(Y,2)    

    @assert size(X0) == size(X1)  
    @assert size(Y0) == size(Y1)  
    Xp = ndims(X1)>1 ? X1 : reshape(X1, 1, :)  # vec to matrix, create a reference not a copy
    Yp = ndims(Y1)>1 ? Y1 : reshape(Y1, 1, :)        

    (any(isnan.(X)) || any(isnan.(Y))) && throw(ValueError("Inputs cannot contain nan values!"))
    (any(isnan.(Xp)) || any(isnan.(Yp))) && throw(ValueError("Inputs cannot contain nan values!"))
    L = size(X,2)  # total time
    @assert L >= w >= s
    
    res = []

    if mode == :causal  # causal
        tf, xf, yf = [], [], []  # future time and data used for prediction

        for t = L:-p:w
            # printfmtln("Processing time {}...\r", t)

            xv = rolling_apply_hard(trans, view(X, :, t-w+1:t), s, d; mode=:causal)
            yv0 = view(Y, :, t-w+1:t)[:, end:-d:1]  # time-reversed
            yv = yv0[:,1:size(xv,2)][:,end:-1:1]  # reverse back
    
            reg = regressor(yv, xv)  # regression

            xf = trans(Xp[:,t-s+1:t])
            yf = Yp[:,t]
            prd = predictor(reg, xf, yf)  # prediction using future data
            
            pushfirst!(res, (time=t, regression=reg, prediction=prd))
        end
    else  # anticausal: TODO
        for t = 1:p:L
            xv = rolling_apply_hard(trans, view(X, :, t:min(L, t+w-1)), s, d; mode=:anticausal)
            yv = rolling_apply_hard(trans, view(Y, :, t:min(L, t+w-1)), 1, d; mode=:anticausal)

            reg = regressor(yv, xv)                
            prd = isempty(res_reg) ? [] : predictor(res_reg[end], xv, yv[:,end])
            # TODO
        end
    end

    return res
end


"""
Rolling mean in row direction.
"""
function rolling_mean(X0::AbstractVecOrMat{T}, w::Int, d::Int=1; boundary::Symbol=:hard) where {T<:Number}
    Xm = if boundary == :hard
        rolling_apply_hard(x->mean(x, dims=2), X0, w, d)
    else
        rolling_apply_soft(x->mean(x, dims=2), X0, w, d)
    end
    return ndims(X0)>1 ? Xm : vec(Xm)
end



function rolling_estim_predict(estim::Function, predict::Function, X0::AbstractVecOrMat{T}, (w,s,d)::Tuple{Int,Int,Int}, p::Int, trans::Function=(x->vec(x)); mode::Symbol=:causal, nan::Symbol=:ignore) where {T<:Number}    
    X = ndims(X0)>1 ? X0 : reshape(X0, 1, :)  # vec to matrix, create a reference not a copy
    L = size(X,2)
    # println(L); println(w); println(s)
    # @assert w >= s && L >= s
    res = []

    if mode == :causal  # causal
        for t = L:-p:1
            # xv = view(X, :, t:-1:max(1, t-w+1))
            xv = view(X, :, max(1, t-w+1):t)
            xs = if nan == :ignore
                idx = findall(.!any(isnan.(xv), dims=1)[:])  # ignore columns containing nan values
                if length(idx) > 0
                    rolling_apply_hard(trans, view(xv,:,idx), s, d; mode=:causal)
                else
                    []
                end
            else
                rolling_apply_hard(trans, xv, s, d; mode=:causal)
            end

            if length(xs) > 0
                # println(size(xs))
                Θ = estim(squeezedims(xs, dims=[1,2]))  # estimations
                Π = predict(Θ, xv)
                pushfirst!(res, (t,Π))
            end
        end
    else  # anticausal
        for t = 1:p:L
            xv = view(X, :, t:min(L, t+w-1))
            xs = if nan == :ignore
                idx = findall(.!any(isnan.(xv), dims=1)[:])  # ignore columns containing nan values
                if length(idx) > 0
                    rolling_apply_hard(trans, view(xv,:,idx), s, d; mode=:anticausal)
                else
                    []
                end
            else
                rolling_apply_hard(trans, xv, s, d; mode=:anticausal)
            end
            
            if length(xs) > 0
                Θ = estim(squeezedims(xs, dims=[1,2]))  # estimations
                Π = predict(Θ, xv)
                push!(res, (t,Π))
            end
        end
    end
    return res
end


function linear_predictor(R::Tuple,X0::AbstractVecOrMat,Y0::AbstractVecOrMat=[])
    A,β = R[1]
    Yp = A * X0 .+ β
#     println(size(A))
#     println(size(β))
#     println(size(Yp))
#     println(size(Y0))
    Ep = isempty(Y0) ? [] : Y0-Yp  # error of prediction
    return Yp, Y0, Ep
end

