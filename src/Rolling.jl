########## Rolling window ##########

"""
Apply a function on a rolling window with hard truncation at boundaries.

# Args
- func: function to be applied, taking matrix as input and returning a vector or a scalar
- X: input data, vector or matrix. For matrix the rolling window runs through the row direction (i.e. horizontally, each column corresponds to a time)
- s: size of rolling window
- d: step of rolling window
- mode: :causal or :anticausal
"""
function rolling_apply_hard(func::Function, X::AbstractVector, s::Int, d::Int=1; mode::Symbol=:causal)
    # @assert s>0
    L = length(X)
    res = []
    if mode==:causal
        for t=L:-d:s
            pushfirst!(res, (t, func(view(X,t-s+1:t))))
        end
    else  # anti-causal
        for t=1:d:L-s+1
            push!(res, (t, func(view(X,t:t+s-1))))
        end
    end
    return res
end


function rolling_apply_hard(func::Function, X::AbstractMatrix, s::Int, d::Int=1; mode::Symbol=:causal)
    # @assert s>0
    L = size(X,2)
    res = []
    if mode==:causal
        for t=L:-d:s
            pushfirst!(res, (t, func(view(X,:,t-s+1:t))))
        end
    else  # anti-causal
        for t=1:d:L-s+1
            push!(res, (t, func(view(X,:,t:t+s-1))))
        end
    end
    return res
end


"""
Apply a function on a rolling window with soft truncation at boundaries.
"""
function rolling_apply_soft(func::Function, X::AbstractVector, s::Int, d::Int=1; mode::Symbol=:causal)
    # @assert s>0
    L = length(X)
    res = []
    if mode==:causal
        for t=L:-d:1
            pushfirst!(res, (t, func(view(X,max(1,t-s+1):t))))
        end
    else
        for t=1:d:L
            push!(res, (t, func(view(X, t:min(L,t+s-1)))))
        end
    end
    return res
end


function rolling_apply(func::Function, X0::AbstractVector, s::Int, d::Int=1; mode::Symbol=:causal, boundary::Symbol=:hard)
    return if boundary == :hard
        rolling_apply_hard(func, X0, s, d; mode=mode)
    elseif boundary == :soft
        rolling_apply_soft(func, X0, s, d; mode=mode)
    else
        error("Unknown boundary condition: $(boundary)")
    end
end


# function rolling_apply_hard(func::Function, X0::AbstractVecOrMat{T}, s::Int, d::Int=1; mode::Symbol=:causal) where {T<:Number}
#     # @assert s>0
#     X = ndims(X0)>1 ? X0 : reshape(X0, 1, :)  # vec to matrix, create a reference not a copy
#     L = size(X,2)
#     res = []
#     if mode==:causal
#         for t=L:-d:s
#             pushfirst!(res, (t, func(X[:,t-s+1:t])))
#         end
#     else  # anti-causal
#         for t=1:d:L-s+1
#             push!(res, (t, func(X[:,t:t+s-1])))
#         end
#     end
#     return res
# end


# function rolling_apply_soft(func::Function, X0::AbstractVecOrMat{T}, s::Int, d::Int=1; mode::Symbol=:causal) where {T<:Number}
#     # @assert s>0
#     X = ndims(X0)>1 ? X0 : reshape(X0, 1, :)  # vec to matrix, create a reference not a copy
#     L = size(X,2)
#     res = []
#     if mode==:causal
#         for t=L:-d:1
#             pushfirst!(res, (t, func(X[:,max(1,t-s+1):t])))
#         end
#     else
#         for t=1:d:L
#             push!(res, (t, func(X[:,t:min(L,t+s-1)])))
#         end
#     end
#     return res
# end

# function rolling_apply(func::Function, X0::AbstractVecOrMat{T}, s::Int, d::Int=1; mode::Symbol=:causal, boundary::Symbol=:hard) where {T<:Number}
#     return if boundary == :hard
#         rolling_apply_hard(func, X0, s, d; mode=mode)
#     elseif boundary == :soft
#         rolling_apply_soft(func, X0, s, d; mode=mode)
#     else
#         error("Unknown boundary condition: $(boundary)")
#     end
# end


"""
    rolling_vectorize(X0::AbstractVecOrMat{T}, w::Int, d::Int=1) where {T<:Real}

Rolling vectorization.

# Args
- X0: real vector or matrix
- w: size of rolling window
- d: step of rolling window
"""
function rolling_vectorize(X0::AbstractVector{<:AbstractArray{T,N}}, w::Int, d::Int=1; kwargs...) where {T<:Number, N}
    res = rolling_apply_hard(x->vec(hcat(x...)), X0, w, d; kwargs...)
    # return [x[1] for x in res], [x[2] for x in res]
    return [x[2] for x in res]
end

# function rolling_vectorize(X0::AbstractVecOrMat{T}, w::Int, d::Int=1) where {T<:Number}
#     res = rolling_apply_hard(x->vec(x), X0, w, d)
#     return hcat(res...)
# end


######## Rolling estimators ########

abstract type AbstractRollingFunction end

struct RollingEstimPredictor <: AbstractRollingFunction
    prepare::Function  # data preparation
    estim::Function  # estimator
    predict::Function  # predictor
end


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

