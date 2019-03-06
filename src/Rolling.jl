########## Rolling window ##########

"""
Apply a function on a rolling window with truncation at boundaries.

# Args
- func: function to be applied
- X: input array. The rolling window is applied on the last dimension from small to large index (time arrow). For example, on a vector it runs through the column direction (i.e. vertically), while on a matrix it runs through the row direction (i.e. horizontally, and each column corresponds to an observation).
- w: number of samples on the rolling window
- d: downsampling factor on the rolling window
- p: step of the rolling window

# Kwargs
- mode: :causal or :anticausal
- boundary: truncation of boundary, :hard or :soft
"""
function rolling_apply(func::Function, X::AbstractArray, w::Integer, d::Integer=1, p::Integer=1; mode::Symbol=:causal, boundary::Symbol=:hard)
    return if boundary == :hard
        rolling_apply_hard(func, X, w, d, p; mode=mode)
    elseif boundary == :soft
        rolling_apply_soft(func, X, w, d, p; mode=mode)
    else
        error("Unknown boundary condition: $(boundary)")
    end
end


"""
# Notes
The result is empty if `d*(w-1)+1 > L`
"""
function rolling_apply_hard(func::Function, X::AbstractArray, w::Integer, d::Integer=1, p::Integer=1; mode::Symbol=:causal)
    @assert w>=1 && d>=1 && p>=1

    L = size(X)[end]  # number of samples in X
    res = []
    if mode==:causal
        gs = t -> reverse(t:-d:(t-d*(w-1)))  # grid of samples, ending by t
        for t=L:-p:(1+d*(w-1))
            # V = ndims(X) == 1 ? reverse(view(X,t:-d:t-d*w+1)) : reverse(view(X,:,t:-d:t-d*w+1),dims=2)  # old version, WRONG
            # correct version:
            # V = ndims(X) == 1 ? view(X,gs(t)) : view(X,:,gs(t))
            # pushfirst!(res, (t, func(V)))
            pushfirst!(res, (t, func(selectdim(X, ndims(X), gs(t)))))
        end
    else  # anti-causal
        gs = t -> t:d:(t+d*(w-1))  # grid of samples, starting by t
        for t=1:p:(L-d*(w-1))
            # V = ndims(X) == 1 ? view(X,t:d:t+d*w-1) : view(X,:,t:d:t+d*w-1)  # old version, WRONG
            # correct version
            # V = ndims(X) == 1 ? view(X,gs(t)) : view(X,:,gs(t))
            # push!(res, (t, func(V)))
            push!(res, (t, func(selectdim(X, ndims(X), gs(t)))))
        end
    end
    return res
end


function rolling_apply_soft(func::Function, X::AbstractArray, w::Integer, d::Integer=1, p::Integer=1; mode::Symbol=:causal)
    @assert w>=1 && d>=1 && p>=1

    L = size(X)[end]  # number of samples in X
    res = []
    if mode==:causal
        gs = t -> reverse(t:-d:max(1,t-d*(w-1)))  # grid of samples, ending by t
        for t=L:-p:1
            pushfirst!(res, (t, func(selectdim(X, ndims(X), gs(t)))))
        end
    else  # anti-causal
        gs = t -> t:d:min(L,t+d*(w-1))  # grid of samples, starting by t
        for t=1:p:L
            push!(res, (t, func(selectdim(X, ndims(X), gs(t)))))
        end
    end
    return res
end


"""
Rolling vectorization.

Vectorize the contents of a rolling window and make horizontal concatenation.

# Args
- X: input array. The window rolls in the last dimension from small to large index.
- w: size of rolling window, or number of consecutive samples to be concatenated together
- d: downsampling factor
- p: period
- mode: :causal (proceed from right to left) or :anticausal (proceed from left to right)

# Returns
A named tuple `(index, value)` which are
- index of `X` where the rolling window starts (mode=:anticausal) or ends (mode=:causal, default)
- output matrix that row dimension equals to `w` if parameters are valid, otherwise it is an empty matrix.
"""
function rolling_vectorize(X::AbstractArray{T}, w::Integer, d::Integer=1, p::Integer=1; kwargs...) where {T}
    # res = rolling_apply_hard(x->vec(hcat(x...)), X, w, 1, p; kwargs...)
    # res = rolling_apply_hard(x->vec(x), X, w, 1, p; kwargs...)
    res = rolling_apply_hard(x->vec(x), X, w, d, p; kwargs...)

    tidx::Vector{Integer} = [x[1] for x in res]
    # In case of empty array `hcat([]...)` gives `0-element Array{Any,1}`
    # Assure the return type in this case is a numerical, not `any`.
    val::Matrix{T} = length(tidx)==0 ? Array{T}(undef,0,0) : hcat([x[2] for x in res]...)
    return (index=tidx, value=val)
end


"""
Rolling mean in row direction.
"""
function rolling_mean(X::AbstractVecOrMat{<:Number}, w::Integer, d::Integer=1, p::Integer=1; kwargs...)
    res = rolling_apply(x->mean(x, dims=1), X, w, d, p; kwargs...)
    Xm = hcat([x[2] for x in res]...)[:]
    # tidx = [x[1] for x in res]
    return Xm
end

"""
Rolling median in row direction.
"""
function rolling_median(X::AbstractVecOrMat{<:Number}, w::Integer, d::Integer=1, p::Integer=1; kwargs...)
    res = rolling_apply(x->median(x, dims=1), X, w, d, p; kwargs...)
    Xm = hcat([x[2] for x in res]...)[:]
    # tidx = [x[1] for x in res]
    return Xm
end


######## Below is not maintained ########

"""
Make rolling orders based on fGn-MLE.

# Args
- X: raw price
- d: time lag for finite difference operator of the return
- w: number of samples on the rolling window
- d: downsampling factor on the rolling window
- p: period of the rolling window
- s: size of sub window
- l: length of decorrelation
- mode: :causal or :anticausal
- boundary: truncation, :hard or :soft

"""
function rolling_order_fGn_MLE(R::AbstractVector{<:Real}, d::Integer; wsize::Integer, dfct::Integer, pov::Integer, ssize::Integer, dlen::Integer, nspl::Integer, sdfc::Integer, nprd::Integer, dmode::Symbol=:logscale, kwargs...)
    @assert 0 < ssize <= wsize

    func = x -> fGn_MLE_estim_predict(x, d, ssize, dlen, nspl, sdfc, nprd; dmode=dmode)

    # kwargs_rolling = Dict(k=>v for (k,v) in kwargs if k in [:causal, :boundary])
    return rolling_apply(func, R, wsize, dfct, pov; kwargs...)
end


# function rolling_order_fGn_MLE(P::AbstractVector{<:Real}, d::Integer; wsize::Integer, pov::Integer, dmode::Symbol=:regular, kwargs...)
#     R = price2return(P, d, mode=:causal, method=:ori)  # return
#     # println(size(R))
#     func = x -> fGn_MLE_estim_predict(x, 1, wsize, 1, 0, 1, 1; dmode=dmode)

#     # kwargs_rolling = Dict(k=>v for (k,v) in kwargs if k in [:causal, :boundary])
#     return rolling_apply(func, R, wsize, d, pov; kwargs...)
# end



# abstract type AbstractDataStream <: TimeSeries.AbstractTimeSeries end

# function fetch_data(X::AbstractDataStream, )
# end

# abstract type AbstractRollingFunction end

# struct RollingEstimPredictor <: AbstractRollingFunction
#     prepare::Function  # data preparation
#     estim::Function  # estimator
#     predict::Function  # predictor
# end


"""
    rolling_estim(estim::Function, X0::AbstractVecOrMat{T}, (w,s,d)::Tuple{Integer,Integer,Integer}, p::Integer, trans::Function=(x->vec(x)); mode::Symbol=:causal) where {T<:Real}

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
function rolling_estim(estim::Function, X0::AbstractVecOrMat{T}, (w,s,d)::Tuple{Integer,Integer,Integer}, p::Integer, trans::Function=(x->vec(x)); mode::Symbol=:causal, nan::Symbol=:ignore) where {T<:Number}
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


function rolling_estim(func::Function, X0::AbstractVecOrMat{T}, w::Integer, p::Integer, trans::Function=(x->x); kwargs...) where {T<:Number}
    return rolling_estim(func, X0, (w,w,1), p, trans; kwargs...)
end


function rolling_regress_predict(regressor::Function, predictor::Function, X0::AbstractVecOrMat{T}, Y0::AbstractVecOrMat{T}, X1::AbstractVecOrMat{T}, Y1::AbstractVecOrMat{T}, (w,s,d)::Tuple{Integer,Integer,Integer}, p::Integer, trans::Function=(x->vec(x)); mode::Symbol=:causal) where {T<:Number}
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


function rolling_estim_predict(estim::Function, predict::Function, X0::AbstractVecOrMat{T}, (w,s,d)::Tuple{Integer,Integer,Integer}, p::Integer, trans::Function=(x->vec(x)); mode::Symbol=:causal, nan::Symbol=:ignore) where {T<:Number}
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
            xs  = hcat([x[2] for x in xs]...)  # added 2019-01
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
