########## Utility functions ##########

#### Rolling window ####

"""
Apply a function on a rolling window with hard truncation at boundaries.

# Args
- func: function to be applied, taking matrix as input and returning a vector or a scalar
- X0: input data, vector or matrix. For matrix the rolling window runs through the row direction (i.e. horizontally, each column corresponds to a time)
- s: size of rolling window
- d: step of rolling window
- mode: :causal or :anticausal

# Returns
- a matrix or a row vector
"""
function rolling_apply_hard(func::Function, X0::AbstractVecOrMat{T}, s::Int, d::Int=1; mode::Symbol=:causal) where {T<:Number}
    # @assert s>0
    X = ndims(X0)>1 ? X0 : reshape(X0, 1, :)  # vec to matrix, create a reference not a copy
    L = size(X,2)
    return if mode==:causal
        hcat(reverse([func(X[:,t-s+1:t]) for t=L:-d:s])...)
    else  # anti-causal
        hcat([func(X[:,t:t+s-1]) for t=1:d:L-s+1]...)
    end
end

"""
Apply a function on a rolling window with soft truncation at boundaries.
"""
function rolling_apply_soft(func::Function, X0::AbstractVecOrMat{T}, s::Int, d::Int=1; mode::Symbol=:causal) where {T<:Number}
    # @assert s>0
    X = ndims(X0)>1 ? X0 : reshape(X0, 1, :)  # vec to matrix, create a reference not a copy
    L = size(X,2)
    return if mode==:causal
        hcat(reverse([func(X[:,max(1,t-s+1):t]) for t=L:-d:1])...)
    else  # anti-causal
        hcat([func(X[:,t:min(L,t+s-1)]) for t=1:d:L]...)
    end
end


"""
    rolling_vectorize(X0::AbstractVecOrMat{T}, w::Int, d::Int=1) where {T<:Real}

Rolling vectorization.

# Args
- X0: real vector or matrix
- w: size of rolling window
- d: step of rolling window
"""
function rolling_vectorize(X0::AbstractVecOrMat{T}, w::Int, d::Int=1) where {T<:Number}
    return rolling_apply_hard(x->vec(x), X0, w, d)
end


##### Useful functions #####

function shrinkage_by_value(X0::AbstractArray{T}, v::T, mode::Symbol=:soft) where {T<:Number}
    X1 = fill(zero(T), size(X0))  # or zero(X0)
    idx = findall(abs.(X0) .> v)
    if mode == :soft
        X1[idx] .= sign.(X0[idx]) .* (abs.(X0[idx]) .- v)
    else
        X1[idx] .= X0[idx]
    end
    return X1
end

function shrinkage_by_number(X0::AbstractArray{T}, n::Int, mode::Symbol=:soft) where {T<:Number}
    Xv = ndims(X0)>1 ? vec(X0) : X0
    X1 = fill(zero(T), length(Xv))  # or zero(X0)
    idx = sortperm(abs.(Xv))[end-n+1:end]  # increasing order
    if mode == :soft
        X1[idx] .= sign.(Xv[idx]) .* (abs.(Xv[idx]) .- Xv[idx[1]])
    else
        X1[idx] .= Xv[idx]
    end
    return reshape(X1, size(X0))
end

function shrinkage_by_percentage(X0::AbstractArray{T}, p::Real, mode::Symbol=:soft) where {T<:Number}
    @assert 0 <= p <= 1
    return shrinkage_by_number(X0, floor(Int,p*length(X0)), mode)
end


"""
Sigmoid function.
"""
sigmoid(α::Real) = exp(α)/(1+exp(α))

"""
Derivative of sigmoid function.
"""
diff_sigmoid(α::Real) = exp(α)/(1+2*exp(α)+exp(2α))


"""
Remove singular dimensions of an array.

# Note
This function is safe for array of size (1,1,..1).
"""
function squeezedims(X::AbstractArray{T}; dims::Union{Int,AbstractVector{Int}}) where {T<:Real}
    dimx = tuple(intersect(tuple(findall(size(X).==1)...), dims)...)
    return if length(dimx) == 0
        X
    elseif length(dimx) == ndims(X)
        vec(X)
    else
        dropdims(X, dims=dimx)
    end
end

# function squeezedims(A::AbstractArray{T}) where {T<:Real}
#     dims = tuple(findall(size(A).==1)...)
#     return length(dims)>0 ? dropdims(A, dims=dims) : A
# end


ifloor(x::Int, y::Int) = floor(Int, x/y) * y
# ifloor(x::Int, y::Int) = x - (x%y)
iceil(x::Int, y::Int) = ceil(Int, x/y) * y


"""
    vec2mat(x::AbstractVector{T}, r::Int) where {T<:Real}

Reshape a vector `x` to a matrix of `r` rows with truncation if `length(x)` does not divide `r`.
"""
function vec2mat(x::AbstractVector{T}, r::Int) where {T<:Real}
    @assert r<=length(x)  # number of row must be smaller than the size of x
    n = length(x) - (length(x)%r)
    return reshape(x[1:n], r, :)
end

##### Algebra #####

function norm(X::AbstractMatrix{T}, p::Real=2; dims::Int=1) where {T<:Number}
    if dims==1
        return [LinearAlgebra.norm(X[:,n],p) for n=1:size(X,2)]
    else
        return [LinearAlgebra.norm(X[n,:],p) for n=1:size(X,1)]
    end
end


"""
Compute A^-1 * B using the lsqr iterative method.
"""
function lsqr(A::AbstractMatrix{T}, B::AbstractMatrix{T}; kwargs...) where {T<:Real}
    # println("My lsqr")
    X = zeros(Float64, (size(A,2), size(B,2)))
    for n=1:size(B,2)
        X[:,n] = lsqr(A, B[:,n])
    end
    return X
end


"""
Diagonalization of a symmetric positive definite Toeplitz matrix using Levinson-Durbin (LD) method by providing `cseq`,
the covariance sequence of a stationary process.

# Returns
- `pseq::Vector{Vector{Float64}}`: linear prediction coefficients
- `sseq`: variances of residual
- `rseq`: partial correlation coefficients

# Explanation
`pseq` forms the lower triangular matrix diagonalizing the covariance matrix Γ, and `sseq` forms the resulting diagonal matrix. `rseq[n]` is just `pseq[n][n]`.
"""
function LevinsonDurbin(cseq::AbstractVector{<:Real})
    N = length(cseq)

    if N > 1
        # check that cseq is a validate covariance sequence
        @assert cseq[1] > 0
        @assert all(abs.(cseq[2:end]) .<= cseq[1])
        # @assert all(diff(abs.(cseq)) .<= 0)

        # initialization
        # pseq: linear prediction coefficients
        pseq = Vector{Vector{Float64}}(undef, N-1); pseq[1] = [cseq[2]/cseq[1]]
        # sseq: variances of residual
        sseq = zeros(N); sseq[1] = cseq[1]; sseq[2] = (1-pseq[1][1]^2) * sseq[1]
        # rseq: partial correlation coefficients
        rseq = zeros(N-1); rseq[1] = pseq[1][1]

        # recursive construction of the prediction coefficients and variances
        for n=2:N-1
            pseq[n] = zeros(n)
            pseq[n][n] = (cseq[n+1] - cseq[2:n]' * pseq[n-1][end:-1:1]) / sseq[n]
            pseq[n][1:n-1] = pseq[n-1] - pseq[n][n] * pseq[n-1][end:-1:1]
            sseq[n+1] = (1 - pseq[n][n]^2) * sseq[n]
            rseq[n] = pseq[n][n]
        end
    else
        pseq = Vector{Float64}[]
        sseq = copy(cseq)
        rseq = Float64[]
    end
    return pseq, sseq, rseq
end

LevinsonDurbin(p::StationaryProcess{T}, g::AbstractVector{<:T}) where T<:TimeStyle = LevinsonDurbin(covseq(p, g))


"""
Cholesky decomposition based on SVD.
"""
function chol_svd(W::AbstractMatrix{T}) where {T<:Real}
    Um, Sm, Vm = svd((W+W')/2)  # svd of forced symmetric matrix
    Ss = sqrt.(Sm[Sm.>0])  # truncation of negative singular values
    return Um*diagm(Ss)
end


"""
Vandermonde matrix.
"""
function vandermonde(dim::Tuple{Int,Int})
    nrow, ncol = dim
    V = zeros(Float64, dim)
    for c = 1:dim[2]
        V[:,c] = collect((1:dim[1]).^(c-1))
    end
    return V
end

vandermonde(nrow::Int, ncol::Int) = vandermonde((nrow, ncol))

function col_normalize(A::AbstractMatrix{T}, p::Real=2) where {T<:Real}
    return A / diagm([norm(A[:,n], p) for n=1:size(A,2)])
end

function col_normalize!(A::AbstractMatrix{T}, p::Real=2) where {T<:Real}
    for n=1:size(A,2)
        A[:,n] ./= norm(A[:,n], p)
    end
    return A
end

row_normalize(A) = col_normalize(transpose(A))
row_normalize!(A) = col_normalize!(transpose(A))


"""
Compute d-lag finite difference of a vector or matrix (in row direction).
"""
function lagdiff(X::AbstractVecOrMat{<:Number}, d::Int, mode::Symbol=:causal)
    dX = fill(NaN, size(X))
    if ndims(X) == 1
        dX[d+1:end] = X[d+1:end]-X[1:end-d]
    else
        dX[d+1:end,:] = X[d+1:end,:]-X[1:end-d,:]
    end
    return (mode==:causal) ? dX : circshift(dX, -d)
end


###### Time series manipulation ######

function ffill!(X::AbstractVector{T}) where T<:Number
    for n=2:length(X)
        if ismissing(X[n]) || isnan(X[n])
            X[n] = X[n-1]
        end
    end
    return X
end

ffill(X) = ffill!(copy(X))

function bfill!(X::AbstractVector{T}) where T<:Number
    for n=length(X)-1:-1:1
        if ismissing(X[n]) || isnan(X[n])
            X[n] = X[n+1]
        end
    end
    return X
end
bfill(X) = bfill!(copy(X))


"""
Split an object of `TimeArray` by applying truncation.

# Args
- data: input object of `TimeArray`
- T: unit of time period, e.g. `Dates.Day(1)`
- (wa,wb): relative starting and ending time w.r.t. to `T`, e.g., `Dates.Hour(9) + Dates.Minute(5)` for `09:05` and `Dates.Hour(17) + Dates.Minute(24)` for `17:24`.
- fillmode: fill mode for nan values, :f forward, :b backward, :fb forward-backward, :bf backward-forward
- endpoint: if true include the endpoint whenever possible

# Notes
"""
function window_split_timearray(data::TimeArray, T::AbstractTime, (wa,wb)::NTuple{2, Union{Nothing, AbstractTime}}=(nothing, nothing); fillmode::Symbol=:fb, endpoint::Bool=true)
    stamp = TimeSeries.timestamp(data)  # time stamp
    time_begin, toto = Dates.floorceil(stamp[1], T)
    toto, time_end = Dates.floorceil(stamp[end], T)
    unit = minimum(diff(stamp)) # ÷ Dates.Millisecond(1000)
    # unit = Dates.Second(dt)  # time unit of data

    res = []
    for t in time_begin:T:time_end
        ta = (wa == nothing) ? t : t+wa
        tb = (wb == nothing) ? t+T : t+wb

        y = TimeSeries.to(TimeSeries.from(data,ta),tb)
        if length(y) > 0
            # x0 = (wa == nothing && wb == nothing) ? y : window_timearray(y, ta:unit:tb, fillmode)
            x0 = _window_timearray(y, ta:unit:tb, fillmode)
            x = (TimeSeries.timestamp(x0)[end] == tb && !endpoint) ? x0[1:end-1] : x0
            if length(x) > 0
                push!(res, x)
            end
        end
    end
    return res
end


function _window_timearray(A::TimeArray, tstp::AbstractVector{<:Dates.AbstractTime}, fillmode::Symbol)
    if fillmode == :o
        return A
    else
        cnames = TimeSeries.colnames(A)
        vstp = intersect(tstp, TimeSeries.timestamp(A))  # valid timestamps
        sidx = map(t->findall(isequal(t), tstp)[1], vstp)  # relative index of valid timestamp
        # shape = (ndims(TimeSeries.values(A)) == 1) ? (length(tstp),) : (length(tstp),length(cnames))

        x = fill(NaN, (length(tstp),length(cnames)))
        x[sidx,:] = TimeSeries.values(A[vstp])

        xf = if fillmode == :n
            x
        elseif fillmode==:f
            xf = hcat([ffill(x[:,n]) for n=1:size(x,2)]...)
        elseif fillmode==:b
            xf = hcat([bfill(x[:,n]) for n=1:size(x,2)]...)
        elseif fillmode==:fb
            xf = hcat([bfill(ffill(x[:,n])) for n=1:size(x,2)]...)
        elseif fillmode==:bf
            xf = hcat([ffill(bfill(x[:,n])) for n=1:size(x,2)]...)
        else
            error("Unknown symbol: $(fill)")
        end

        return TimeArray(tstp, (ndims(TimeSeries.values(A))==1) ? vec(xf) : xf, cnames)
    end
end


"""
Equalize the first point of the day with the last point of the previous day.

# Notes
- Useful in processing of stock price
"""
function equalize_daynight(sdata::AbstractVector)
    @assert typeof(sdata[1]) <: TimeArray
    N = ndims(sdata[1])
    @assert N<=2
    edata = [sdata[1]]
    func = (x,y) -> (N==2) ? x[:,1]-y[:,end] : x[1]-y[end]

    for n=2:length(sdata)
        d0 = TimeSeries.values(edata[n-1])
        d1 = TimeSeries.values(sdata[n])
        t1 = TimeSeries.timestamp(sdata[n])
        # push!(edata, TimeSeries.TimeArray(t1, d1.-(d1[1]-d0[end]), TimeSeries.colnames(sdata[1])))  # <- TODO: d0[end] or d1[1] is NaN? d1 is 2d array?
        push!(edata, TimeSeries.TimeArray(t1, d1.-func(d1,d0), TimeSeries.colnames(sdata[1])))  # <- TODO: d0[end] or d1[1] is NaN? d1 is 2d array?
    end
    return edata
end

