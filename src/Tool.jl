########## Utility functions ##########

#### Rolling window ####

"""
Apply a function on a rolling window with hard truncation at boundaries.

# Args
- func: function to be applied, taking matrix as input and returning a vector or a scalar
- X0: input data, vector or matrix. For matrix the rolling window runs through the row direction.
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
Remove singular dimensions
"""
function squeezedims(A::AbstractArray{T}; dims::Union{Int,AbstractVector{Int}}) where {T<:Real}
    dima = intersect(tuple(findall(size(A).==1)...), dims)
    return length(dima)>0 ? dropdims(A, dims=dima) : A
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


function pinv_iter(A::AbstractMatrix{T}, method::Symbol=:lsqr) where {T<:Number}
    iA = zeros(Float64, size(A'))
    try
        iA = pinv(A)
    catch
        for c = 1:size(iA, 2)
            b = zeros(Float64, size(A,1))
            b[c] = 1.
            iA[:,c] = IterativeSolvers.lsqr(A, b)
        end
    end
    return iA
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
    LevinsonDurbin(cseq::Vector{Float64})

Diagonalization of a symmetric positive definite Toeplitz matrix using Levinson-Durbin (LD) method by providing `cseq`,
the covariance sequence of a stationary process.

# Returns
- `pseq::Vector{Vector{Float64}}`: linear prediction coefficients
- `sseq`: variances of residual
- `rseq`: partial correlation coefficients

# Explanation
`pseq` forms the lower triangular matrix diagonalizing the covariance matrix Γ, and `sseq` forms the resulting diagonal matrix. `rseq[n]` is just `pseq[n][n]`.
"""
function LevinsonDurbin(cseq::AbstractVector{T}) where {T<:Real}
    N = length(cseq)

    if N > 1
        # check that cseq is a validate covariance sequence
        @assert cseq[1] > 0
        @assert all(abs.(cseq[2:end]) .<= cseq[1])
        # @assert all(diff(abs.(cseq)) .<= 0)

        # initialization
        # pseq: linear prediction coefficients
        pseq = Vector{Vector{Float64}}(N-1); pseq[1] = [cseq[2]/cseq[1]]
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

LevinsonDurbin(p::StationaryProcess{T}, g::RegularGrid{<:T}) where T = LevinsonDurbin(covseq(p, g))


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

