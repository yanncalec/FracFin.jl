import RCall  # slow

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
