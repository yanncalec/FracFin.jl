########## Strategies for automatic trading ##########

function signedgain(N::AbstractVector{<:Real}, P::AbstractVector{<:Real}, C::Real)
	#     @assert all.(P.>0)
	#     @assert Cv >= 0
	#     @assert length(N) == length(P)
	#     @assert all.(N.>0)
	Q = diff(N)
	return N[end] * P[end] - sum(Q.*P[1:end-1]) - sum(abs.(Q).*P[1:end-1]) * C
end

"""
# Args
- A: ask or short position
- B: bid or long position
- P: price
- C: commission rate

# Return
Time series of signed gain
"""
function gain(A::AbstractVector{<:Real}, B::AbstractVector{<:Real}, P::AbstractVector{<:Real}, C::Real)
    @assert all(A.>=0) && all(B.>=0) && all(P.>0)
    @assert length(A) == length(B) == length(P)
#     @assert 1 >= C >= 0
    
    return cumsum((1-C) * (A.*P) - (1+C) * (B.*P)) + cumsum(B-A) .* P
end


"""
# Args
- R: return 
- C: commission rate
- position: :long, :short or :both
"""
function make_orders(R::AbstractVector{<:Real}, C::Real; position::Symbol=:both)
#     @assert C>0
    A, B = zeros(length(R)+1), zeros(length(R)+1)
    A0, B0 = R.<-C, R.>C
    
    if position in (:long, :both)
        B[1:end-1] += B0
        A[2:end] += B0
    end
    
    if position in (:short, :both)
        A[1:end-1] += A0
        B[2:end] += A0
    end
    
    return A,B
end

