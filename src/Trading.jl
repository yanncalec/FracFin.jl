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
    portfolio_value(A::AbstractVector{<:Real}, B::AbstractVector{<:Real}, P::AbstractVector{<:Real}, C::Real)

Compute the gain of a series of orders given the price and the commission rate.

# Args
- A: orders of ask (short/sell)
- B: orders of bid (long/buy)
- P: time series of price
- C: commission rate

# Return
Time series of cumulative signed gain
"""
function portfolio_value(A::AbstractVector{<:Real}, B::AbstractVector{<:Real}, P::AbstractVector{<:Real}, C::Real)
    @assert all(A.>=0) && all(B.>=0) && all(P.>0)
    @assert length(A) == length(B) == length(P)
#     @assert 1 >= C >= 0
    
    # The cash value at time t is the cumulation of historical gain in cash,
    # The spot value of asset at time t is the quantity of asset in possesion times the spot price.
    # The portfolio value is the sum of 1) the cash value and the spot value of asset
    return cumsum((1-C) * (A.*P) - (1+C) * (B.*P)) + cumsum(B-A) .* P
end


"""
Make unitary rolling orders from a time series of return value.

The unitary rolling order consists in holding a unitary asset for only one time unit:
- if buy at time `t` then must sell out at time `t+1`
- if short at time `t` then must buy back at time `t+1`

# Args
- R: (future) return, a time series (of fixed step)
- C: commission rate
- position: :long, :short or :both

# Returns
A, B: the series of ask and bid orders, which is one element longer than the input `R`.
"""
function make_unitary_rolling_orders(R::AbstractVector{<:Real}, C::Real; position::Symbol=:both)
#     @assert C>0
    # orders for Ask (short/sell) and Bid (long/buy): 
    A, B = zeros(length(R)+1), zeros(length(R)+1)  
    A0, B0 = R.<-C, R.>C
    
    if position in (:long, :both)
        # make long position rolling order: buy then sell
        B[1:end-1] += B0
        A[2:end] += B0
    end
    
    if position in (:short, :both)
        # make short position rolling order: short then buy
        A[1:end-1] += A0
        B[2:end] += A0
    end
    
    return A,B
end

