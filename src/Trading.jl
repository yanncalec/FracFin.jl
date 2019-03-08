########## Strategies for automatic trading ##########

"""
Compute the return from raw price.
"""
function price2return(P::AbstractVector{<:Real}, lag::Integer; mode::Symbol=:causal,method::Symbol=:ori)
    # @assert all(P.>0)
    return if method==:ori
        if mode==:causal
            lagdiff(P, lag; mode=mode) ./ circshift(P, lag)
        else
            lagdiff(P, lag; mode=mode) ./ P
        end
    elseif method==:log
        lagdiff(log.(P), lag; mode=mode)
    else
        error("Unknown method: $(method)")
    end
end

function price2return(P::AbstractVector{<:Real}, lags::AbstractVector{<:Integer}; kwargs...)
    hcat([price2return(P, d; kwargs...) for d in lags]...)
end


function signed_gain(N::AbstractVector{<:Real}, P::AbstractVector{<:Real}, C::Real)
	#     @assert all.(P.>0)
	#     @assert Cv >= 0
	#     @assert length(N) == length(P)
	#     @assert all.(N.>0)
	Q = diff(N)
	return N[end] * P[end] - sum(Q.*P[1:end-1]) - sum(abs.(Q).*P[1:end-1]) * C
end


"""
    portfolio_value(A::AbstractVector{<:Real}, B::AbstractVector{<:Real}, P::AbstractVector{<:Real}, C::Real)

Compute the gain of a series of orders (value of a portfolio) given the price and the commission rate.

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
    @assert length(A) == length(B) # <= length(P)
    # @assert 1 >= C >= 0
    L = min(length(A), length(P))

    Ask = view(A, 1:L)
    Bid = view(B, 1:L)
    Price = view(P, 1:L)

    # The portfolio value is the sum of 1) the cash value and 2) the spot value of asset.
    # 1) The cash value at time t is the cumulation of historical gain in cash.
    # 2) The spot value of asset at time t is the quantity of asset in possesion times the spot price.
    return cumsum((1-C) * (Ask.*Price) - (1+C) * (Bid.*Price)) + cumsum(Bid-Ask) .* Price
end


"""
Make unitary rolling orders from a time series of return value.

The unitary rolling order consists in holding a unitary asset for only one time unit:
- if buy now then must sell out at next
- if short now then must buy back at next

# Args
- R: (future, or anti-causal) return, a time series (of fixed step)
- d: time unit
- C: commission rate
- position: :long, :short or :both

# Returns
A, B: the series of ask and bid orders, which is `d` elements longer than the input `R`.
"""
function make_unitary_rolling_orders(R::AbstractVector{<:Real}, d::Integer, C::Real; position::Symbol=:both)
    # @assert C>0
    # orders for Ask (short/sell) and Bid (long/buy):
    A, B = zeros(length(R)+d), zeros(length(R)+d)
    A0, B0 = R.<-C, R.>C  # NaN-safe

    if position in (:long, :both)
        # make long position rolling order: buy then sell
        B[1:end-d] += B0
        A[d+1:end] += B0
    end

    if position in (:short, :both)
        # make short position rolling order: short then buy
        A[1:end-d] += A0
        B[d+1:end] += A0
    end

    return A,B
end
