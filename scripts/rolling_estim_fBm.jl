__precompile__()

using ArgParse
import ArgParse: parse_item

using Formatting
using Statistics

using Plots
# pyplot(reuse=true, size=(900,300))
pyplot()
theme(:ggplot2)

import TimeSeries
import Dates
import FracFin


function ArgParse.parse_item(::Type{AbstractVector{Int}}, S::AbstractString)
    # return ParseAbstractVector{Int}(S) #
    return ParseAbstractVector(Int, S)
end


function ArgParse.parse_item(::Type{NTuple{N, Int}}, S::AbstractString) where N
    return ParseNTuple(Int, S)
end


# function ParseAbstractVector{T}(S::AbstractString) where {T<:Real}
#     A = if something(findfirst(isequal(':'), S), 0) != 0  # step range
#         N = [parse(T, s) for s in split(S, ':')]
#         # collect(N[1]:N[2]:N[3])
#         N[1]:N[2]:N[3]
#     elseif S[1] == '[' && S[end] == ']'  # vector
#         [parse(T, s) for s in split(S[2:end-1], ',')]
#     else
#         error("Invalid input: $S")
#     end
#     # all((A .% 2) .== 0) || error("Invalid input: $S")

#     return A
# end


function ParseAbstractVector(T::Type{P}, S::AbstractString) where P<:Real
    A = if something(findfirst(isequal(':'), S), 0) != 0  # step range
        N = [parse(Int, s) for s in split(S, ':')]
        # collect(N[1]:N[2]:N[3])
        s = (length(N) == 2) ? 1 : N[2]
        StepRange(N[1], s, N[end])
    elseif S[1] == '[' && S[end] == ']'  # vector
        [parse(Int, s) for s in split(S[2:end-1], ',')]
    else
        error("Invalid input: $S")
    end
    # all((A .% 2) .== 0) || error("Invalid input: $S")

    return A
end


function ParseNTuple(T::Type{P}, S::AbstractString) where {P<:Real}
    return (parse(T, s) for s in split(S, ','))
#     A = if S[1] == '(' && S[end] == ')'  # tuple
#         (parse(T, s) for s in split(S[2:end-1], ','))
#     else
#         error("Invalid input: $S")
#     end
#     return A
end


function prepare_fGn(X0::AbstractVector{T}, dlag::Int) where {T<:Real}
    dX = fill(NaN, size(X0))  # fGn estimator is not friendly with NaN
    dX[1+dlag:end] = X0[1+dlag:end]-X0[1:end-dlag]
    return dX
end


function prepare_bspline(X0::AbstractVector{T}, sclrng::AbstractVector{Int}, vm::Int) where {T<:Real}
    W0, Mask = FracFin.cwt_bspline(X0, sclrng, vm, :left)
    t0 = findall(all(Mask, dims=2)[:])[1]
    # t1 = findall(all(Mask, dims=2)[:])[end]
    W0[1:t0-1,:] .= NaN
    return transpose(W0)
end


function parse_commandline()
    settings = ArgParseSettings("Apply rolling window estimator of fBm.")

    @add_arg_table settings begin
        "--wsize"
        help = "size of rolling window"
        arg_type = Int
        default = 180
        "--ssize"
        help = "size of sub window"
        arg_type = Int
        default = 60
        "--dlen"
        help = "length of decorrelation"
        arg_type = Int
        default = 5
        "--pov"
        help = "period of innovation"
        arg_type = Int
        default = 5
        "--ipts"
        help = "number of points per day (for intraday data only), 0 for no down-sampling"
        arg_type = Int
        default = 0
        "--idta"
        help = " starting time of truncation (for intraday data only)"
        arg_type = String
        default = "09:05"
        "--idtb"
        help = " ending time of truncation (for intraday data only)"
        arg_type = String
        default = "17:25"
        "--tfmt"
        help = "time format for parsing csv file"
        arg_type = String
        default = "yyyy-mm-dd HH:MM:SS"
        "--ncol"
        help = "index of the column to be analyzed, if the input file contains multiple columns"
        arg_type = Int
        default = 1
        "--verbose", "-v"
        help = "print message"
        action = :store_true
        "infile"
        help = "input file (with suffix '.csv' or '.txt') containing a time series"
        required = true
        "outdir"
        help = "output directory"
        required = true
        "powlaw"
        action = :command        # adds a command which will be read from an argument
        help = "power law method: 'ssize' and 'dlen' are ignored."
        "fGn-MLE"
        action = :command        # adds a command which will be read from an argument
        help = "fGn-MLE"
        "fWn-bspline-MLE"
        action = :command
        help = "fWn-bspline-MLE"
        "bspline-scalogram"
        action = :command
        help = "bspline-scalogram method: 'ssize' and 'dlen' are ignored."
    end

    @add_arg_table settings["powlaw"] begin    # add command arg_table: same as usual, but invoked on s["cmd"]
        "--pow"
        help = "compute the 'pow'-th moment "
        arg_type = Float64
        default = 1.
        "--dlags"
        help = "time lags for finite difference"
        arg_type = AbstractVector{Int}
        default = 2:10
    end

    settings["powlaw"].description = "power law description"  # this is how settings are tweaked for commands
    settings["powlaw"].commands_are_required = true # this makes the sub-commands optional
    settings["powlaw"].autofix_names = true # this uses dashes in long options, underscores in auto-generated dest_names

    @add_arg_table settings["fGn-MLE"] begin    # add command arg_table: same as usual, but invoked on s["cmd"]
        "--dlag"
        help = "time lag for finite difference"
        arg_type = Int
        default = 1
    end

    settings["fGn-MLE"].description = "fGn description"  # this is how settings are tweaked for commands
    settings["fGn-MLE"].commands_are_required = true # this makes the sub-commands optional
    settings["fGn-MLE"].autofix_names = true # this uses dashes in long options, underscores in auto-generated dest_names

    @add_arg_table settings["fWn-bspline-MLE"] begin    # add command arg_table: same as usual, but invoked on s["cmd"]
        "--vm"
        help = "vanishing moments of the wavelet"
        arg_type = Int
        default = 2
        "--sclidx"
        help = "indices of integer scales (scale must be an even number)"
        arg_type = AbstractVector{Int}
        default = 2:10
    end

    settings["fWn-bspline-MLE"].description = "fWn description"  # this is how settings are tweaked for commands
    settings["fWn-bspline-MLE"].commands_are_required = true # this makes the sub-commands optional

    @add_arg_table settings["bspline-scalogram"] begin    # add command arg_table: same as usual, but invoked on s["cmd"]
        "--vm"
        help = "vanishing moments of the wavelet"
        arg_type = Int
        default = 2
        "--sclidx"
        help = "indices of even integer scales (scale must be an even number)"
        arg_type = AbstractVector{Int}
        default = 2:10
        "--pow"
        help = "compute the 'pow'-th moment "
        arg_type = Float64
        default = 1.
    end

    settings["bspline-scalogram"].description = "fWn description"  # this is how settings are tweaked for commands
    settings["bspline-scalogram"].commands_are_required = true # this makes the sub-commands optional

    return parse_args(settings)
end


function main()
    parsed_args = parse_commandline()
    # println("Parsed args:")
    # for (arg,val) in parsed_args
    #     println("  $arg  =>  $val")
    # end

    # recover global parameters and options
    infile = parsed_args["infile"]
    outdir0 = parsed_args["outdir"]

    i1 = something(findlast(isequal('/'), infile), 0)
    i2 = something(findlast(isequal('.'), infile), length(infile)+1)
    sname = infile[i1+1:i2-1]  # filename without extension

    cmd = parsed_args["%COMMAND%"]  # name of command
    wsize = parsed_args["wsize"]  # size of rolling window
    ssize = parsed_args["ssize"]  # size of sub window
    dlen = parsed_args["dlen"]  # length of decorrelation
    pov = parsed_args["pov"]  # period of innovation
    ipts = parsed_args["ipts"]  # number of pts per day
    idta = parsed_args["idta"]
    idtb = parsed_args["idtb"]
    tfmt = parsed_args["tfmt"]  # time format
    ncol = parsed_args["ncol"]  # index of column
    verbose = parsed_args["verbose"]  # print messages

    # recover command options
    prepare::Function = X->()  # function for data preparation
    estim::Function = X->()
    trans::Function = X->vec(X)
    outstr::String = ""

    if cmd  == "powlaw"
        ssize = wsize
        dlen = 1
        pow = parsed_args[cmd]["pow"]  # time-lags of finite difference
        dlags = parsed_args[cmd]["dlags"]  # time-lags of finite difference
        outstr = format("wsize[{}]_ipts[{}]_pov[{}]_pow[{}]_dlags[{}]", wsize, ipts, pov, pow, dlags)
        prepare = X -> X
        estim = X -> FracFin.powlaw_estim(X, dlags, pow)
    elseif cmd  == "fGn-MLE"
        dlag = parsed_args[cmd]["dlag"]  # time-lag of finite difference
        outstr = format("wsize[{}]_ssize[{}]_dlen[{}]_ipts[{}]_pov[{}]_dlag[{}]", wsize, ssize, dlen, ipts, pov, dlag)
        prepare = X -> prepare_fGn(X, dlag)
        estim = X -> FracFin.fGn_MLE_estim(X, dlag; method=:optim, ε=1e-2)
    elseif cmd == "fWn-bspline-MLE"
        sclrng = 2 * parsed_args[cmd]["sclidx"]  # range of scales
        vm = parsed_args[cmd]["vm"]  # vanishing moments
        outstr = format("wsize[{}]_ssize[{}]_dlen[{}]_ipts[{}]_pov[{}]_vm[{}]_sclrng[{}]", wsize, ssize, dlen, ipts, pov, vm, sclrng)
        prepare = X -> prepare_bspline(X, sclrng, vm)
        estim = X -> FracFin.fWn_bspline_MLE_estim(X, sclrng, vm; method=:optim, ε=1e-2)
    elseif cmd == "bspline-scalogram"
        ssize = wsize
        dlen = 1
        sclrng = 2 * parsed_args[cmd]["sclidx"]  # range of scales
        pow = parsed_args[cmd]["pow"]  # time-lags of finite difference
        vm = parsed_args[cmd]["vm"]  # vanishing moments
        outstr = format("wsize[{}]_ipts[{}]_pov[{}]_pow[{}]_vm[{}]_sclrng[{}]", wsize, ipts, pov, pow, vm, sclrng)
        prepare = X -> prepare_bspline(X, sclrng, vm)
        estim = X -> FracFin.bspline_scalogram_estim(X, sclrng, vm; p=pow, mode=:left)
        # trans = X -> var(X, dims=2)
        # trans = X -> vec(FracFin.robustvar(X, dims=2))
        # trans = X -> vec(mean(abs.(X .- mean(X, dims=2)).^pow, dims=2))
        trans = X -> vec(mean(abs.(X).^pow, dims=2))
    else
        error("Unknown command")
    end

    # load data
    if verbose
        printfmtln("Loading file {}...", infile)
    end
    # data0 = CSV.read(infile)
    toto = TimeSeries.readtimearray(infile, format=tfmt, delim=',')
    cname = TimeSeries.colnames(toto)[ncol]  # name of the column
    unit = Dates.Minute(minimum(diff(TimeSeries.timestamp(toto))))
    data0 = toto[cname]

    if Int(unit/Dates.Minute(1)) < 1440  # intraday data
        tdta, tdtb = Dates.Time(idta), Dates.Time(idtb)
        dha, dhb = Dates.Hour(tdta), Dates.Hour(tdtb)
        dma, dmb = Dates.Minute(tdta), Dates.Minute(tdtb)
        daytime = (dha+dma, dhb+dmb)

        Xm, Tm, Xt, Tt = [], [], [], []
        sdata0 = []

        if ipts>0
            wh = (dhb-dha) ÷ ipts + dmb-dma # length of intraday window
            nh = Int((dhb-dha)/Dates.Hour(1)) # number of hours in intraday data
            dh = nh ÷ ipts
            windows = [(w=daytime[1]+Dates.Hour(n); (w,w+wh)) for n=0:dh:(nh-1)]

            # splitted data of identical length
            Xms = []
            Tms = []
            for (wa,wb) in windows
                sdata0 = FracFin.window_split_timearray(data0, Dates.Hour(24), (wa,wb), fillmode=:fb, endpoint=false)
                push!(Xms, log.(hcat([TimeSeries.values(v) for v in sdata0]...)))
                # println(length(sdata0))
                # println(size(Xms[end]))
                push!(Tms, hcat([TimeSeries.timestamp(v) for v in sdata0]...))
            end

            Xm = reshape(vcat(Xms...), (size(Xms[1],1), :))
            Tm = reshape(vcat(Tms...), (size(Tms[1],1), :))
            Xt = mean(Xm, dims=1)[:]
            Tt = [Dates.DateTime(t) for t in Tm[1,:]]
        else
            sdata0 = FracFin.window_split_timearray(data0, Dates.Hour(24), daytime, fillmode=:fb, endpoint=false)
            Xm = reshape(vcat(TimeSeries.values.(sdata0)...), 1, :)
            Tt = vcat(TimeSeries.timestamp.(sdata0)...)
            Xt = Xm[1, :]
        end
        # println(size(Xm))
        # println(size(Tm))
        # println(size(Xt))
        # println(size(Tt))

        Res = []
        for r = 1:size(Xm,1)
            Xi = prepare(Xm[r,:])  # input to rolling estimator
            res = FracFin.rolling_estim(estim, Xi, (wsize,ssize,dlen), pov, mode=:causal)
            push!(Res, res)
        end

        Hm = hcat([[x[2][1][1] for x in res] for res in Res]...)'
        σm = hcat([[x[2][1][2] for x in res] for res in Res]...)'
        tidx = [x[1] for x in Res[1]]

        # embedding into a longer time series
        Ht = fill(NaN, length(Tt)); Ht[tidx] = mean(Hm, dims=1)[:]
        σt = fill(NaN, length(Tt)); σt[tidx] = mean(σm, dims=1)[:]
        Hs = fill(NaN, length(Tt)); Hs[tidx] = std(Hm, dims=1)[:]
        σs = fill(NaN, length(Tt)); σs[tidx] = std(σm, dims=1)[:]

        At = TimeSeries.TimeArray(Tt, [Xt Ht Hs σt σs], ["Log_Price", "Hurst", "Hurst_std", "σ", "σ_std"])

        # make the output folder
        outdir = format("{}/{}/{}/{}/{}/", outdir0, sname, cname, cmd, outstr)
        try
            mkpath(outdir)
        catch
        end

        TimeSeries.writetimearray(At, format("{}/estimate.csv",outdir))
        # At = TimeArray(T0[tidx], [X0[tidx] Ht σt Hs σs], ["Log_Price", "Hurst", "Hurst_std" "σ" "σ_std"])
        # TimeSeries.writetimearray(At, format("{}/estimate-ts.csv",outdir))

        fig1 = plot(Tt, Xt, ylabel="Log Price", label="")
        fig2 = plot(Tt, Ht, ribbon=(2*Hs, 2*Hs), ylim=[0,1], shape=:circle, ylabel="Hurst", label="")
        hline!(fig2, [0.5], width=[3], color=[:red], label="")
        fig3 = plot(Tt, σt, ribbon=(2*σs, 2*σs), shape=:circle, ylabel="σ", label="")
        plot(fig1, fig2, fig3, layout=(3,1), size=(1200,900), legend=true)
        savefig(format("{}/estimate.pdf",outdir))

        if ipts==0
            sdata1 = FracFin.window_split_timearray(At, Dates.Hour(24); fillmode=:o, endpoint=true)
            for Y0 in sdata1
                T0 = TimeSeries.timestamp(Y0)
                fig1 = plot(Y0[:Log_Price], ylabel="Log Price", label="")
                fig2 = plot(Y0[:Hurst], shape=:circle, ylabel="Hurst", label="")
                fig3 = plot(Y0[:σ], shape=:circle, ylabel="σ", label="")
                fig4 = plot(Y0[:Hurst], ylim=[0,1], shape=:circle, ylabel="Hurst", label="")
                hline!(fig4, [0.5], width=[3], color=[:red], label="")
                plot(fig1, fig2, fig3, fig4, layout=(4,1), size=(1200,900), xticks=T0[1]:Dates.Hour(1):T0[end], legend=true)
                savefig(format("{}/{}.pdf",outdir, Dates.Date(T0[1])))
                # TimeSeries.writetimearray(Yt, format("{}/csv/{}.csv",outdir, day))
            end
        end
    else
    end


    if verbose
        printfmtln("Outputs saved in {}", outdir)
    end
end

main()
