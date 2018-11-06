__precompile__()

using ArgParse
import ArgParse: parse_item

using Formatting

using Plots
# pyplot(reuse=true, size=(900,300))
pyplot()
theme(:ggplot2)

# # using LinearAlgebra
# # using PyCall

# # import DataFrames
# # import DSP
# # import GLM
# # import Optim
# # import ForwardDiff
# # import CSV
import TimeSeries
import Dates
# import StatsBase

# # import PyPlot; const plt = PyPlot
# # plt.plt[:style][:use]("ggplot")

import FracFin

function split_timearray_by_day(data::TimeArray)
    day_start = TimeSeries.Date(TimeSeries.timestamp(data)[1])
    day_end = TimeSeries.Date(TimeSeries.timestamp(data)[end]) #-Dates.Day(1)
    res = []
    for day in day_start:Dates.Day(1):day_end
        t0 = Dates.DateTime(day)
        t1 = t0+Dates.Day(1) # -Dates.Minute(1)
        D0 = data[t0:Dates.Minute(1):t1]
        if length(D0)>0
            push!(res, D0)  # Minute or Hours, but Day won't work
        end
    end
    return res
end

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


function prepare_data_for_estimator_fGn(X0::AbstractVector{T}, dlag::Int) where {T<:Real}
    dX = fill(NaN, size(X0))  # fGn estimator is not friendly with NaN
    dX[1+dlag:end] = X0[1+dlag:end]-X0[1:end-dlag]
    return dX
end


function prepare_data_for_estimator_bspline(X0::AbstractVector{T}, sclrng::AbstractVector{Int}, vm::Int) where {T<:Real}
    W0, Mask = FracFin.cwt_bspline(X0, sclrng, vm, :left)
    t0 = findall(all(Mask, dims=2)[:])[1]
    # t1 = findall(all(Mask, dims=2)[:])[end]
    W0[1:t0-1,:] .= NaN
    return transpose(W0)
end


function parse_commandline()
    settings = ArgParseSettings("Apply the fWn-bspline maximum likelihood estimator.")

    @add_arg_table settings begin
        "--intraday"
        help = "apply estimation on a daily basis"
        action = :store_true
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
        "--dwin"
        help = "number of intraday window"
        arg_type = Int
        default = 0
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
    intraday = parsed_args["intraday"]  # intraday estimation
    wsize = parsed_args["wsize"]  # size of rolling window
    ssize = parsed_args["ssize"]  # size of sub window
    dlen = parsed_args["dlen"]  # length of decorrelation
    pov = parsed_args["pov"]  # period of innovation
    dwin = parsed_args["dwin"]  # number of intraday window
    tfmt = parsed_args["tfmt"]  # time format
    ncol = parsed_args["ncol"]  # index of column
    verbose = parsed_args["verbose"]  # print messages

    # recover command options
    prepare_data_for_estimator::Function = X->()  # function for data preparation
    outstr::String = ""
    estim::Function = X->()
    trans::Function = X->vec(X)

    if cmd  == "powlaw"
        ssize = wsize
        dlen = 1
        pow = parsed_args[cmd]["pow"]  # time-lags of finite difference
        dlags = parsed_args[cmd]["dlags"]  # time-lags of finite difference
        outstr = format("wsize[{}]_pov[{}]_pow[{}]_dlags[{}]", wsize, pov, pow, dlags)
        prepare_data_for_estimator = X -> X
        estim = X -> FracFin.powlaw_estim(X, dlags, pow)
    elseif cmd  == "fGn-MLE"
        dlag = parsed_args[cmd]["dlag"]  # time-lag of finite difference
        outstr = format("wsize[{}]_ssize[{}]_dlen[{}]_pov[{}]_dlag[{}]", wsize, ssize, dlen, pov, dlag)
        prepare_data_for_estimator = X -> prepare_data_for_estimator_fGn(X, dlag)
        estim = X -> FracFin.fGn_MLE_estim(X, dlag; method=:optim, ε=1e-2)
    elseif cmd == "fWn-bspline-MLE"
        sclrng = 2 * parsed_args[cmd]["sclidx"]  # range of scales
        vm = parsed_args[cmd]["vm"]  # vanishing moments
        outstr = format("wsize[{}]_ssize[{}]_dlen[{}]_pov[{}]_vm[{}]_sclrng[{}]", wsize, ssize, dlen, pov, vm, sclrng)
        prepare_data_for_estimator = X -> prepare_data_for_estimator_bspline(X, sclrng, vm)
        estim = X -> FracFin.fWn_bspline_MLE_estim(X, sclrng, vm; method=:optim, ε=1e-2)
    elseif cmd == "bspline-scalogram"
        ssize = wsize
        dlen = 1
        sclrng = 2 * parsed_args[cmd]["sclidx"]  # range of scales
        pow = parsed_args[cmd]["pow"]  # time-lags of finite difference
        vm = parsed_args[cmd]["vm"]  # vanishing moments
        outstr = format("wsize[{}]_pov[{}]_pow[{}]_vm[{}]_sclrng[{}]", wsize, pov, pow, vm, sclrng)
        prepare_data_for_estimator = X -> prepare_data_for_estimator_bspline(X, sclrng, vm)
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
    # toto = CSV.read(infile)
    toto = TimeSeries.readtimearray(infile, format=tfmt, delim=',')
    cname = TimeSeries.colnames(toto)[ncol]  # name of the column

    t0 = Dates.Hour(9) + Dates.Minute(5)
    t1 = Dates.Hour(17) + Dates.Minute(24)

    data0 = toto[cname]
    sdata0 = FracFin.split_by_day_with_truncation(data0, t0, t1)  # splitted data of identical length
    data = vcat(sdata0...)

    # data = TimeArray(TimeSeries.timestamp(toto), TimeSeries.values(toto)[:,ncol])
    any(isnan.(TimeSeries.values(data))) && error("NaN values detected in input data!")

    # make the output folder
    outdir = format("{}/{}/{}/{}/{}/", outdir0, sname, cname, cmd, outstr)
    try
        mkpath(outdir)
        mkpath(outdir * "/csv/")
    catch
    end

    # process consolidated dataset
    if verbose
        printfmtln("Processing the consolidated dataset...")
    end

    T0, X0 = TimeSeries.timestamp(data), log.(TimeSeries.values(data))

    # day_start = TimeSeries.Date(data.timestamp[1])
    # day_end = TimeSeries.Date(data.timestamp[end]) + Dates.Day(1)
    # t0 = Dates.DateTime(day_start)
    # t1 = Dates.DateTime(day_end)
    # D0 = data[t0:Dates.Minute(1):t1]  # Minute or Hours, but Day won't work  <- Bug here
    # T0, X0 = D0.timestamp, log.(D0.values)  # log price

    Xdata = prepare_data_for_estimator(X0)

    # estimation of hurst and σ
    res= FracFin.rolling_estim(estim, Xdata, (wsize, ssize, dlen), pov, trans; mode=:causal)
    Ht = [x[2][1][1] for x in res]
    σt = [x[2][1][2] for x in res]
    tidx = [x[1] for x in res]
    # fit in
    Ht0 = fill(NaN, length(T0)); Ht0[tidx] = Ht
    σt0 = fill(NaN, length(T0)); σt0[tidx] = σt

    A0 = TimeArray(T0, [X0 Ht0 σt0], ["Log_Price", "Hurst", "σ"])
    At = TimeArray(T0[tidx], [X0[tidx] Ht σt], ["Log_Price", "Hurst", "σ"])

    #     title_str = format("{}, wsize={}", sname, wsize)
    # fig1 = plot(A0["Log-price"], title=title_str, ylabel="Log-price", label="")
    # fig2 = plot(A0["Hurst"], shape=:circle, ylabel="Hurst", label="")
    # fig3 = plot(A0["σ"], shape=:circle, ylabel="σ", label="")
    # fig4 = plot(A0["Hurst"], ylim=[0,1], shape=:circle, ylabel="Hurst", label="")
    # plot!(fig4, T0, 0.5*ones(length(T0)), w=3, color=:red, label="")
    # plot(fig1, fig2, fig3, fig4, layout=(4,1), size=(1200,1000), legend=true)
    # outfile = format("{}/estime.pdf",outdir)
    # savefig(outfile)

    fig1 = plot(A0[:Log_Price], ylabel="Log Price", label="")
    fig2 = plot(A0[:Hurst], shape=:circle, ylabel="Hurst", label="")
    fig3 = plot(A0[:σ], shape=:circle, ylabel="σ", label="")
    fig4 = plot(A0[:Hurst], ylim=[0,1], shape=:circle, ylabel="Hurst", label="")
    plot!(fig4, T0, 0.5*ones(length(T0)), w=3, color=:red, label="")
    plot(fig1, fig2, fig3, fig4, layout=(4,1), size=(1200,1000), legend=true)
    savefig(format("{}/estimate-ts.pdf",outdir))

    fig1 = plot(TimeSeries.values(At[:Log_Price]), ylabel="Log Price", label="")
    fig2 = plot(TimeSeries.values(At[:Hurst]), shape=:circle, ylabel="Hurst", label="")
    fig3 = plot(TimeSeries.values(At[:σ]), shape=:circle, ylabel="σ", label="")
    fig4 = plot(TimeSeries.values(At[:Hurst]), ylim=[0,1], shape=:circle, ylabel="Hurst", label="")
    plot!(fig4, 1:length(tidx), 0.5*ones(length(tidx)), w=3, color=:red, label="")
    plot(fig1, fig2, fig3, fig4, layout=(4,1), size=(1200,1000), legend=true)
    savefig(format("{}/estimate.pdf",outdir))
    TimeSeries.writetimearray(At, format("{}/csv/estimate.csv",outdir))

    if intraday
        S0 = split_timearray_by_day(A0)
        St = split_timearray_by_day(At)

        for (Y0, Yt) in zip(S0, St)
            day = Dates.Date(TimeSeries.timestamp(Y0)[1])
            fig1 = plot(Y0[:Log_Price], ylabel="Log Price", label="")
            fig2 = plot(Y0[:Hurst], shape=:circle, ylabel="Hurst", label="")
            fig3 = plot(Y0[:σ], shape=:circle, ylabel="σ", label="")
            fig4 = plot(Y0[:Hurst], ylim=[0,1], shape=:circle, ylabel="Hurst", label="")
            plot!(fig4, Y0.timestamp, 0.5*ones(length(T0)), w=3, color=:red, label="")
            plot(fig1, fig2, fig3, fig4, layout=(4,1), size=(1200,1000), xticks=T0[1]:Dates.Hour(1):T0[end], legend=true)
            savefig(format("{}/{}.pdf",outdir, day))
            TimeSeries.writetimearray(Yt, format("{}/csv/{}.csv",outdir, day))
        end
    end

    # if intraday
    #     day_iter = day_start:Dates.Day(1):day_end
    #     for (n,day) in enumerate(day_iter)
    #         if verbose
    #             printfmt("Processing {} of {} days...\r", n, length(day_iter))
    #         end

    #         t0 = Dates.DateTime(day)
    #         t1 = t0+Dates.Day(1)
    #         D0 = data[t0:Dates.Minute(1):t1]  # Minute or Hours, but Day won't work

    #         if length(D0)>0
    #             # log price
    #             T0, X0 = D0.timestamp, log.(D0.values)

    #             # prepare data for estimator
    #             Xdata = prepare_data_for_estimator(X0)

    #             # estimation of hurst and σ
    #             res = FracFin.rolling_estim(estim, Xdata, (wsize, ssize, dlen), pov, trans; mode=:causal)
    #             Ht = [x[2][1][1] for x in res]
    #             σt = [x[2][1][2] for x in res]
    #             tidx = [x[1] for x in res]
    #             # fit in
    #             Ht0 = fill(NaN, length(T0)); Ht0[tidx] = Ht
    #             σt0 = fill(NaN, length(T0)); σt0[tidx] = σt

    #             # plot
    #             title_str = format("{}, wsize={}", sname, wsize)
    #             fig1 = plot(T0, X0, title=title_str, ylabel="Log-price", label="")
    #             fig2 = plot(T0, Ht0, shape=:circle, ylabel="Hurst", label="")
    #             fig3 = plot(T0, σt0, shape=:circle, ylabel="σ", label="")
    #             fig4 = plot(T0, Ht0, ylim=[0,1], shape=:circle, ylabel="Hurst", label="")
    #             plot!(fig4, T0, 0.5*ones(length(T0)), w=3, color=:red, label="")
    #             plot(fig1, fig2, fig3, fig4, layout=(4,1), size=(1200,1000), xticks=T0[1]:Dates.Hour(1):T0[end], legend=true)
    #             #         fig2a = plot(T0, Ht0, ylabel="Hurst", label="Hurst")
    #             #         fig2b = twinx(fig2a)
    #             #         plot!(fig2b, T0, σt0, yrightlabel="σ", color=:red, label="σ")
    #             #         plot(fig1, fig2a, layout=(2,1), size=(1200,600), legend=true)
    #             outfile = format("{}/{}.pdf",outdir, day)
    #             savefig(outfile)

    #             At = TimeArray(T0[tidx], [Ht σt], ["Hurst", "σ"])
    #             outfile = format("{}/csv/{}.csv",outdir, day)
    #             TimeSeries.writetimearray(At, outfile)
    #         end
    #     end
    # end

    if verbose
        printfmtln("Outputs saved in {}", outdir)
    end
end

main()
