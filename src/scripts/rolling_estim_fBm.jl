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

function split_timearray_by_day(data::TimeSeries.TimeArray)
    day_start = TimeSeries.Date(data.timestamp[1])
    day_end = TimeSeries.Date(data.timestamp[end]) #-Dates.Day(1)
    res = []
    for day in day_start:Dates.Day(1):day_end
        t0 = Dates.DateTime(day)
        t1 = t0+Dates.Day(1)
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
        N[1]:N[2]:N[3]
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


function prepare_data_for_estimator_fGn_MLE(X0::AbstractVector{T}, dlag::Int) where {T<:Real}
    dX = fill(NaN, size(X0))  # fGn estimator is not friendly with NaN
    dX[1+dlag:end] = X0[1+dlag:end]-X0[1:end-dlag]
    return dX
end


function prepare_data_for_estimator_fWn_bspline_MLE(X0::AbstractVector{T}, sclrng::AbstractVector{Int}, vm::Int) where {T<:Real}
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
        default = 10        
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
        help = "input csv file containing a time series"
        required = true
        "outdir"
        help = "output directory"
        required = true    
        "fGn-MLE"
        action = :command        # adds a command which will be read from an argument
        help = "fGn-MLE"
        "fWn-bspline-MLE"
        action = :command
        help = "fWn-bspline-MLE"
    end

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
        "--sclrng"
        help = "range of integer scales"
        arg_type = AbstractVector{Int}
        default = 4:2:50
    end

    settings["fWn-bspline-MLE"].description = "fWn description"  # this is how settings are tweaked for commands
    settings["fWn-bspline-MLE"].commands_are_required = true # this makes the sub-commands optional

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
    tfmt = parsed_args["tfmt"]  # time format
    ncol = parsed_args["ncol"]  # index of column
    verbose = parsed_args["verbose"]  # print messages

    # recover command options
    outstr0 = format("wsize[{}]_ssize[{}]_dlen[{}]_pov[{}]", wsize, ssize, dlen, pov)
    prepare_data_for_estimator::Function = X->()  # function for data preparation    
    outstr::String = ""
    estim::Function = X->()
    
    if cmd  == "fGn-MLE"
        dlag = parsed_args[cmd]["dlag"]  # time-lag of finite difference
        outstr = outstr0 * format("dlag[{}]", wsize, ssize, dlen, pov, dlag)
        prepare_data_for_estimator = X -> prepare_data_for_estimator_fGn_MLE(X, dlag)
        estim = X -> FracFin.fGn_MLE_estim(X, dlag; method=:optim, ε=1e-2)
    elseif cmd == "fWn-bspline-MLE"
        sclrng = parsed_args[cmd]["sclrng"]  # range of scales
        vm = parsed_args[cmd]["vm"]  # vanishing moments
        outstr = outstr0 * format("vm[{}]_sclrng[{}]", vm, sclrng)
        prepare_data_for_estimator = X -> prepare_data_for_estimator_fWn_bspline_MLE(X, sclrng, vm)
        estim = X -> FracFin.fWn_bspline_MLE_estim(X, sclrng, vm; method=:optim, ε=1e-2)
    else
        error("Unknown command")
    end
    
    # make the output folder    
    outdir = format("{}/{}/{}/{}/{}/", outdir0, sname, ncol, cmd, outstr)
    try
        mkpath(outdir)
        mkpath(outdir * "/csv/")
    catch
    end
    
    # load data
    if verbose
        printfmtln("Loading file {}...", infile)
    end
    # toto = CSV.read(infile)
    toto = TimeSeries.readtimearray(infile, format=tfmt, delim=',')
    data0 = TimeSeries.TimeArray(toto.timestamp, toto.values[:,ncol])
    data = data0[findall(.~isnan.(data0.values))] # remove nan values

    # process consolidated dataset
    if verbose
        printfmtln("Processing the consolidated dataset...")
    end
    day_start = TimeSeries.Date(data.timestamp[1])
    day_end = TimeSeries.Date(data.timestamp[end]) + Dates.Day(1)
    
    t0 = Dates.DateTime(day_start)
    t1 = Dates.DateTime(day_end)
    D0 = data[t0:Dates.Minute(1):t1]  # Minute or Hours, but Day won't work
    
    # log price
    T0, X0 = D0.timestamp, log.(D0.values)
    
    Xdata = prepare_data_for_estimator(X0)
    
    # estimation of hurst and σ
    res= FracFin.rolling_estim(estim, Xdata, (wsize, ssize, dlen), pov, mode=:causal)
    Ht = [x[2][1][1] for x in res]
    σt = [x[2][1][2] for x in res]
    tidx = [x[1] for x in res]
    # fit in
    Ht0 = fill(NaN, length(T0)); Ht0[tidx] = Ht
    σt0 = fill(NaN, length(T0)); σt0[tidx] = σt
    
    A0 = TimeSeries.TimeArray(T0, [X0 Ht0 σt0], ["Log-price", "Hurst", "σ"])
    At = TimeSeries.TimeArray(T0[tidx], [X0[tidx] Ht σt], ["Log-price", "Hurst", "σ"])

    #     title_str = format("{}, wsize={}", sname, wsize)
    # fig1 = plot(A0["Log-price"], title=title_str, ylabel="Log-price", label="")
    # fig2 = plot(A0["Hurst"], shape=:circle, ylabel="Hurst", label="")
    # fig3 = plot(A0["σ"], shape=:circle, ylabel="σ", label="")        
    # fig4 = plot(A0["Hurst"], ylim=[0,1], shape=:circle, ylabel="Hurst", label="")
    # plot!(fig4, T0, 0.5*ones(length(T0)), w=3, color=:red, label="")                
    # plot(fig1, fig2, fig3, fig4, layout=(4,1), size=(1200,1000), legend=true)                
    # outfile = format("{}/estime.pdf",outdir)
    # savefig(outfile)

    fig1 = plot(At["Log-price"].values, ylabel="Log-price", label="")
    fig2 = plot(At["Hurst"].values, shape=:circle, ylabel="Hurst", label="")
    fig3 = plot(At["σ"].values, shape=:circle, ylabel="σ", label="")        
    fig4 = plot(At["Hurst"].values, ylim=[0,1], shape=:circle, ylabel="Hurst", label="")
    plot!(fig4, 1:length(tidx), 0.5*ones(length(tidx)), w=3, color=:red, label="")                
    plot(fig1, fig2, fig3, fig4, layout=(4,1), size=(1200,1000), legend=true)                
    savefig(format("{}/estimate.pdf",outdir))
    TimeSeries.writetimearray(At, format("{}/csv/estimate.csv",outdir))

    if intraday
        S0 = split_timearray_by_day(A0)
        St = split_timearray_by_day(At)

        for (Y0, Yt) in zip(S0, St)
            day = Dates.Date(Y0.timestamp[1])
            fig1 = plot(Y0["Log-price"], ylabel="Log-price", label="")
            fig2 = plot(Y0["Hurst"], shape=:circle, ylabel="Hurst", label="")
            fig3 = plot(Y0["σ"], shape=:circle, ylabel="σ", label="")        
            fig4 = plot(Y0["Hurst"], ylim=[0,1], shape=:circle, ylabel="Hurst", label="")
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
    #             res = FracFin.rolling_estim(estim, Xdata, (wsize, ssize, dlen), pov, mode=:causal)
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

    #             At = TimeSeries.TimeArray(T0[tidx], [Ht σt], ["Hurst", "σ"])
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