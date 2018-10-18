__precompile__()

using ArgParse
using Formatting

# push!(LOAD_PATH, "/home/han/Codes/Development")

# using LinearAlgebra
# using PyCall

import FracFin
# import DataFrames
# import DSP
# import GLM
# import Optim
# import ForwardDiff
# import CSV
# import TimeSeries
# import Dates
# import StatsBase

# import PyPlot; const plt = PyPlot
# plt.plt[:style][:use]("ggplot")

# using Plots
# # pyplot(reuse=true, size=(900,300))
# pyplot()
# set_theme(:ggplot2)


function prepare_data_for_estimator(X0::AbstractVector{T}, dlag::Int) where {T<:Real}
    dX = zero(X0)  # fGn estimator is not friendly with NaN
    dX[1+dlag:end] = X0[1+dlag:end]-X0[1:end-dlag]
    return dX
end


function parse_commandline()
    settings = ArgParseSettings("Apply the fGn maximum likelihood estimator.")

    @add_arg_table settings begin
        # "fGn"
        #     help = "fGn MLE"
        #     action = :command
        # "BSpline"
        #     help = "BSpline MLE"
        #     action = :command        
        "infile"
            help = "input csv file containing a time series"
            required = true
        "outdir"
            help = "output directory"
            required = true    
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
        "--dlag" 
            help = "time lag for finite difference"
            arg_type = Int
            default = 1
        "--tfmt"
            help = "time format for parsing csv file"
            arg_type = String
            default = "yyyy-mm-dd HH:MM:SS"
        "--ncol"
            help = "index of the column to be analyzed, if the input file contains multiple columns"
            arg_type = Int
            default = 1
        # "cmd"
        #     help = "name of estimator"
        #     action = :command
    end

    # add_arg_group(settings, "fGn-MLE")

    # @add_arg_table settings begin
    #     "--flag1"
    #         help = "an option without argument, i.e. a flag"
    #         action = :store_true
    # end

    return parse_args(settings)
end


function main()
    pargs = parse_commandline()
    # println("Parsed args:")
    # for (arg,val) in parsed_args
    #     println("  $arg  =>  $val")
    # end

    infile = pargs["infile"]
    outdir0 = pargs["outdir"]

    i1 = something(findlast(isequal('/'), infile), 1)
    i2 = something(findlast(isequal('.'), infile), length(infile))
    sname = infile[i1:i2]  # filename without extension

    # recover parameters
    wsize = pargs["wsize"]  # size of rolling window
    ssize = pargs["ssize"]  # size of i.i.d. sub-window
    dlen = pargs["dlen"]  # length of decorrelation
    pov = pargs["pov"]  # period of innovation
    dlag = pargs["dlag"]  # time-lag of finite difference
    # tfmt = pargs["tfmt"]  # time format
    ncol = pargs["ncol"]  # index of column

    # make the output folder
    outstr = format("wsize[{}]_ssize[{}]_dlen[{}]_pov[{}]_dlag[{}]", wsize, ssize, dlen, pov, dlag)
    outdir = format("{}/{}/{}/fGn-MLE/{}/", outdir0, sname, ncol, outstr)
    try
        mkpath(outdir)
    catch
    end

    # load data
    # toto = CSV.read(infile)
    toto = TimeSeries.readtimearray(infile, format=pargs["tfmt"], delim=',')
    data0 = TimeSeries.TimeArray(toto.timestamp, toto.values[:,ncol])
    data = data0[findall(.~isnan.(data0.values))] # remove nan values

    day_start = TimeSeries.Date(data.timestamp[1])
    day_end = TimeSeries.Date(data.timestamp[end])
    
    # # trunctation
    # data = data0[Dates.DateTime(day_start0):Dates.Minute(1):(Dates.DateTime(day_end0)+Dates.Day(1))];

    estim = X -> FracFin.fGn_MLE_estim(X, dlag; method=:optim, ε=1e-2)
    
    for day in day_start:Dates.Day(1):day_end
        t0 = Dates.DateTime(day)
        t1 = t0+Dates.Day(1)
        D0 = data[t0:Dates.Minute(1):t1]  # Minute or Hours, but Day won't work
        
        if length(D0)>0
            # log price
            vidx = .!isnan.(D0.values)  # exclude nan values
            T0 = D0.timestamp[vidx]  # timestamp
            V0 = D0.values[vidx]
            X0 = log.(V0)  # log transformed data
            
            # prepare data for estimator
            Xdata = prepare_data_for_estimator(X0, dlag)

            # estimation of hurst and σ
            res = FracFin.rolling_estim(estim, Xdata, (wsize, ssize, dlen), pov, mode=:causal)
            Ht = [x[2][1][1] for x in res]
            σt = [x[2][1][2] for x in res]
            tidx = [x[1] for x in res]        
            
            # fit in
            Ht0 = fill(NaN, length(T0)); Ht0[tidx] = Ht
            σt0 = fill(NaN, length(T0)); σt0[tidx] = σt
    
            # plot
            title_str = format("{}, wsize={}", sname, wsize)
            fig1 = plot(T0, X0, title=title_str, ylabel="Log-price", label="")
            fig2 = plot(T0, Ht0, shape=:circle, ylabel="Hurst", label="")
            fig3 = plot(T0, σt0, shape=:circle, ylabel="σ", label="")        
            fig4 = plot(T0, Ht0, ylim=[0,1], shape=:circle, ylabel="Hurst", label="")
            plot!(fig4, T0, 0.5*ones(length(T0)), w=3, color=:red, label="")                
            plot(fig1, fig2, fig3, fig4, layout=(4,1), size=(1200,1000), xticks=T0[1]:Dates.Hour(1):T0[end], legend=true)                
    #         fig2a = plot(T0, Ht0, ylabel="Hurst", label="Hurst")
    #         fig2b = twinx(fig2a)
    #         plot!(fig2b, T0, σt0, yrightlabel="σ", color=:red, label="σ")                
    #         plot(fig1, fig2a, layout=(2,1), size=(1200,600), legend=true)        
            outfile = format("{}/{}.pdf",outdir, day)
            savefig(outfile)
        end
    end
    
    day_start = TimeSeries.Date(data.timestamp[1])
    day_end = TimeSeries.Date(data.timestamp[end]) #-Dates.Day(1)
    
    t0 = Dates.DateTime(day_start)
    t1 = Dates.DateTime(day_end) #+Dates.Day(1)
    D0 = data[t0:Dates.Minute(1):t1]  # Minute or Hours, but Day won't work
    
    # log price
    vidx = .!isnan.(D0.values)  # exclude nan values
    T0 = D0.timestamp[vidx]  # timestamp
    V0 = D0.values[vidx]
    X0 = log.(V0)  # log transformed data
    
    Xdata = prepare_data_for_estimator(X0)
    
    # estimation of hurst and σ
    res= FracFin.rolling_estim(estim, Xdata, (wsize, ssize, dlen), pov, mode=:causal)
    Ht = [x[2][1][1] for x in res]
    σt = [x[2][1][2] for x in res]
    tidx = [x[1] for x in res]
    
    Ht0 = fill(NaN, length(T0)); Ht0[tidx] = Ht
    σt0 = fill(NaN, length(T0)); σt0[tidx] = σt;
    
    title_str = format("{}, wsize={}", sname, wsize)
    fig1 = plot(T0, X0, title=title_str, ylabel="Log-price", label="")
    fig2 = plot(T0, Ht0, shape=:circle, ylabel="Hurst", label="")
    fig3 = plot(T0, σt0, shape=:circle, ylabel="σ", label="")
    fig4 = plot(T0, Ht0, ylim=[0,1], shape=:circle, ylabel="Hurst", label="")
    plot!(fig4, T0, 0.5*ones(length(T0)), w=3, color=:red, label="")        
    plot(fig1, fig2, fig3, fig4, layout=(4,1), size=(1200,1000), legend=true)        
    
    #         fig2a = plot(T0, Ht0, ylabel="Hurst", label="Hurst")
    #         fig2b = twinx(fig2a)
    #         plot!(fig2b, T0, σt0, yrightlabel="σ", color=:red, label="σ")                
    #         plot(fig1, fig2a, layout=(2,1), size=(1200,600), legend=true)
    
    outfile = format("{}/continous-date.pdf",outdir)
    savefig(outfile)
    
    title_str = format("{}, wsize={}", sname, wsize)
    fig1 = plot(X0, title=title_str, ylabel="Log-price", label="")
    fig2 = plot(Ht0, shape=:circle, ylabel="Hurst", label="")
    fig3 = plot(σt0, shape=:circle, ylim=extrema(σt[150:end]), ylabel="σ", label="")
    fig4 = plot(Ht0, shape=:circle, ylim=[0,1], ylabel="Hurst", label="")
    plot!(fig4, 0.5*ones(length(T0)), w=3, color=:red, label="")
    plot(fig1, fig2, fig3, fig4, layout=(4,1), size=(1200,1000), legend=true)        
    
    outfile = format("{}/continous.pdf",outdir)
    savefig(outfile)    
end

main()
