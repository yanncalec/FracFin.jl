__precompile__()

using ArgParse
import ArgParse: parse_item

using Formatting

using Plots
# pyplot(reuse=true, size=(900,300))
pyplot()
theme(:ggplot2)

import TimeSeries
import Dates
import FracFin


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
    
    # data = TimeSeries.TimeArray(TimeSeries.timestamp(toto), TimeSeries.values(toto)[:,ncol])
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
    
    A0 = TimeSeries.TimeArray(T0, [X0 Ht0 σt0], ["Log_Price", "Hurst", "σ"])
    At = TimeSeries.TimeArray(T0[tidx], [X0[tidx] Ht σt], ["Log_Price", "Hurst", "σ"])

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
