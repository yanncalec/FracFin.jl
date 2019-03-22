__precompile__()

using Formatting
using Plots
# pyplot(reuse=true, size=(900,300))
pyplot()
theme(:ggplot2)

using ArgParse
import ArgParse: parse_item

import TimeSeries
import Distributions
import Dates
import FracFin

function parse_commandline()
    settings = ArgParseSettings("Plot results of estimation.")

    @add_arg_table settings begin
        "infile"
        help = "input .csv file containing results of estimation"
        required = true
        "outdir"
        help = "output directory"
        required = true
        "--tfmt"
        help = "time format for parsing csv file"
        arg_type = String
        default = "yyyy-mm-ddTHH:MM:SS"
        "--ci"
        help = "size of the confident interval"
        default = 0.75
        "--intraday", "-i"
        help = "plot intraday results (if available)"
        action = :store_true
        "--verbose", "-v"
        help = "print message"
        action = :store_true
    end

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
    outdir = parsed_args["outdir"]

    tfmt = parsed_args["tfmt"]  # time format
    intraday = parsed_args["intraday"]  # intraday estimation
    ci = parsed_args["ci"]  # size of the confident interval
    verbose = parsed_args["verbose"]  # print messages

    # load data
    if verbose
        printfmtln("Loading file {}...", infile)
    end
    toto = TimeSeries.readtimearray(infile, format=tfmt, delim=',')
    X0 = toto[:Log_Price]
    Ht0 = toto[:Hurst]
    Hs0 = toto[:Hurst_std]
    σt0 = toto[:σ]
    σs0 = toto[:σ_std]

    # make the output folder
    try
        mkpath(outdir)
    catch
    end

    # plot overall estimate
    q = Distributions.quantile(Distributions.Normal(), (1+ci)/2)  # quantile

    fig1 = plot(X0, ylabel="Log Price", label="")
    fig2 = plot(Ht0, ribbon=(2*Hs0, 2*Hs0), ylim=[0,1], shape=:circle, ylabel="Hurst", label="")
    hline!(fig2, 0.5, w=3, color=:red, label="")
    fig3 = plot(σt0, ribbon=(q*σs0, q*σs0), shape=:circle, ylabel="σ", label="")
    plot(fig1, fig2, fig3, layout=(3,1), size=(1200,900), legend=true)
    savefig(format("{}/estimate.pdf",outdir))

    if intraday
        sdata = FracFin.window_split_timearray(toto, Dates.Hour(24); fillmode=:o, endpoint=true)

        for Y0 in sdata
            T0 = TimeSeries.timestamp(Y0)
            fig1 = plot(Y0[:Log_Price], ylabel="Log Price", label="")
            fig2 = plot(Y0[:Hurst], shape=:circle, ylabel="Hurst", label="")
            fig3 = plot(Y0[:σ], shape=:circle, ylabel="σ", label="")
            fig4 = plot(Y0[:Hurst], ylim=[0,1], shape=:circle, ylabel="Hurst", label="")
            hline!(fig4, 0.5, w=3, color=:red, label="")
            plot(fig1, fig2, fig3, fig4, layout=(4,1), size=(1200,900), xticks=T0[1]:Dates.Hour(1):T0[end], legend=true)
            savefig(format("{}/{}.pdf",outdir, Dates.Date(T0[1])))
            # TimeSeries.writetimearray(Yt, format("{}/csv/{}.csv",outdir, day))
        end
    end

    if verbose
        printfmtln("Outputs saved in {}", outdir)
    end
end

main()
