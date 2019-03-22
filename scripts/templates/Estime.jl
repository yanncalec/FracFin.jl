# using Revise
using Formatting
using LinearAlgebra
using Statistics, StatsBase

import DataFrames
import TimeSeries
import Dates
import FileIO
# import GLM

# import PyPlot; const plt = PyPlot
# plt.plt[:style][:use]("ggplot")

using Plots
# pyplot(reuse=true, size=(1000,250))
pyplot()
theme(:ggplot2)

Plots.scalefontsizes(1.)
# fnt = Plots.font("Helvetica", 10.0)
# default(titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)

import FracFin

include("Utils.jl")

########### Load data and Visualization ##########
include("Parms_IO.jl")

project = "Analysis of Lazard NY dataset 2"  # name of the current project

outdir_root = homedir() * "/Outputs/$(project)/"  # root output directory

Market = "Equity_1"  # market name
# input csv file
infile = homedir() * "/Database/Economy_and_Finance/Lusenn/Lazard NY 2/$(Market).csv"

println("Loading data...")
DataRaw, DataSplt = load_data(infile, parms_io)
Component = TimeSeries.colnames(DataRaw)[1]  # name of the working time-series
Price = TimeSeries.values(DataRaw)
TimeStamp = TimeSeries.timestamp(DataRaw)
LogPrice = log.(Price)  # supposed to be a fBm
Npd = isnothing(DataSplt) ? nothing : length(DataSplt[1])  # number of points per day, if meaningful

outdir_component = outdir_root * "/$(Market)/$(Component)/"  # outdir of the working time-series


# #### Visualization of data ####
# println("Plotting data...")

# outdir_fig = outdir_component * "/data/"
# try mkpath(outdir_fig) catch end

# # fig1 = plot(TimeSeries.values(Data), label="Raw price", title=cname, size=(1000,250))
# # savefig(outdir_fig * "/$(cname).pdf")

# fig = plot(DataRaw, label="Raw price", title=Component, size=(1000,250))
# fname = outdir_fig * "/$(Component).pdf"
# savefig(fig, fname)

# if parms_io.intraday
#     outdir_fig = outdir_component * "/data/intraday/"
#     try mkpath(outdir_fig) catch end

#     for data in DataSplt
#         date = Dates.Date(TimeSeries.timestamp(data)[1])

#         fig = plot(data, label="Raw price", size=(1000,250))
#         savefig(outdir_fig * "/$(date).pdf")
#     end
# end


########## Estimation ##########
include("Parms_BSpline_MLE_timescale.jl")
# include("Parms_BSpline_MLE_scale.jl")
# include("Parms_BSpline_MLE_time.jl")
# include("Parms_BSpline_Scalogram.jl")

rolling_window_estimator = (estimator, X) -> FracFin.rolling_apply(estimator, X, parms.wsize, 1, parms.pov; mode=:causal, boundary=parms.boundary)

outdir = outdir_component * "/estimation/$(parms)/"
try mkpath(outdir) catch end

#### Wavelet transform ####
W0, Mask = FracFin.cwt_bspline(LogPrice, parms.sclrng, parms.vm, parms.fmode)
# W0, Mask = FracFin.cwt_haar(LogPrice, parms.sclrng, parms.fmode);

# # truncation of boundary points
# t0 = findall(prod(Mask, dims=2))[1][1]
# t1 = findall(prod(Mask, dims=2))[end][1]
# Wt = W0[t0:t1, :]


#### Apply rolling window estimator ####
println("Applying rolling window estimation...")

# The rolling window estimator should be applied on `W0` but not `Wt` otherwise the output time indexes in `res` will not be aligned with the original ones. Use post-processing to remove the boundary coefficients in `W0` if necessary.
res = rolling_window_estimator(base_estimator, W0')

#### Post-processing of results ####
Tx = [x[1] for x in res]  # time index of estimation
Ts = TimeStamp[Tx] # time stamp of estimation
Ht = [x[2].hurst for x in res]  # hurst
Vt = [x[2].σ for x in res]  # volatility
# Et = [x[2].η for x in res]  # scaled log-volatility
# Rt = [x[2].residual for x in res];  # residual
Lt = [x[2].loglikelihood for x in res]  # log-likelihood

# Hs = [try GLM.stderror(x[2].optimizer)[2] catch NaN end for x in res]  # stderror of hurst
# Es = [try GLM.stderror(x[2].optimizer)[1] catch NaN end for x in res]  # stderror of scaled log-volatility
# cp = FracFin.normal_moment_factor(parms.pow)  # constant factor depending on p
# Vs = exp.((Es.-log(cp))/parms.pow)/parms.pow

# At0 = TimeSeries.TimeArray(Ts, [Tx Price[Tx] Ht Vt Rt], Symbol.(["Index", "Price", "Hurst", "Volatility", "Residual"]));
At0 = TimeSeries.TimeArray(Ts, [Tx Price[Tx] Ht Vt Lt], Symbol.(["Index", "Price", "Hurst", "Volatility", "LogLikelihood"]));

FileIO.save(outdir * "/results.jld2", Dict("parameters"=>parms, "estimates"=>At0))
TimeSeries.writetimearray(At0, outdir * "/results.csv")

#### Visualization ####
println("Plotting results of estimation...")

At = At0[5:end]  # truncation of boundary wavelet coefficients

fig1 = plot(At.Price, label="Price")
fig2 = plot(At.Hurst, shape=:circle, label="Hurst", ylim=(0.,1.))
hline!(fig2, [0.5], alpha=0.5, width=2, color=:red, label="")
fig3 = plot(At.Volatility, shape=:circle, label="Volatility")
fig4 = plot(At.LogLikelihood, shape=:circle, label="LogLikelihood")
# fig4 = plot(At.Residual, shape=:circle, label="Residual")

fig = vplot(fig1, fig2, fig3, fig4, link=:x)
outfile = outdir * "/results.pdf"
savefig(fig, outfile)


println("Results saved in $(outdir).")
