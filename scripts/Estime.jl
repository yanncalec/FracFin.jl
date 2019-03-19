# using Revise
using Formatting
using LinearAlgebra
using Statistics, StatsBase
# using FileIO

import DataFrames
import TimeSeries
import Dates
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
include("Parms.jl")

########### Load data and Visualization ##########
outdir_root = homedir() * "/tmp/Outputs/"  # root output directory

Market = "Bitcoin"  # market name
infile = "./$(Market).csv"  # input csv file

println("Loading data...")

DataRaw, DataSplt = load_data(infile, parms_io)
Component = TimeSeries.colnames(DataRaw)[1]  # name of the component

outdir_component = outdir_root * "/$(Market)/$(Component)/"
# try mkpath(outdir_market) catch end

# Visualization of data
println("Plotting data...")

outdir_fig = outdir_component * "/data/"
try mkpath(outdir_fig) catch end

# fig1 = plot(TimeSeries.values(Data), label="Raw price", title=cname, size=(1000,250))
# savefig(outdir_fig * "/$(cname).pdf")

fig = plot(DataRaw, label="Raw price", title=Component, size=(1000,250))
fname = outdir_fig * "/$(Component).pdf"
savefig(fig, fname)

if parms_io.intraday
    outdir_fig = outdir_component * "/data/intraday/"
    try mkpath(outdir_fig) catch end

    for data in DataSplt
        date = Dates.Date(TimeSeries.timestamp(data)[1])

        fig = plot(data, label="Raw price", size=(1000,250))
        savefig(outdir_fig * "/$(date).pdf")
    end
end


########## Estimation ##########


