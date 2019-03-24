#!/usr/bin/env python
# coding: utf-8

# Plot results of estimations

# get_ipython().run_line_magic('pylab', '')

import pandas as pd
import datetime as dt
import os
import pandas_datareader as pdr

import plotly.plotly as py
import plotly.graph_objs as go

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# from matplotlib.dates import (MONDAY, DateFormatter, MonthLocator, WeekdayLocator, date2num)
plt.style.use('ggplot')

from mpl_finance import plot_day_summary_oclh, candlestick_ohlc

params = {'legend.fontsize': 'xx-large',
#           'figure.figsize': (15, 5),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)


mondays = mdates.WeekdayLocator(mdates.MONDAY)
daysFmt = mdates.DateFormatter("%d %b %y")

def ohlc_plot(ax, quotes, width=0.5, label=""):
    quotes = quotes.reset_index().copy()
    quotes['Timestamp'] = quotes['Timestamp'].map(mdates.date2num)

    ax.xaxis_date()

    # plot_day_summary_oclh(ax, zip(date2num(quotes.index.to_pydatetime()),
    #                               quotes['open'], quotes['close'],
    #                               quotes['low'], quotes['high']),
    #                       ticksize=3, colorup='g', colordown='r')

    candlestick_ohlc(ax, quotes.values, width=width, colorup='g', colordown='r')

    if len(label) > 0:
        ax.legend([label])



market = "Equity_1"
indir = "/Users/hanwang/Outputs/Analysis of Lazard NY dataset 2/{}/".format(market)

outdir = indir + "/figures/"
try:
    os.makedirs(outdir)
except:
    pass


scale = 2
fname = indir + "/scale_{}.csv".format(scale)

df0 = pd.read_csv(fname, index_col=0, parse_dates=True, infer_datetime_format=True)
df = df0['2018-10-15':]
df_ohlc = df.resample('1D').ohlc()
# df_ohlc.reset_index(inplace=True)

# date_start = "2018"
# df = df[(df.index >= date1) & (df.index <= date2)]

fig, axes = plt.subplots(3,1,figsize=(20,15), sharex=True)
ohlc_plot(axes[0], df_ohlc.Price, label="Price")
ohlc_plot(axes[1], df_ohlc.Hurst, label="Hurst")
ohlc_plot(axes[2], df_ohlc.Volatility, label="Volatility")

# Set title
axes[0].set_title("{}, scale=1/{} day".format(market, scale))

ax = axes[1]
ax.axhline(0.5)
ax.set_ylim((0,1))

# Modify the x-axis: alignement wrt monday
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_major_formatter(daysFmt)
ax.autoscale_view()
ax.xaxis.grid(True, 'major')
ax.grid(True)

fig.autofmt_xdate()

plt.tight_layout()

outfile = outdir + "/scale_{}.pdf".format(scale)
savefig(outfile)
