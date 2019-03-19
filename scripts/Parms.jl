########## Parameters for scripts ##########

# parameters for loading data
parms_io = (
    # for intraday data only
    intraday = true,
    # date range of data
    date_start = Dates.DateTime("0000-01-01"), date_end = Dates.DateTime("9999-01-01"),
    # daily time range
    wa = Dates.Hour(9) + Dates.Minute(30), wb = Dates.Hour(16) + Dates.Minute(0),
    adjusted = false,  # adjust intraday data

    # index of the column in the dataframe to work on
    column_index = 1,
)


parms_estim = (
    moving_average
    name = "variogram",
)

# Multifractal
lagmax = 60

parms = (
    estimator = "variogram",
    method = :optim,  # method of optimizer
    adjusted = true,  # true for adjusted data, false for raw data
    # parameters for variogram estimation
    dlags = (lagmax÷2):lagmax,  # inertial scale range
#     pows = 1.0:0.1:2.0, # multifractal moments
    pows = [1.0],
    cmode = :causal,  # mode of convolution
    wsize = lagmax*15,  # size of moving window
    pov = lagmax÷2,  # step of moving window
)

estimator = (x,p) -> FracFin.variogram_estim(x, parms.dlags; pow=p, method=parms.method)
method = x -> FracFin.multifractal_estim(estimator, x, parms.pows)
# method = x -> FracFin.variogram_estim(x, parms.dlags; pow=parms.pow, method=parms.method)

outdir = outdir0 * "/estimation/$(cname)/$(parms)/"
try mkpath(outdir) catch end