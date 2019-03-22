########## Parameters for scripts ##########

###### bspline-MLE ######

#### cross-timescale ####
censcl = Npd÷1
minscl, maxscl = censcl-60, censcl+60
# minscl, maxscl = censcl÷2, censcl
# minscl, maxscl = censcl-60, censcl

wfct = 20  # empirically: 20 for vm=1, 30 for vm=2

parms = (
    estimator = "bspline_MLE_timescale",
    method = :optim,  # method of optimizer

    # Parameters for B-Spline wavelet transform
    sclrng = 2*(minscl:10:maxscl),  # index of working scale
    vm = 1,  # vanishing moment of the wavelet
    fmode = :causal,  # mode of filtration
    # ρ = 0,  # threshold for truncation of small eigen values in the covariance matrix
    partial = true,  # true for whittle estimator

    dfct = maxscl,  # down-sampling factor
    ssize = 5,  # cross-time range
    dlen = 1,  # length of decorrelation of sub window

    wsize = maxscl*wfct,  # size of moving window
    pov = Npd÷10,  # step of moving window
    boundary = :hard,  # boundary condition
)

base_estimator = x -> FracFin.bspline_MLE_estim(x, parms.sclrng, parms.vm, parms.ssize, parms.dfct, parms.dlen; partial=parms.partial)

