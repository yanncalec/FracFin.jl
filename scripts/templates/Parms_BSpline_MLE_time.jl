########## Parameters for scripts ##########

###### bspline-MLE ######

#### cross-time only ####
maxscl = Npd÷1

wfct = 20  # empirically: 20 for vm=1, 30 for vm=2

parms = (
    estimator = "bspline_MLE_time",
    method = :optim,  # method of optimizer

    # Parameters for B-Spline wavelet transform
    sclrng = 2*[maxscl],  # index of working scale
    vm = 1,  # vanishing moment of the wavelet
    fmode = :causal,  # mode of filtration
    ρ = 0,  # threshold for truncation of small eigen values in the covariance matrix

    dfct = maxscl,  # down-sampling factor
    ssize = 5,  # size of sub window, set ssize=wsize/dfct to disable sub window
    dlen = 1,  # length of decorrelation of sub window

    wsize = maxscl*wfct,  # size of moving window
    pov = maxscl,  # step of moving window
    boundary = :hard,  # boundary condition
)

base_estimator = x -> FracFin.bspline_MLE_estim(x, parms.sclrng, parms.vm, parms.ssize, parms.dfct, parms.dlen; ρ=parms.ρ, partial=false)

