########## Parameters for scripts ##########

###### bspline-MLE ######

#### cross-time only ####
maxscl = 60

parms = (
    estimator = "bspline_MLE_time",
    method = :optim,  # method of optimizer

    # Parameters for B-Spline wavelet transform
    sclrng = [maxscl],  # index of working scale
    vm = 1,  # vanishing moment of the wavelet
    fmode = :causal,  # mode of filtration
    ρ = 0,  # threshold for truncation of small eigen values in the covariance matrix
    partial = false,  # true for whittle estimator

    dfct = maxscl,  # down-sampling factor
    ssize = 5,  # size of sub window, set ssize=wsize/dfct to disable sub window
    dlen = 1,  # length of decorrelation of sub window

    wsize = maxscl*15,  # size of moving window
    pov = maxscl,  # step of moving window
    boundary = :hard,  # boundary condition
)

base_estimator = x -> FracFin.bspline_MLE_estim(x, 2*parms.sclrng, parms.vm, parms.ssize, parms.dlen, parms.dlen; ρ=parms.ρ, partial=parms.partial)

