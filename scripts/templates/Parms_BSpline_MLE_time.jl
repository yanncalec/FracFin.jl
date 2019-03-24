########## Parameters for scripts ##########

###### bspline-MLE ######

#### cross-time only ####

cscl = Int(Npd*sfct)  # central scale
wfct = 40  # factor of moving window

parms = (
    estimator = "bspline_MLE_time",
    method = :optim,  # method of optimizer

    # Parameters for B-Spline wavelet transform
    scalefactor = sfct,
    sclrng = 2*[cscl],  # index of working scale
    vm = 1,  # vanishing moment of the wavelet
    fmode = :causal,  # mode of filtration
    ρ = 0,  # threshold for truncation of small eigen values in the covariance matrix

    dfct = cscl,  # down-sampling factor
    ssize = 10,  # size of sub window, set ssize=wsize/dfct to disable sub window
    dlen = 1,  # length of decorrelation of sub window

    windowfactor = wfct,
    wsize = cscl*wfct,  # size of moving window
    pov = Npd÷1,  # step of moving window
    boundary = :hard,  # boundary condition
)

base_estimator = x -> FracFin.bspline_MLE_estim(x, parms.sclrng, parms.vm, parms.ssize, parms.dfct, parms.dlen; ρ=parms.ρ, partial=false)

