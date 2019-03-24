########## Parameters for scripts ##########

###### bspline-MLE ######

#### cross-timescale ####

cscl = Int(Npd*sfct)  # central scale
wscl = 10  # half scale window width, e.g. <=10
minscl, maxscl = max(1, cscl-wscl), cscl+wscl
# maxscl = Npd - 0*Npd÷4
# # minscl = maxscl ÷ 2
# minscl = maxscl - Npd÷2

wfct = 40  # factor of moving window

parms = (
    estimator = "bspline_MLE_timescale",
    method = :optim,  # method of optimizer

    # Parameters for B-Spline wavelet transform
    scalefactor = sfct,
    sclrng = 2*(minscl:maxscl),  # index range of inertial scales
    # sclrng = 2*[minscl,maxscl],  # index of working scale
    vm = 1,  # vanishing moment of the wavelet
    fmode = :causal,  # mode of filtration
    # ρ = 0,  # threshold for truncation of small eigen values in the covariance matrix
    partial = true,  # true for whittle estimator

    dfct = cscl,  # down-sampling factor
    ssize = wfct,  # size of sub window, set ssize=wsize/dfct to disable sub window
    dlen = 1,  # length of decorrelation of sub window

    windowfactor = wfct,
    wsize = cscl*wfct,  # size of moving window
    pov = Npd÷1,  # step of moving window
    boundary = :hard,  # boundary condition
)

base_estimator = x -> FracFin.bspline_MLE_estim(x, parms.sclrng, parms.vm, parms.ssize, parms.dfct, parms.dlen; partial=parms.partial)

