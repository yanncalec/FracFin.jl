########## Parameters for scripts ##########

###### bspline-MLE ######

#### cross-scale only ####
maxscl = Npd÷1
minscl = maxscl - 60
# minscl, maxscl = Npd÷2, Npd÷1

wfct = 20  # empirically: 20 for vm=1, 30 for vm=2

parms = (
    estimator = "bspline_MLE_scale",
    method = :optim,  # method of optimizer

    # Parameters for B-Spline wavelet transform
    sclrng = 2*(minscl:2:maxscl),  # index range of inertial scales
    vm = 1,  # vanishing moment of the wavelet
    fmode = :causal,  # mode of filtration
    ρ = 0,  # threshold for truncation of small eigen values in the covariance matrix
    partial = true,  # true for whittle estimator

    # Parameters for moving window
    wsize = maxscl*wfct,  # size of moving window
    pov = maxscl,  # step of moving window
    boundary = :hard,  # boundary condition
)

base_estimator = x -> FracFin.bspline_MLE_estim(x, parms.sclrng, parms.vm; ρ=parms.ρ, partial=parms.partial)


