########## Parameters for scripts ##########

######## moving window estimators ########

###### bspline-scalogram ######
maxscl = Npd

parms = (
    estimator = "bspline_scalogram",
    method = :optim,  # method of optimizer

    # Parameters for B-Spline wavelet transform
    sclrng = 2*(maxscl÷2:maxscl),  # index range of inertial scales
    vm = 1,  # vanishing moment of the wavelet
#     pows = 1.0:0.1:2.0, # multifractal moments
    pows = [2.0],
    reweight = false,  # reweighted regression, warning: heavy computation if turned on
    fmode = :causal,  # mode of filtration

    # Parameters for moving window
    wsize = maxscl*20,  # size of moving window
    pov = maxscl,  # step of moving window
    boundary = :hard,  # boundary condition
)

# multi-fractal estimator
mono_estimator = (x,p) -> FracFin.bspline_scalogram_estim(x, parms.sclrng, parms.vm; pow=p, method=parms.method, reweight=parms.reweight)
base_estimator = x -> FracFin.multifractal_estim(mono_estimator, x, parms.pows)

# mono-fractal estimator
# base_estimator = x -> FracFin.bspline_scalogram_estim(x, parms.sclrng, parms.vm; pow=parms.pow, method=parms.method, reweight=parms.reweight)
