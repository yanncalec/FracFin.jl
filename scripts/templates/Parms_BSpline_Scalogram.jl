########## Parameters for scripts ##########

######## moving window estimators ########

###### bspline-scalogram ######
sfct = 1/2 # factor of scale
cscl = Int(Npd*sfct)  # central scale
wscl = 10  # half scale window width, e.g. <=10
minscl, maxscl = max(1, cscl-wscl), cscl+wscl
# maxscl = Npd - 0*Npd÷4
# # minscl = maxscl ÷ 2
# minscl = maxscl - Npd÷2

wfct = 40  # factor of moving window

parms = (
    estimator = "bspline_scalogram",
    method = :optim,  # method of optimizer

    # Parameters for B-Spline wavelet transform
    scalefactor = sfct,
    sclrng = 2*(minscl:maxscl),  # index range of inertial scales
    vm = 2,  # vanishing moment of the wavelet
    pows = 1.0:0.1:2.0, # multifractal moments
    # pows = [2.0],
    reweight = false,  # reweighted regression, warning: heavy computation if turned on
    fmode = :causal,  # mode of filtration

    # Parameters for moving window
    windowfactor = wfct,
    wsize = cscl*wfct,  # size of moving window
    pov = Npd÷1,  # step of moving window
    boundary = :soft,  # boundary condition
)

# multi-fractal estimator
mono_estimator = (x,p) -> FracFin.bspline_scalogram_estim(x, parms.sclrng, parms.vm; pow=p, method=parms.method, reweight=parms.reweight)
base_estimator = x -> FracFin.multifractal_estim(mono_estimator, x, parms.pows)

# mono-fractal estimator
# base_estimator = x -> FracFin.bspline_scalogram_estim(x, parms.sclrng, parms.vm; pow=parms.pow, method=parms.method, reweight=parms.reweight)
