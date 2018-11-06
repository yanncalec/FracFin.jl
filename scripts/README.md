# Examples of Usage
Difference between estimators is greater on real data than on simulated data (of
pure fBm). This is especially true as the scale becomes large. To get similar
estimates, it is advised to work with small scales. 

### bspline-scalogram
```sh
julia rolling_estim_fBm.jl --wsize=250 --ssize=1 --ncol=1 -v $INFILE $OUTDIR bspline-scalogram --sclidx=2:10
```

### fWn-bspline-MLE
cross-time estimation doesn't work (constant estimation)
without cross-time:
```sh
julia rolling_estim_fBm.jl --wsize=250 --ssize=1 --ncol=1 -v $INFILE $OUTDIR fWn-bspline-MLE --sclidx=2:10
```

### fGn-MLE
```sh
julia rolling_estim_fBm.jl --wsize=250 --ssize=50 --ncol=1 -v $INFILE $OUTDIR fGn-MLE
```

### powlaw
```sh
julia rolling_estim_fBm.jl --wsize=250 --ncol=1 -v $INFILE $OUTDIR powlaw --dlags=2:10
```

