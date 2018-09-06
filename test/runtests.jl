using Base.Test
import FracFin

# write your own tests here
# @test 1 == 2

grid = 1:10^6


fBM = FracFin.FractionalBrownianMotion(0.7)
# sampler = FracFin.CholeskySampler(fBM, grid)
sampler = FracFin.CircSampler(fBM, grid)
# sampler = FracFin.CircSampler{FracFin.FractionalBrownianMotion, UnitRange{Int64}}(fBM, grid)
X = rand(sampler)

# fGN = FracFin.FractionalGaussianNoise(0.7)
# sampler = FracFin.CircSampler(fGN, grid)
# # sampler = CholeskySampler{FractionalGaussianNoise, UnitRange{Int64}}(fGN, grid)
# X = FracFin.rand(sampler)

# print(X)
