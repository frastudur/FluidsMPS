import QuanticsGrids as QG
using QuanticsTCI: quanticscrossinterpolate
import TensorCrossInterpolation as TCI
using TCIITensorConversion
using ITensorMPS


R=10 #Resolution
dimension=2 #Space dimension
grid = QG.DiscretizedGrid{2}(R, (0,0), (1,1); includeendpoint = true)


#laminar flow 
ux = (x,y) -> 1.0
uy = (x,y) -> 1.0

# build mps with QuanticsTCI
u1Q, rank1, error1 = quanticscrossinterpolate(Float64, ux, grid) 
u2Q, rank2, error2 = quanticscrossinterpolate(Float64, uy, grid)

# convert to ITensorMPS format
ttx=TCI.TensorTrain(u1Q.tci)
tty=TCI.TensorTrain(u2Q.tci)

sites = siteinds("Qudit", R, dim=2^dimension)
ux=ITensorMPS.MPS(ttx, sites=sites)
uy=ITensorMPS.MPS(tty, sites=sites)

u = [ux, uy]


