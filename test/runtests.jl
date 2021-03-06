using Test
using Distributed
using LinearAlgebra
using NPZ
using Clustering

@everywhere using Random
@everywhere using DPMMSubClustersStreaming
include("multinomial_tests.jl")
include("niw_tests.jl")
include("unitests.jl")
include("module_tests.jl")



# addprocs(2)
# @everywhere using Random
# @everywhere using DPMMSubClustersStreaming

# include("multinomial_tests.jl")
# include("niw_tests.jl")
# include("unitests.jl")
# include("module_tests.jl")
