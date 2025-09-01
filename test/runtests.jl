using Test
using TestItems
using TestItemRunner

@run_package_tests

include("test_kernels.jl")
include("test_linalg.jl")
include("test_polynomials.jl")
