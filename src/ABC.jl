module ABC

# package code goes here

using Statistics
using Distributions
using PDMats
using Distributed
using DistributedArrays
using Compat

export
  # types
  abc_plan_type,
  abc_pmc_plan_type,
  abc_population_type,

  # util methods

  # abc alg methods
  generate_theta,
  init_abc,
  update_abc_pop,
  run_abc


import StatsBase.sample
import Statistics: mean, median, maximum, minimum, quantile, std, var, cov, cor
import Base: rand
#import Compat.String
#import Compat: UTF8String, ASCIIString, readstring, is_windows
import Compat: @compat

# until added to distributions, use our own
include("GaussianMixtureModelCommonCovar.jl")
#include("CompositeDistributions.jl")

include("types.jl")
include("util.jl")
include("alg.jl")
include("alg_serial.jl")
include("alg_parallel.jl")
#include("alg_parallel_custom.jl")
include("make_proposal.jl")
include("log.jl")
#include("emulator.jl")

end # module
