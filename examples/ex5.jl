using ApproximateBayesianComputing
const ABC = ApproximateBayesianComputing
using Distributions, Random

# Currently, hackishly included from ABC, but in the ABC namespace
#include(joinpath(dirname(pathof(ApproximateBayesianComputing)),"composite.jl"))
#include(joinpath(dirname(pathof(ApproximateBayesianComputing)),"beta_linear_transformed.jl"))
import ApproximateBayesianComputing.CompositeDistributions.CompositeDist
import ApproximateBayesianComputing.TransformedBetaDistributions.LinearTransformedBeta

Random.seed!(1234)

# Function to adjust originally proposed model parameters, so that they will be valid
function normalize_theta!(theta::Array) 
  @views theta[2:end]./= sum(theta[2:end])
  theta
end

# Set Prior for Population Parameters
max_rate = 3.0
nbins = 8
theta_true = vcat(max_rate*rand(),rand(nbins))
normalize_theta!(theta_true)

param_prior = CompositeDist(vcat(Uniform(0.0,max_rate),ContinuousDistribution[Uniform(0.0,1.0) for i in 1:nbins]))

# Function to test if the proposed model parameter are valid
is_valid(theta::Array) = all([minimum(param_prior.dist[i])<=theta[i]<=maximum(param_prior.dist[i]) for i in 1:length(theta)])

# Code to generate simulated data given array of model parameters
num_data_default = 1000
function draw_dirchlet_multinomial_with_poisson_rate(theta::Array)
   counts = zeros(Int64,length(theta)-1)
   n = rand(Poisson(theta[1]))
   d_cat = Categorical(theta[2:end])
   for i in 1:n
     counts[rand(d_cat)] += 1
   end
   counts
end

function gen_data(theta::Array, n::Integer = num_data_default)
   data = Array{Int64}( undef, (length(theta)-1, n) )
   for i in 1:n
      data[:,i] = draw_dirchlet_multinomial_with_poisson_rate(theta)
   end
   data     
end

function calc_mean_per_bin(data::Array{Int64,2})
   mean(data,dims=2)
end

calc_dist_l1(x::Array{Float64},y::Array{Float64}) = sum(abs.(x.-y))

function make_proposal_dist_multidim_beta(theta::AbstractArray{Float64,2}, weights::AbstractArray{Float64,1},  tau_factor::Float64; verbose::Bool = false)
  
  theta_mean =  sum(theta.*weights',dims=2) # weighted mean for parameters
  tau = tau_factor*ABC.var_weighted(theta'.-theta_mean',weights)  # scaled, weighted covar for parameters

  function alpha(x_bar::T, v_bar::T) where T<: Real 
    x_bar * (((x_bar * (1 - x_bar)) / v_bar) - 1)
  end
  function beta(x_bar::T, v_bar::T) where T<: Real 
    (1 - x_bar) * (((x_bar * (1 - x_bar)) / v_bar) - 1)
  end
  
  #=
  println("mean= ",theta_mean)
  println("var= ",tau)
  for i in 1:length(theta_mean)
     println("a= ",alpha(theta_mean[i],tau[i]), "  b= ",beta(theta_mean[i],tau[i]))
  end
  =#

  dist = ApproximateBayesianComputing.CompositeDistributions.CompositeDist(
         vcat(LinearTransformedBeta(alpha(theta_mean[1]/max_rate,tau[1]/max_rate^2), beta(theta_mean[1]/max_rate,tau[1]/max_rate^2),xmin=0.0,xmax=max_rate), 
         ContinuousDistribution[ Beta(alpha(theta_mean[i],tau[i]), beta(theta_mean[i],tau[i])) for i in 2:length(theta_mean)]   ))
end

function make_proposal_dist_multidim_beta(pop::abc_population_type, tau_factor::Float64; verbose::Bool = false)
	make_proposal_dist_multidim_beta(pop.theta, pop.weights, tau_factor, verbose=verbose)
end


# Tell ABC what it needs to know for a simulation
abc_plan = abc_pmc_plan_type(gen_data,calc_mean_per_bin,calc_dist_l1, param_prior; is_valid=is_valid,normalize=normalize_theta!,make_proposal_dist=make_proposal_dist_multidim_beta,tau_factor=1.0,target_epsilon=0.01*nbins,num_max_attempt=1000);

# Generate "true/observed data" and summary statistics
data_true = abc_plan.gen_data(theta_true)
ss_true = abc_plan.calc_summary_stats(data_true)
#println("theta= ",theta_true," ss= ",ss_true, " d= ", 0.)

# Run ABC simulation
@time pop_out = run_abc(abc_plan,ss_true;verbose=true);


