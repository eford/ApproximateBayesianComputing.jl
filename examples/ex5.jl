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
max_rate = 3.
nbins = 13
theta_true = vcat(max_rate*0.1,rand(nbins))
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
   vec(mean(data,dims=2))
end

calc_dist_l1(x::Array{Float64},y::Array{Float64}) = sum(abs.(x.-y))/length(x) + abs(sum(x)-sum(y))
calc_dist_l2(x::Array{Float64},y::Array{Float64}) = sum(abs2.(x.-y))/length(x) + abs2(sum(x)-sum(y))


#using Distributions 
using SpecialFunctions
using Statistics

# https://en.wikipedia.org/wiki/Trigamma_function
function trigamma_x_gr_4(x::T) where T<: Real
   1/x + 0.5/x^2 + 1/(6*x^3) - 1/(30*x^5) + 1/(42*x^7) - 1/(30*x^9) + 5/(66*x^11) - 691/(2730*x^13) + 7/(6*x^15)
end

function trigamma_x_lt_4(x::T) where T<: Real
  n = floor(Int64,5-x)
  z = x+n 
  val = trigamma_x_gr_4(z)
  for i in 1:n
    z -= 1
    val += 1/z^2
  end
  val 
end

function trigamma(x::T) where T<: Real
   x >= 4 ? trigamma_x_gr_4(x) : trigamma_x_lt_4(x)
end


function var_weighted(x::AbstractArray{Float64,1}, w::AbstractArray{Float64,1} )
  #println("# size(x) = ",size(x), " size(w) = ", size(w)); flush(stdout)
  @assert(length(x)==length(w) )
  sumw = sum(w)
  @assert( sumw > 0. )
  if(sumw!= 1.0)
     w /= sum(w)
     sumw = 1.0
  end
  sumw2 = sum(w.*w)
  xbar =  sum(x.*w)
  covar = sum((x.-xbar).*(x.-xbar) .* w) * sumw/(sumw*sumw-sumw2)
end

function make_proposal_dist_multidim_beta(theta::AbstractArray{Float64,2}, weights::AbstractArray{Float64,1},  tau_factor::Float64; verbose::Bool = false)
  
  function mom_alpha(x_bar::T, v_bar::T) where T<: Real 
    x_bar * (((x_bar * (1 - x_bar)) / v_bar) - 1)
  end
  function mom_beta(x_bar::T, v_bar::T) where T<: Real 
    (1 - x_bar) * (((x_bar * (1 - x_bar)) / v_bar) - 1)
  end
  # For algorithm, see https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=2613&context=etd 
  function fit_beta_mle(x::AbstractArray{T,1}; tol::T = 1e-6, max_it::Int64 = 10, init_guess::AbstractArray{T,1} = Array{T}(undef,0), w::AbstractArray{T,1} = Array{T}(undef,0), verbose::Bool = false ) where T<: Real
    lnxbar =   length(w)>1 ? Statistics.mean(log.(x),AnalyticWeights(w)) : Statistics.mean(log.(x))
    ln1mxbar = length(w)>1 ? Statistics.mean(log.(1.0.-x),AnalyticWeights(w)) : Statistics.mean(log.(1.0.-x))

    function itterate( mle_guess::Vector{T} ) where T<:Real
       (alpha, beta) = (mle_guess[1], mle_guess[2])
       dgab = digamma(alpha+beta)
       g1 = dgab - digamma(alpha) + lnxbar
       g2 = dgab - digamma(beta) + ln1mxbar
       tgab = trigamma(alpha+beta)
       G = [dgab-trigamma(alpha) tgab; tgab tgab-trigamma(beta)]
       mle_guess -= G \ [g1, g2]
    end 
  
    local mle_new 
    if length(init_guess) != 2
       xbar = length(w)>1 ? Statistics.mean(x,AnalyticWeights(w)) : Statistics.mean(x)
       vbar = length(w)>1 ? Statistics.varm(x,xbar,AnalyticWeights(w)) : Statistics.varm(x,xbar)
       mle_new = (vbar < xbar*(1.0-xbar)) ? [mom_alpha(xbar, vbar), mom_beta(xbar,vbar)] : ones(T,2)
    else
       mle_new = init_guess
    end
    if verbose
       println("it = 0: ", mle_new)
    end
    if any(mle_new.<=zero(T))
       println("# Warning: mean= ", xbar, " var= ",var," (alpha,beta)_init= ",mle_new," invalid, reinitializing to (1,1)")
       verbose = true
       mle_new = ones(T,2)
    end
    for i in 1:max_it
       mle_old = mle_new
       mle_new = itterate( mle_old )
       epsilon = max(abs.(mle_old.-mle_new))
       if verbose
          println("# it = ", i, ": ", mle_new, " max(Delta alpha, Delta beta)= ", epsilon)
       end
       if epsilon < tol
          break
       end
    end
    return mle_new
  end
  function make_beta(x::AbstractArray{T,1}, w::AbstractArray{T,1}; 
              mean::T = Statistics.mean(x,AnalyticWeights(w)), 
              var::T = Statistics.varm(x,xbar,AnalyticWeights(w)), tau_factor::T=one(T) ) where T<:Real
       alpha_beta = (var < mean*(1.0-mean)) ? [mom_alpha(mean, var), mom_beta(mean,var)] : ones(T,2)
       if any(alpha_beta.<=zero(T))
          alpha_beta = fit_beta_mle(x, w=w, init_guess=alpha_beta, verbose=true)
       end
       if any(alpha_beta.<=zero(T))
          alpha_beta = ones(T,2)
       else 
          if minimum(alpha_beta)>1.5*tau_factor && sum(alpha_beta)>=20.0*tau_factor
             alpha_beta ./= tau_factor
          end
       end
       Beta(alpha_beta[1], alpha_beta[2])
  end
  function make_beta_transformed(x::AbstractArray{T,1}, w::AbstractArray{T,1}; xmin::T=zero(T), xmax::T=one(T),
              mean::T = Statistics.mean(x,AnalyticWeights(w)), 
              var::T = Statistics.varm(x,xbar,AnalyticWeights(w)), tau_factor::T=one(T) ) where T<:Real
       alpha_beta = (var < mean*(1.0-mean)) ? [mom_alpha(mean, var), mom_beta(mean,var)] : ones(T,2)
       if any(alpha_beta.<=zero(T))
          alpha_beta = fit_beta_mle(x, w=w, init_guess=alpha_beta, verbose=true)
       end
       if any(alpha_beta.<=zero(T))
          alpha_beta = ones(T,2)
       else 
          if minimum(alpha_beta)>1.5*tau_factor && sum(alpha_beta)>=20.0*tau_factor
             alpha_beta ./= tau_factor
          end
       end
       LinearTransformedBeta(alpha_beta[1], alpha_beta[2], xmin=xmin, xmax=xmax)
  end
 
  theta_mean =  sum(theta.*weights',dims=2) # weighted mean for parameters
  theta_var = ABC.var_weighted(theta'.-theta_mean',weights)  # scaled, weighted covar for parameters
  tau_factor_indiv = fill(tau_factor,length(theta_var))
  if verbose
     println("total: ",theta_mean[1]," ",theta_var[1])
  end
  for i in 2:size(theta,1)
    mean_ratio = sum(theta[1,:].*theta[i,:].*weights) /(theta_mean[1]*theta_mean[i]) # weighted mean for parameters
    var_ratio = var_weighted(vec(theta[1,:].*theta[i,:]).-(theta_mean[1]*theta_mean[i]),weights)/(2 * theta_mean[1] * theta_var[i]) # scaled, weighted covar for parameters
    if verbose
       println("i=",i,": ",theta_mean[i]," ",theta_var[i]," ratios: ",mean_ratio, " ",var_ratio)
    end
    var_ratio  = var_ratio  >= one(var_ratio)  ? var_ratio  : one(var_ratio)
    tau_factor_indiv[i] = tau_factor*var_ratio
  end
  if verbose
     flush(stdout)
  end
  #=
  println("mean= ",theta_mean)
  println("var= ",theta_var)
  for i in 1:length(theta_mean)
     println("a= ",alpha(theta_mean[i],tau_factor*theta_var[i]), "  b= ",beta(theta_mean[i],tau_factor*theta_var[i]))
  end
  =#
  
  dist = ApproximateBayesianComputing.CompositeDistributions.CompositeDist( vcat(
         make_beta_transformed(theta[1,:], weights, xmin=0.0, xmax=max_rate, mean=theta_mean[1]/max_rate, var=theta_var[1]/max_rate^2, tau_factor=tau_factor_indiv[1]), 
         ContinuousDistribution[ make_beta(theta[i,:], weights, mean=theta_mean[i], var=theta_var[i], tau_factor=tau_factor_indiv[i]) for i in 2:size(theta,1) ] ))

end

function make_proposal_dist_multidim_beta(pop::abc_population_type, tau_factor::Float64; verbose::Bool = false)
	make_proposal_dist_multidim_beta(pop.theta, pop.weights, tau_factor, verbose=verbose)
end



# Tell ABC what it needs to know for a simulation
abc_plan = abc_pmc_plan_type(gen_data,calc_mean_per_bin,calc_dist_l1, param_prior; is_valid=is_valid,normalize=normalize_theta!,make_proposal_dist=make_proposal_dist_multidim_beta,epsilon_reduction_factor=0.501,tau_factor=1.1,target_epsilon=0.00001*nbins,num_max_attempt=1000);

# Generate "true/observed data" and summary statistics
data_true = abc_plan.gen_data(theta_true)
ss_true = abc_plan.calc_summary_stats(data_true)
#println("theta= ",theta_true," ss= ",ss_true, " d= ", 0.)

# Run ABC simulation
@time pop_out = run_abc(abc_plan,ss_true;verbose=true);

