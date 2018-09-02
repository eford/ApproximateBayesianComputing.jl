# Linear Transformed Beta distribution

module TransformedBetaDistributions


if VERSION >= v"0.7"
  using Statistics
  using Distributed
  import Statistics: mean, median, maximum, minimum, quantile, std, var, cov, cor
else
  using Compat
  using Compat.Statistics
  using Compat.Distributed
  import Base: mean, median, maximum, minimum, quantile, std, var, cov, cor
end

import Base.length, Base.show
using Distributions 
import Distributions.params, Distributions.@check_args
import Distributions.rand 
import Distributions.pdf, Distributions.logpdf, Distributions.cdf, Distributions.gradlogpdf
import Distributions.quantile, Distributions.insupport
import Distributions.minimum, Distributions.maximum
import Distributions.mean, Distributions.var 
import Distributions.mode, Distributions.modes 

export LinearTransformedBeta
export params, rand, pdf, logpdf, cdf, gradlogpdf, quantile, insupport
export minimum, maximum, mean, var, mode, modes


struct LinearTransformedBeta{T<:Real} <: ContinuousUnivariateDistribution 
    dist::Beta{T}
    xmin::T
    xmax::T

    function LinearTransformedBeta{T}(α::T, β::T; xmin::T = zero(T), xmax::T = one(T) ) where T
        @check_args(LinearTransformedBeta, α > zero(α) && β > zero(β))
        @check_args(LinearTransformedBeta, -Inf < xmin < xmax < Inf )
        new{T}(Beta(α, β),xmin,xmax)
    end
end
LinearTransformedBeta(α::T, β::T; xmin::T=zero(T), xmax::T = one(T) ) where T = LinearTransformedBeta{T}(α, β, xmin=xmin, xmax=xmax)


#### Conversions
function convert(::Type{LinearTransformedBeta{T}}, α::Real, β::Real, xmin::Real, xmax::Real) where T<:Real
    LinearTransformedBeta(T(α), T(β), T(xmin), T(xmax) )
end
function convert(::Type{LinearTransformedBeta{T}}, d::LinearTransformedBeta{S}) where {T <: Real, S <: Real}
    LinearTransformedBeta(T(d.α), T(d.β), T(d.xmin), T(d.xmax) )
end

### Parameters
params(d::LinearTransformedBeta) = (d.dist.α, d.dist.β,d.xmin,d.xmax)
@inline partype(d::LinearTransformedBeta{T}) where {T<:Real} = T


### Sampling
rand(d::LinearTransformedBeta) = d.xmin+(d.xmax-d.xmin)*rand(d.dist)
#sampler

### Evaluation 
pdf(d::LinearTransformedBeta, x::T) where T<:Real = pdf(d.dist, (x.-d.xmin)./(d.xmax-d.xmin) )
logpdf(d::LinearTransformedBeta, x::T) where T<:Real = logpdf(d.dist, (x.-d.xmin)./(d.xmax-d.xmin) )
cdf(d::LinearTransformedBeta, q::T) where T<:Real = cdf(d.dist, (x.-d.xmin)./(d.xmax-d.xmin) )
quantile(d::LinearTransformedBeta, q::T) where T<:Real = quantile(d.dist, (x.-d.xmin)./(d.xmax-d.xmin) )
minimum(d::LinearTransformedBeta) = d.xmin
maximum(d::LinearTransformedBeta) = d.xmax

function insupport(d::LinearTransformedBeta, x::T)  where T<:Real
  d.xmin<d.xmax && insupport(d.dist, (x.-d.xmin)./(d.xmax-d.xmin) )
end


function gradlogpdf(d::LinearTransformedBeta, x::T) where T<:Real
  (d.xmax-d.xmin)*gradlogpdf(d.dist, (x-d.xmin)/(d.xmax-d.xmin) )
end


### Basic statistics
mean(d::LinearTransformedBeta) = d.xmin+(d.xmax-d.xmin)*mean(d.dist)
var(d::LinearTransformedBeta) = (d.xmax-d.xmin)^2*var(d.dist)
mode(d::LinearTransformedBeta) = d.xmin+(d.xmax-d.xmin)*mode(d.dist)
modes(d::LinearTransformedBeta) = d.xmin.+(d.xmax-d.xmin).*modes(d.dist)
# TODO skewness
# TODO kurtosis
# TODO entropy
# TODO mgf
# TODO cf


### Show

distrname(d::LinearTransformedBeta) = "LinearTransformedBeta"
function Base.show(io::IO, d::LinearTransformedBeta)
  show(io,distrname(d) * "(α=" * string(d.dist.α) * ", β=" * string(d.dist.β) * ", min=" * string(d.xmin) * " ,max=" * string(d.xmax) * ")" )
end

end # module
