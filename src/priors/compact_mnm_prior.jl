"""
    compact_mnm_hyper(α::AbstractArray{Float32,1})

[Dirichlet Distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution)
"""
struct compact_mnm_hyper <: distribution_hyper_params
    α::AbstractArray{Float32,1}
end

mutable struct compact_mnm_sufficient_statistics <: sufficient_statistics
    N::Float32
    points_sum::AbstractArray{Float32,1}
end


function calc_posterior(prior:: compact_mnm_hyper, suff_statistics::Vector{Tuple{sufficient_statistics,Number}})
    if length(suff_statistics) == 0
        return prior
    end
    points_num = sum([post_kernel(x[2],global_time)*x[1].N for x in suff_statistics])
    if points_num == 0
        return prior
    end

    suff_sum = sum([post_kernel(x[2],global_time)*x[1].points_sum for x in suff_statistics],dims=1)[1]
    return compact_mnm_hyper(prior.α + suff_sum)
end

function sample_distribution(hyperparams::compact_mnm_hyper)
    return compact_mnm_dist(log.(rand(Dirichlet(Float64.(hyperparams.α)))))
end

function create_sufficient_statistics(hyper::compact_mnm_hyper,posterior::compact_mnm_hyper,points::AbstractArray{Float32,2}, pts_to_group = 0)
    pts = copy(points)
    points_sum = sum(pts, dims = 2)[:]
    return compact_mnm_sufficient_statistics(size(points,2),points_sum)
end

function log_marginal_likelihood(hyper::compact_mnm_hyper, posterior_hyper::compact_mnm_hyper, suff_stats::Vector{Tuple{sufficient_statistics,Number}})
    D = length(suff_stats[1][1].points_sum)
    logpi = log(pi)
    val = logabsgamma(sum(hyper.α))[1] -logabsgamma(sum(posterior_hyper.α))[1] + sum((x-> logabsgamma(x)[1]).(posterior_hyper.α) - (x-> logabsgamma(x)[1]).(hyper.α))
    return val
end

function aggregate_suff_stats(suff_l::compact_mnm_sufficient_statistics, suff_r::compact_mnm_sufficient_statistics)
    return compact_mnm_sufficient_statistics(suff_l.N+suff_r.N, suff_l.points_sum + suff_r.points_sum)
end
