# DPMMSubClustersStreaming.jl
This is the code repository for the *Julia* package (with an optional [Python wrapper](https://github.com/BGU-CS-VIL/dpmmpythonStreaming)) that corresponds to our paper, [Sampling in Dirichlet Process Mixture Models for Clustering Streaming Data](), AISTATS 2022.

## Requirements
This package was developed and tested on *Julia 1.0.3*, prior versions will not work.
The following dependencies are required:
- Distributed
- DistributedArrays
- Distributions
- JLD2
- LinearAlgebra
- NPZ
- Random
- SpecialFunctions
- StatsBase


## Installation

Use Julia's package manager:
`(v1.5) pkg> add DPMMSubClustersStreaming`

## Usage

This package can be used either as Multi-Threaded (recommended for Gaussian) or Multi-Process (recommended for Multinomials).

If you opt for the multi-process version:

It is recommended to use `BLAS.set_num_threads(1)`, when working with larger datasets increasing the amount of workers will do the trick, `BLAS` multi threading might disturb the multiprocessing, resulting in slower inference.

For all the workers to recognize the package, you must start with `@everywhere using DPMMSubClustersStreaming`. If you require to set the seed (using the `seed` kwarg), add `@everywhere using Random` as well.

### Quick Start
In order to run in the basic mode, use the function on the first batch:
```
model = fit(all_data::AbstractArray{Float32,2},local_hyper_params::distribution_hyper_params,α_param::Float32;
        iters::Int64 = 300, init_clusters::Int64 = 1,seed = nothing, verbose = true, save_model = false, burnout = 20, gt = nothing,epsilon = 0.00001,kernel_func = RBFKernel() )
```

* all_data - The data, should be `DxN`.
* local_hyper_params - The prior you plan to use, can be either Multinomial, or `NIW` (example below on how to create one)
* α_param - Concetration parameter
* iters - Number of iterations
* seed - Random seed, can also be set seperatly. note that if seting seperatly you must set it on all workers.
* verbose - Printing status on every iteration.
* save_model - If true, will save a checkpoint every 25 iterations, note that if you opt for saving, I recommend the advanced mode.
* burnout - How many iteration before allowing clusters to split/merge, reducing this number will result in faster inference, but with higher variance between the different runs.
* gt - Ground Truth, if supplied will perform `NMI` and `VI` tests on every iteration.
* epsilon - batches with smaller weight than this will be droped.
* kernel_func - distance function to weight the batches


After the model initialize on the first batch, you can continue to the next batches:

```
run_model_streaming(model,iters, cur_time, new_data=nothing)
```
Where the new data will be fed to the last argument, the `cur_time` argument is used for batch weighting (e.g. the further it is from the previous batch, the less weight previous batches will have).

At any point, the lastest label assignment can be retrieved via `get_labels(model)`, and the model can be used to predict new samples via `predict(model,data)`.


### Example

```
    x,labels,clusters = generate_gaussian_data(10^6,3,10,100.0)
    parts = 10000
    xs = [x[:,i:parts:end] for i=1:parts]
    labelss = [labels[i:parts:end] for i=1:parts]
    hyper_params = DPMMSubClustersStreaming.niw_hyperparams(Float32(1.0),
               zeros(Float32,3),
               Float32(5),
               Matrix{Float32}(I, 3, 3)*1)
    dp = dp_parallel_streaming(xs[1],hyper_params,Float32(10000000.0), 1,1,nothing,true,false,15,labelss[1],0.0001)
    labels = get_labels(dp)
    avg_nmi = mutualinfo(Int.(labelss[1]),labels,normed=true)
    for i=2:parts
        run_model_streaming(dp,1,i*0.5,xs[i])
        labels = get_labels(dp)
        avg_nmi += mutualinfo(Int.(labelss[i]),labels,normed=true)
    end
    println("NMI: ",avg_nmi/parts)    
```

In the above example, we initially generate `10^6` points, samples from 10 3D Gaussians.
We then split it to 10000 parts, create the model hyper params (Normal Inverse Wishart) and start clustering the first batch.
We then iterate over all the batches, feeding each 1 at time of `i*0.5`, and storing the NMI, which we later use to show the average NMI.

### Misc

For any questions: dinari@post.bgu.ac.il

Contributions, feature requests, suggestion etc.. are welcomed.

If you use this code for your work, please cite the following:

```
@inproceedings{dinari2022streaming,
  title={Sampling in Dirichlet Process Mixture Models for Clustering Streaming Data},
  author={Dinari, Or and  Freifeld, Oren},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  year={2022}
}
```
