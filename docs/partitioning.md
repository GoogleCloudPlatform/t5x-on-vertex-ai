# Parallelize T5X jobs

One of the biggest challenges of working with large ML models is scaling model
training, which you do by parallelizing computation on large clusters of ML
accelerators. The two primary methods for scaling large models are model
parallelism and data parallelism.  

A large model might not fit on a single accelerator device. To scale the model
beyond a single device, you have to distribute the model parameters and model
computations across multiple devices. This is called _model parallelism_.  

In data parallelism, you create multiple replicas of a parallelized model and
give each replica a different set of examples to compute over.  

T5X supports both model and data parallelism. For a detailed description of how
model and data parallelism are implemented, see
[Data, Model, and Activation Partitioning](https://github.com/google-research/t5x/blob/main/docs/usage/partitioning.md).  

This section summarizes key configurations for parallelizing T5X jobs.  

In T5X, you achieve model and data parallelism through partitioning model
parameters, activations, and data across a logical mesh of accelerator devices.
T5X partitioning uses the [jax.pjit
API](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html) that in turn
builds on the XLA[ SPMD partitioner](https://arxiv.org/pdf/2105.04663.pdf).  

The fundamental partitioning concept is a logical mesh. The logical mesh
represents a 2-dimensional arrangement of devices that are required in order to
support a given model and data parallelization. In T5X, the two axes of a
logical mesh are called `data` and `model`.  

For example, if the computation requires 4-way model parallelism and you want to
run it on a Cloud TPU slice with 32 cores, you can arrange the cores into a 8x4
logical mesh. In this case, the `data` dimension is 8 and the `model` dimension
is 4, as shown in the following diagram.  

![8x4 mesh](/images/8x4mesh.png)



The logical mesh abstracts a physical device mesh, and therefore the layout of
the logical mesh might be different from that of the physical mesh. For example,
the following diagram shows the physical topology of a Cloud TPU v3-32 slice.  

![v3-32 slice](/images/v3-32-slice.png)



A v3-32 slice includes four TPU boards, each with four TPU chips. TPU chips are
connected into a 2-D mesh network using a high-speed interconnect. Each TPU chip
has two TPU cores.  

An abstraction of this physical mesh into a 8x4 logical mesh could have the
mapping of physical cores to a logical mesh that's shown in the following
diagram.  

![16x2 mapping](/images/16x2mapping.png)


In T5X, you define the mapping of a physical TPU topology to a logical mesh by
using the `PjitPartitioner` class. As with many T5X functions and classes, you
can configure `PjitPartitoner` in your Gin file. The constructor for the
`PjitPartioner` class takes three arguments:

-  `model_parallel_submesh`
-  `num_partitions`
-  `logical_axis_rules`

The `model_parallel_submesh` and `num_partitions` arguments provide two ways to
specify the mapping of a physical topology to a logical mesh. The
`num_partitions` argument specifies the number of cores that you want to
distribute your model over. When you use `num_partitions`, T5X automatically
selects the mapping.

The `model_parallel_submesh` argument provides the most control over the
mapping. To understand how to use the argument, it's useful to begin with a
formal description of the topology of a TPU slice. A TPU slice is described by a
4-tuple `(x,y,z,c)`, where `x`, `y`, and `z` are dimensions of a toroidal mesh
network, and `c` is the number of cores on a TPU chip. For TPU v2 and TPU v3
slices, `z` is always 1 because TPU v2 and v3 slices are connected in a 2-D
mesh. For example, a TPU v3-32 slice has a `(4,4,1,2)` topology. For TPU v4, `z`
can be greater than 1 because TPU v4 features a 3-D interconnect.  

The `model_parallel_submesh` argument is also a `(x,y,z,c)` 4-tuple. It
specifies the shape of a submesh (or a tile) in the physical device mesh that
will be allocated to a single model-parallel replica. The number of model
parallel submeshes that fit in a physical mesh of the slice defines the level of
data parallelism.  

For example, if `model_parallel_submesh` is set to `(1,1,1,2)`, each
model-parallel replica will be allocated 2 cores on a single TPU chip—an
instance of 2-way model parallelism. On a `(4,4,1,2)` slice, you will have 16
model-parallel replicas or 16-way data parallelism. The shape of a logical
device mesh that results from setting `model_parallel_submesh` to `(1,1,1,2)` on
a `(4,4,1,2)` slice is 16x2. 

If `model_parallel_submesh` is set to `(2,1,1,2)` or `(1,2,1,2)`, each model
parallel replica will be allocated 4 cores and the shape of a logical mesh will
be 8x4.   

The third argument (`logical_axis_rules`) controls how input data, parameters,
and activations are distributed (or sharded) across a logical device mesh.

The `logical_axis_rules` argument gives you fine-grained control over sharding,
but it is complex to configure. Instead of setting `logical_axis_rules`
directly, you can use the `standard_logical_axis_rules` function that generates
canonical rule sets based on a couple of high level parameters:
`parameter_partitioning_dims` and `activation_partioning_dims`.  

The following discussion focuses on how to use the `standard_logical_axis_rules
`function. If you want to understand how to configure `logical_axis_rules`
directly, see
[Data, Model, and Activation Partitioning](https://github.com/google-research/t5x/blob/main/docs/usage/partitioning.md).  

To begin, it's useful to have a definition of the terms 1-D partitioning and 2-D
partitioning. By default, the batch dimension of input data and activations
(only input data and activations have batch dimension) is sharded over the
`data` axis of a logical mesh, and the parameters are sharded over the `model`
axis. This is called _1-D partitioning_. You can also shard parameters or
activations over both the `data` and the `model` axes of a logical mesh. This is
called _2-D partitioning_.  

The `parameter_partitioning_dims` and `activation_partioning_dims` parameters
control 2-D partitioning of parameters and activations, respectively. When  set
to 1 (the default), 1-D partitioning is used. When set to 2, 2-D partitioning is
used.  

Using 2-D partitioning of parameters lets you fit larger models on a given TPU
slice. Enabling 2-D partitioning of activations accommodates larger batches.
