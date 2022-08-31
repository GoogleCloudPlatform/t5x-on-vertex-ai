## Develop custom SeqIO tasks

For many standard NLP use cases, you can use one of the
[predefined SeqIO Tasks](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/tasks.py).
However, if you want to use your own dataset, you need to create a custom SeqIO
Task. For details about how to create a custom SeqIO Task, see the [SeqIO
repo](https://github.com/google/seqio).  

This repo contains a few
[examples of custom SeqIO Tasks](https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai/blob/master/examples/custom_tasks.py),
including the Task for the [XSum
Dataset](https://www.tensorflow.org/datasets/catalog/xsum).
