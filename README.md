# Developing NLP solutions with T5X and Vertex AI

This repository compiles prescriptive guidance and code samples that show how to
operationalize the Google Research T5X framework using Google Cloud Vertex AI.
Using T5X with Vertex AI enables streamlined experimentation, development, and
deployment of natural language processing (NLP) solutions at scale.

The guidance assumes that you're familiar with ML concepts such as large
language models (LLMs), and that you're generally familiar with Google Cloud
features like Cloud Storage, Cloud TPUs, and Google Vertex AI. 

## Introduction

T5X is a machine learning (ML) framework for developing high-performance
sequence models, including large language models (LLMs). For more information
about T5X, see the following resources:

-  [Scaling Up Models and Data with `t5x` and `seqio`](https://arxiv.org/abs/2203.17189)
-  [T5X Github repo](https://github.com/google-research/t5x)

T5X is built as a [JAX](https://jax.readthedocs.io/en/latest/index.html)-based
library for training, evaluating, and inferring with sequence models. T5X's
primary focus is on Transformer type language models. You can use T5X to
pretrain language models and to fine-tune a pretrained language model. The T5X
GitHub repo includes
[references](https://github.com/google-research/t5x/blob/main/docs/models.md) to
a large number of pretrained Transformer models, including the T5 and Switch
Transformer families of models.

T5X is streamlined, modular, and composable. You can implement pretraining,
fine-tuning, evaluating, and inferring by configuring reusable components that
are provided by T5X rather than having to develop custom Python modules. 

[Vertex AI](https://cloud.google.com/vertex-ai) is Google Cloud's unified ML
platform that's designed to help data scientists and ML engineers increase their
velocity of experimentation, deploy faster, and manage models with confidence.

## Repository structure


```
.
├── configs
├── docs
├── examples
├── notebooks
├── tasks 
├── Dockerfile
└── README.md
```

- [`/notebooks`](https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai/tree/main/notebooks): Example notebooks demonstrating T5X fine-tuning, evaluating, and inferring scenarios:
     - [Getting Started](https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai/blob/main/notebooks/getting-started.ipynb) 
     - [Fine-tuning T5 1.1 XL on XSum](https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai/blob/main/notebooks/finetune-t511-xl-xsum.ipynb)
     - [Fine-tuning T5 1.1 Large SQuAD](https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai/blob/main/notebooks/finetune-t511-large-squad.ipynb)
     - [Evaluating T5 1.1 XL on XSum](https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai/blob/main/notebooks/eval-t511-xl-xsum.ipynb)

-  [`/configs`](https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai/tree/main/configs): Configuration files for the scenarios demonstrated in notebooks.

- [`/tasks`](https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai/tree/main/tasks): Python modules implementing custom SeqIO Tasks.  
- [`/docs`](https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai/tree/main/docs) - Technical guides compiling best practices for running T5X on Vertex AI:
    - [Running and monitoring T5X jobs with Vertex AI](https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai/tree/main/docs/run-t5x-jobs.md)
    - [Implementing model and data parallelizm](https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai/tree/main/docs/partitioning.md)
- The [main folder](https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai) also includes Dockerfiles for custom container images used by Vertex Training.  

## Environment setup 

This section outlines the steps to configure the Google Cloud environment that is required in order to run the code samples in this repo.
 

![arch](/images/arch.png)



* You use a user-managed instance of Vertex AI Workbench  as your development environment and the primary interface to Vertex AI services.
* You run T5X training, evaluating, and inferring tasks as Vertex Training custom jobs using a custom training container image.
* You use Vertex AI Experiments and Vertex AI Tensorboard for job monitoring and experiment tracking.
* You use a regional Cloud Storage bucket to manage artifacts created by T5X jobs.

To set up the environment execute the following steps.

### Select a Google Cloud project

In the Google Cloud Console, on the project selector page, [select or create a Google Cloud project](https://console.cloud.google.com/projectselector2/home/dashboard?_ga=2.77230869.1295546877.1635788229-285875547.1607983197&_gac=1.82770276.1635972813.Cj0KCQjw5oiMBhDtARIsAJi0qk2ZfY-XhuwG8p2raIfWLnuYahsUElT08GH1-tZa28e230L3XSfYewYaAlEMEALw_wcB). You need to be a project owner in order to set up the environment.

### Enable the required services

From [Cloud Shell](https://cloud.google.com/shell/docs/using-cloud-shelld.google.com/shell/docs/using-cloud-shell), run the following commands to enable the required Cloud APIs:

```bash
export PROJECT_ID=<YOUR_PROJECT_ID>
 
gcloud config set project $PROJECT_ID
 
gcloud services enable \
  cloudbuild.googleapis.com \
  compute.googleapis.com \
  cloudresourcemanager.googleapis.com \
  iam.googleapis.com \
  container.googleapis.com \
  cloudapis.googleapis.com \
  cloudtrace.googleapis.com \
  containerregistry.googleapis.com \
  iamcredentials.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com \
  notebooks.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com
```

**Note**: When you work with Vertex AI user-managed notebooks, be sure that all the services that you're using are provisioned in the same project and the same compute region as the available Vertex AI TPU pods regions. For a list of regions where TPU pods are available, see [Locations](https://cloud.google.com/vertex-ai/docs/general/locations#accelerators) in the Vertex AI documentation.

### Verify quota to run jobs using Vertex AI TPUs

Some notebooks demonstrate scenarios that require as many as 128 TPU cores. 
 
If you need an increase in Vertex AI TPU quota values, follow these steps:

1. In the Cloud Console, navigate to the **Quotas** tab of the [Vertex AI API page](https://console.cloud.google.com/apis/api/aiplatform.googleapis.com/quotas).
2. In the **Enter property name or value box** that's next to the **Filter** label, add a filter that has the following conditions:
* **Quota: Custom model training TPU V2 cores per region** or **Custom model training TPU V3 cores per region** 
* **Dimensions (e.g. location):  Region:** <YOUR_REGION>

**Note**: Vertex AI TPUs are not available in all regions. If the **Limit** value in the listing is 8, TPUs are available, and you can request more by increasing the **Quota** value. If the **Limit** value is 0, no TPUs are available, and the **Quota** value cannot be changed.
 
3. In the listing, select the quota that matches your filter criteria and then click **Edit Quotas**. 
 
4. In the **New limit** box, enter the required value and then submit the quota change request.

Quota increases don’t directly impact your billing because you are still required to specify the number of TPU cores to submit your T5X tasks. Only the tasks submitted with a high number of TPU cores result in higher billing.

### Configure Vertex AI Workbench

You can create a user-managed notebooks instance from the command line.
 
**Note**: Make sure that you're following these steps in the same project as before.
 
In Cloud Shell, enter the following command. For `<YOUR_INSTANCE_NAME>`, enter a name starting with a lower-case letter followed by lower-case letters, numbers or dash sign. For `<YOUR_LOCATION>`, add a zone (for example, `us-central1-a` or `europe-west4-a`).

```bash
PROJECT_ID=$(gcloud config list --format 'value(core.project)')
INSTANCE_NAME=<YOUR_INSTANCE_NAME>
LOCATION=<YOUR_LOCATION>
gcloud notebooks instances create $INSTANCE_NAME \
     --vm-image-project=deeplearning-platform-release \
     --vm-image-family=common-cpu-notebooks \
     --machine-type=n1-standard-4 \
     --location=$LOCATION
```

Vertex AI Workbench creates a user-managed notebooks instance based on the properties that you specified and then automatically starts the instance. When the instance is ready to use, Vertex AI Workbench activates an **Open JupyterLab** link next to the instance name in the [Vertex AI Workbench Cloud Console](https://console.cloud.google.com/vertex-ai/workbench/list/instances) page. To connect to your user-managed notebooks instance, click **Open JupyterLab**.

### Clone the repo, install dependencies, and build the base container image

After the Vertex Workbench user-managed notebook Jupyter lab is launched, perform the following steps:

1. On the Launcher page, start a new terminal session by clicking the Terminal icon.
2. Clone the repository to your notebook instance:

```bash
git clone https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai.git
```
3. Install code dependencies:
 
```bash
cd t5x-on-vertex-ai
pip install -U pip
pip install google-cloud-aiplatform[tensorboard] tfds-nightly t5[gcp]
```

4. Build the base T5X container image in Container Registry. For `<YOUR_PROJECT_ID>`, use the ID of the Google project that you are working with. 

```bash
export PROJECT_ID=<YOUR_PROJECT_ID>
gcloud config set project $PROJECT_ID
 
IMAGE_NAME=t5x-base
IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_NAME}
gcloud builds submit --timeout "2h" --tag ${IMAGE_URI} . --machine-type=e2-highcpu-8
```

### Create a staging Cloud Storage bucket

The notebooks in the repo require access to a Cloud Storage bucket that's used for staging and for managing ML artifacts created by the jobs submitted. The bucket must be in the same Google Cloud region as the region you will use to run Vertex AI custom jobs.
 
* In the Jupyter lab terminal, create the bucket. For `<YOUR_REGION>`, specify the region. For `<YOUR_BUCKET_NAME>`, use a [globally unique name](https://cloud.google.com/storage/docs/naming-buckets). 

```bash
REGION=<YOUR_REGION>
BUCKET_NAME=<YOUR_BUCKET_NAME>
gsutil mb -l $REGION -p $PROJECT_ID gs://$BUCKET_NAME
```

### Create a Vertex AI Tensorboard instance

In the Jupyter lab Terminal, create the Vertex AI Tensorboard instance:
 
```bash
DISPLAY_NAME=<YOUR_INSTANCE_NAME>
gcloud ai tensorboards create --display-name $DISPLAY_NAME --project $PROJECT_ID --region=$REGION
```

### Prepare the datasets

Before you walk through the example notebooks, make sure that you pre-build all the required [TensorFlow Datasets](https://www.tensorflow.org/datasets) (TFDS) datasets.

From the Jupyter lab Terminal:

```bash
BUCKET_NAME=<YOUR_BUCKET_NAME>
export TFDS_DATA_DIR=gs://${BUCKET_NAME}/datasets
```

#### [squad](https://www.tensorflow.org/datasets/catalog/squad)
 
```
tfds build --data_dir $TFDS_DATA_DIR --experimental_latest_version squad
```

#### [wmt_t2t_translate](https://www.tensorflow.org/datasets/catalog/wmt_t2t_translate) 

```
tfds build --data_dir $TFDS_DATA_DIR --experimental_latest_version wmt_t2t_translate
```

#### [cnn_dailymail](https://www.tensorflow.org/datasets/catalog/cnn_dailymail)
 
```
tfds build --data_dir $TFDS_DATA_DIR --experimental_latest_version cnn_dailymail
```

#### [xsum](https://www.tensorflow.org/datasets/catalog/xsum)

To build **xsum** you need to download and prepare the source data manually. 

1. Follow the [instructions](https://github.com/EdinburghNLP/XSum/blob/master/XSum-Dataset/README.md#running-the-download-and-extraction-scripts) to create the `xsum-extracts-from-downloads` folder with source data. 
2. Create a tar archive from the `xsum-extracts-from-downloads` folder.
```
tar -czvf xsum-extracts-from-downloads.tar.gz xsum-extracts-from-downloads/
```
3. Copy the archive to the TFDS manual downloads folder.
```
gsutil cp -r xsum-extracts-from-downloads.tar.gz ${TFDS_DATA_DIR}/downloads/manual/
```
4. Build the dataset
```
tfds build --data_dir $TFDS_DATA_DIR --experimental_latest_version xsum
```

The environment is ready.

## Getting started

Start by reading the [Running and monitoring T5X jobs with Vertex AI](https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai/tree/main/docs/run-t5x-jobs.md) guide and walking through the [Getting Started](https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai/blob/master/notebooks/getting-started.ipynb) notebook.

## Getting help

If you have any questions or if you found any problems with this repository, please report through GitHub issues.

