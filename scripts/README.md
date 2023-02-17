
# Job configuration templates

This document provides Vertex AI Training T5X job configuration templates for selected fine-tuning, evaluating or inferring scenarios.

The `run.py` script in the `<REPO FOLDER>/scripts` encapsulates job configuration and submission. Execute the script from the `<REPO FOLDER>/scripts`  


## Fine-tuning 20B UL2 on XSUM

```
PROJECT_ID=<YOUR PROJECT ID>
REGION=<YOUR REGION>
IMAGE_URI=<YOUR IMAGE_URI>
STAGING_BUCKET=<YOUR STAGING BUCKET>
TFDS_DATA_DIR=<YOUR TFDS DATA DIR>

python run.py \
--project_id=$PROJECT_ID \
--region=$REGION \
--image_uri=$IMAGE_URI \
--staging_bucket=$STAGING_BUCKET \
--tfds_data_dir=$TFDS_DATA_DIR \
--gin_files=../configs/finetune_ul2_xsum.gin,../configs/ul220b_public.gin \
--gin_search_paths=/flaxformer \
--gin_overwrites=USE_CACHED_TASKS=False,TRAIN_STEPS=2_700_000,INITIAL_CHECKPOINT_PATH=\"gs://scenic-bucket/ul2/ul220b/checkpoint_2650000\" \
--accelerator_type=TPU_V3 \
--accelerator_count=128 \
--run_mode=train 



# Tensorboard


export TENSORBOARD_NAME=projects/895222332033/locations/us-central1/tensorboards/2937103421045473280
export REGION=us-central1
export EXPERIMENT_NAME=ul2-xsum-10
export LOG_DIR=gs://jk-staging-europe-west4/t5x_jobs/t5x_job_20221112161252

tb-gcp-uploader --tensorboard_resource_name $TENSORBOARD_NAME \
--logdir $LOG_DIR \
--experiment_name $EXPERIMENT_NAME