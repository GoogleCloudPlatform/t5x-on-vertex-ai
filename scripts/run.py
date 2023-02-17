# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A utility to submit a Vertex Training T5X job."""

import gcsfs
import fsspec
import os

from absl import flags
from absl import app
from absl import logging
from datetime import datetime
from importlib import import_module

from google.cloud import aiplatform as vertex_ai
from google.cloud.aiplatform import CustomJob

from typing import List
from typing import Dict
from typing import Any
from typing import Union
from typing import Optional

MACHINE_TYPE = 'cloud-tpu'

flags.DEFINE_string('project_id', None, 'GCP Project')
flags.DEFINE_string('region', None, 'Vertex Pipelines region')
flags.DEFINE_string('staging_bucket', None, 'Staging bucket')
flags.DEFINE_string('training_sa', None, 'Training SA')
flags.DEFINE_string('image_uri', None, 'Training image')
flags.DEFINE_string('job_name_prefix', 't5x_job', 'Job name prefix')
flags.DEFINE_list('gin_files', None, 'Gin configuration files')
flags.DEFINE_list('gin_overwrites', None, 'Gin overwrites')
flags.DEFINE_list('gin_search_paths', None, 'Gin search paths')
flags.DEFINE_string('accelerator_type', 'TPU_V2', 'Accelerator type')
flags.DEFINE_integer('accelerator_count', 8, 'Number of cores')
flags.DEFINE_string('run_mode', 'train', 'Run mode')
flags.DEFINE_string('tfds_data_dir', None, 'TFDS data dir')
flags.DEFINE_bool('sync', True, 'Execute synchronously')
flags.mark_flag_as_required('project_id')
flags.mark_flag_as_required('region')
flags.mark_flag_as_required('staging_bucket')
flags.mark_flag_as_required('gin_files')
flags.mark_flag_as_required('image_uri')
FLAGS = flags.FLAGS


def _create_t5x_custom_job(
    display_name: str,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    image_uri: str,
    run_mode: str,
    gin_files: List[str],
    model_dir: str,
    gin_search_paths: Optional[List[str]] = None,
    tfds_data_dir: Optional[str] = None,
    replica_count: int = 1,
    gin_overwrites: Optional[List[str]] = None,
    base_output_dir: Optional[str] = None,
) -> CustomJob:
    """Creates a Vertex AI custom T5X training job.
    It copies the configuration files (.gin) to GCS, creates a worker_pool_spec 
    and returns an aiplatform.CustomJob.
    Args:
        display_name (str):
            Required. User defined display name for the Vertex AI custom T5X job.
        machine_type (str):
            Required. The type of machine for running the custom training job on
            dedicated resources. For TPUs, use `cloud-tpu`.
        accelerator_type (str):
            Required. The type of accelerator(s) that may be attached
            to the machine as per `accelerator_count`. Only used if
            `machine_type` is set. Options: `TPU_V2` or `TPU_V3`.
        accelerator_count (int):
            Required. The number of accelerators to attach to the `machine_type`. 
            Only used if `machine_type` is set. For TPUs, this is the number of
            cores to be provisioned.
            Example: 8, 128, 512, etc.
        image_uri (str):
            Required. Full image path to be used as the execution environment of the 
            custom T5X training job.
            Example:
                'gcr.io/{PROJECT_ID}/{IMAGE_NAME}'
        run_mode (str):
            Required. The mode to run T5X under. Options: `train`, `eval`, `infer`.
        gin_files (List[str]):
            Required. Full path to gin configuration file on local filesystem. 
            Multiple paths may be passed and will be imported in the given 
            order, with later configurations overriding earlier ones.
        gin_search_paths (List[str]):
            List of gin config path prefixes to be prepended to gin suffixes in gin includes and gin_files
        model_dir (str):
            Required. Path on Google Cloud Storage to store all the artifacts generated
            by the custom T5X training job. The path must be in this format:
            `gs://{bucket name}/{your folder}/...`.
            Example:
                gs://my_bucket/experiments/model1/
        tfds_data_dir (Optional[str] = None):
            Optional. If set, this directory will be used to store datasets prepared by 
            TensorFlow Datasets that are not available in the public TFDS GCS 
            bucket. Note that this flag overrides the `tfds_data_dir` attribute of 
            all Task`s. This path must be a valid GCS path.
            Example:
                gs://my_bucket/datasets/my_dataset/
        replica_count (int = 1):
            Optional. The number of worker replicas. If replica count = 1 then one chief
            replica will be provisioned. For TPUs this must be set to 1.
        gin_overwrites (Optional[List[str]] = None):
            Optional. List of arguments to overwrite gin configurations. Argument must be 
            enclosed in parentheses.
            Example:
                --gin.TRAIN_PATH=\"gs://my_bucket/folder\"
        base_output_dir (Optional[str] = None):
    Returns:
        (aiplatform.CustomJob):
            Return an instance of a Vertex AI training CustomJob.
    """

    local_fs = fsspec.filesystem('file')
    gcs_fs = gcsfs.GCSFileSystem()

    # Check if gin files exists
    if not gin_files or not all([local_fs.isfile(f) for f in gin_files]):
        raise FileNotFoundError(
            'Provide a list of valid gin files.'
        )

    # Try to copy files to GCS bucket
    try:
        gcs_gin_files = []
        for gin_file in gin_files:
            gcs_path = os.path.join(model_dir, gin_file.split(sep='/')[-1])
            print('********')
            print(gcs_path)
            print(gin_file)
            gcs_fs.put(gin_file, gcs_path)
            print('-------')
            gcs_gin_files.append(gcs_path.replace('gs://', '/gcs/'))
    except:
        raise RuntimeError('Could not copy gin files to GCS.')

    container_spec = {"image_uri": image_uri}
    # Temporary mitigation to address issues with t5x/main.py
    # and inference on tfrecord files
    if run_mode == 'infer':
        args = [
            f'--gin.MODEL_DIR="{model_dir}"',
            f'--tfds_data_dir={tfds_data_dir}',
        ]
        container_spec['command'] = ["python", "./t5x/t5x/infer.py"]
    else:
        args = [
            f'--run_mode={run_mode}',
            f'--gin.MODEL_DIR="{model_dir}"',
            f'--tfds_data_dir={tfds_data_dir}',
        ]

    if gin_search_paths:
        args.append(f'--gin_search_paths={",".join(gin_search_paths)}')

    args += [f'--gin_file={gcs_path}' for gcs_path in gcs_gin_files]
    
    if gin_overwrites:
        args += [f'--gin.{overwrite}' for overwrite in gin_overwrites]
        
    container_spec["args"] = args
    
    worker_pool_specs =  [
        {
            "machine_spec": {
                "machine_type": machine_type,
                "accelerator_type": accelerator_type,
                "accelerator_count": accelerator_count,
            },
            "replica_count": replica_count,
            "container_spec": container_spec,
        }
    ]
    
    job = vertex_ai.CustomJob(
        display_name=display_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=base_output_dir
    )
    
    return job


def _main(argv):

    vertex_ai.init(
        project=FLAGS.project_id,
        location=FLAGS.region,
        staging_bucket=FLAGS.staging_bucket,
    )

    job_name = f'{FLAGS.job_name_prefix}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
    job_dir = f'{FLAGS.staging_bucket}/t5x_jobs/{job_name}'

    job = _create_t5x_custom_job(
        display_name=job_name,
        machine_type=MACHINE_TYPE,
        accelerator_type=FLAGS.accelerator_type,
        accelerator_count=FLAGS.accelerator_count,
        image_uri=FLAGS.image_uri,
        run_mode=FLAGS.run_mode,
        gin_files=FLAGS.gin_files,
        gin_overwrites=FLAGS.gin_overwrites,
        gin_search_paths=FLAGS.gin_search_paths,
        tfds_data_dir=FLAGS.tfds_data_dir,
        model_dir=job_dir,
    ) 

    logging.info(f'Starting job: {job_name}')    
    job.run(sync=FLAGS.sync)

if __name__ == "__main__":
    app.run(_main)

