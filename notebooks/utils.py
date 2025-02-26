# Copyright 2022 Google LLC
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
#

from typing import List
from typing import Dict
from typing import Any
from typing import Union
from typing import Optional

import gcsfs
import fsspec
from IPython import get_ipython
import sys
import os
import pandas as pd
import time
import subprocess
import tempfile

import tensorflow as tf
from t5.evaluation import eval_utils

from google.cloud import aiplatform as vertex_ai
from google.cloud.aiplatform import Artifact
from google.cloud.aiplatform import Execution
from google.cloud.aiplatform import Model
from google.cloud.aiplatform import CustomJob

import google.auth
from google.auth import credentials as auth_credentials
from google.api_core.client_info import ClientInfo

source_credentials, _ = google.auth.default()
request = google.auth.transport.requests.Request()
source_credentials.refresh(request)
source_credentials.apply(headers = {'user-agent': 'cloud-solutions/t5x-on-vertex-ai-v1.0'})
source_credentials.refresh(request)

# set python path from the current environment
PYTHON_BIN = os.path.join(sys.exec_prefix, 'bin', 'python3')
pd.set_option('display.max_colwidth', None)

def create_t5x_custom_job(
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
            gcs_fs.put(gin_file, gcs_path)
            gcs_gin_files.append(gcs_path.replace('gs://', '/gcs/'))
    except:
        raise RuntimeError('Could not copy gin files to GCS.')

    # Temporary mitigation to address issues with t5x/main.py
    # and inference on tfrecord files
    if run_mode == 'infer':
        args = [
            f'--gin.MODEL_DIR="{model_dir}"',
            f'--tfds_data_dir={tfds_data_dir}',
        ]
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
        
        
    container_spec = {
        "image_uri": image_uri,
        "args": args
    }
    
    # Temporary mitigation to address issues with t5x/main.py
    # and inference on tfrecord files
    if run_mode == 'infer':
        container_spec['command'] = ["python", "./t5x/t5x/infer.py"]
    
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


def _create_artifacts(
    tfds_data_dir: str,
    gin_config_path: str,
    model_architecture_path: str,
    model_dir: str,
    execution_name: str,
    job_display_name: str,
    custom_job: CustomJob,
    run_mode: str
):
    """Creates and logs artifacts generated by the Vertex AI custom
    T5X job to an experiment run."""

    # FSSpec to interact with GCS
    gcs_fs = gcsfs.GCSFileSystem()

    # Training dataset artifact
    try:
        dataset_seqio_artifact = vertex_ai.Artifact.create(
            schema_title='system.Dataset',
            display_name=f'{run_mode}_dataset', 
            uri=tfds_data_dir
        )
    except Exception as e:
        print(e)
        print('Dataset URI not logged to metadata.')

    try:
        # Gin configuration file
        with gcs_fs.open(gin_config_path) as fp:
            gin_config_artifact = vertex_ai.Artifact.create(
                schema_title='system.Artifact',
                display_name=f'{run_mode}_gin_config', 
                uri=gin_config_path
            )
    except Exception as e:
        print(e)
        print('Gin config file not logged to metadata.')

    try:
        # Model information
        with gcs_fs.open(model_architecture_path) as fp:
            model_architecture_artifact = vertex_ai.Artifact.create(
                schema_title='system.Artifact',
                display_name='model_architecture', 
                uri=model_architecture_path
            )
    except Exception as e:
        print(e)
        print('Model information/architecture not logged to metadata.')

    try:
        # Trained model
        trained_model_artifact = vertex_ai.Artifact.create(
            schema_title='system.Model',
            display_name='trained_model', 
            uri=model_dir
        )
    except Exception as e:
        print(e)
        print('Trained model URI not logged to metadata.')

    try:
        # Vertex AI training job and model lineage
        with vertex_ai.start_execution(
            schema_title='system.CustomJobExecution',
            display_name=execution_name,
            metadata={
                'job_display_name': job_display_name,
                'job_spec': str(custom_job.job_spec),
                'resource_name': custom_job.resource_name
            }
        ) as execution:
            execution.assign_input_artifacts([
                    dataset_seqio_artifact,
                    gin_config_artifact,
                    model_architecture_artifact
            ])

            execution.assign_output_artifacts([
                    trained_model_artifact
            ])

            vertex_ai.log_params({
                "lineage": execution.get_output_artifacts()[0].lineage_console_uri
            })
    except Exception as e:
        print(e)
        print('Model lineage not logged to metadata.')


def get_all_experiment_run_directories(experiment_name):
    """ Fetch run ids and run directories for a given experiment
    """
    try:
        # Get list of all experiment runs for a given experiment
        runs = vertex_ai.ExperimentRun.list(experiment=experiment_name)
        # Get run id and run directory for each run
        run_details = []
        for run in runs:
            # Create run instance
            run = vertex_ai.ExperimentRun(experiment=experiment_name, run_name=run.name)
            # Fetch artifacts for the run
            run_artifacts = run.get_artifacts()
            run_dir = None
            if len(run_artifacts) > 0:
                # fetch run directory
                run_dir = [artifact.uri for artifact in run.get_artifacts() 
                           if artifact.display_name=='trained_model'][0]
            run_details.append({'RUN_ID':run.name, 'RUN_DIR':run_dir})

        sorted_items = sorted(run_details, key=lambda r: r['RUN_ID'])

        # Return results
        df = pd.DataFrame(run_details)
        return df
    except Exception as e:
        print(e)
        print(f"Cannot fetch runs and run directory for experiment={experiment_name}")
        print("Please check the experiment name.")


def submit_and_track_t5x_vertex_job(
    custom_job: vertex_ai.CustomJob,
    job_display_name: str,
    run_name: str,
    experiment_name: str,
    execution_name: str,
    model_dir: str,
    vertex_ai: vertex_ai,
    run_mode: str,
    tfds_data_dir: str = ''
):
    """Submits a custom T5X Vertex AI training job and
    tracks the execution and metadata of it. 

    Args:
        <arg_name> <type> < = default>:
            Required|Optional. <description>
            Example:

        custom_job (aiplatform.CustomJob):
            Required. Instance of aiplatform.CustomJob with all the configurations
            to run a custom T5X Vertex AI training job. This custom_job 
            can be generated with the function `create_t5x_custom_job`
        job_display_name (str):
            Required. The user-defined name of the aiplatform.CustomJob.
        run_name (str):
            Required. The user-defined name of the Vertex AI experiment run.
        experiment_name (str):
            Required. The user-defined name of the Vertex AI experiment.
        execution_name (str):
            Required. The user-defined name of the execution to be used as
            a reference for metadata tracking.
        model_dir (str):
            Required. Path on Google Cloud Storage to store all the artifacts 
            generated by the custom T5X training job. 
            The path must be in this format: gs://{bucket name}/{your folder}/.
            Example:
                gs://my_bucket/experiments/model1/
        vertex_ai (aiplatform):
            Required. Instance of google.cloud.aiplatform.
        run_mode (str):
            Required. The mode to run T5X under. Options: `train`, `eval`, `infer`.
        tfds_data_dir (str = ''):
            Optional. If set, this directory will be used to store datasets 
            prepared by TensorFlow Datasets that are not available in the 
            public TFDS GCS bucket. Note that this flag overrides the 
            `tfds_data_dir` attribute of all Task's.
    """
    # Define paths where configuration files will be generated
    gin_config_path = os.path.join(model_dir, 'config.gin')
    model_architecture_path = os.path.join(model_dir, 'model-info.txt')

    # FSSpec to interact with GCS
    gcs_fs = gcsfs.GCSFileSystem()

    try:
        # List all the runs in the experiment
        exp_runs = vertex_ai.ExperimentRun.list(
            experiment=experiment_name)

        # If run_name, than stop the execution
        if run_name in [run.name for run in exp_runs]:
            print('Execution stopped.')
            print('Run name already exists in this Experiment.')
            print('Please provide a different and unique run name.')
            return
    except Exception as e:
        print(e)
        print(f'Experiment with name {experiment_name} not found.')
        print(f'Please provite a valid experiment name.')
        return

    # Start Vertex AI custom training job
    custom_job.run(sync=False)
    custom_job.wait_for_resource_creation()

    # Wait for Vertex AI training job to start
    while custom_job.state.value == 2:
        print('Job still pending. Waiting additional 15 seconds.')
        time.sleep(15)

    # Job failed
    if custom_job.state.value == 5:
        print('Execution stopped. Vertex AI training Job has failed.' \
                'Check the Vertex AI logs for additional information.')
        return
    
    # Job running
    if custom_job.state.value == 3:
        print('Vertex AI training job has started.')
        
        if run_mode == 'train':
            print('Waiting for config files to be generated.')

            while not (
                gcs_fs.exists(gin_config_path) and 
                gcs_fs.exists(model_architecture_path)
            ):
                if custom_job.state.value == 5:
                    print('Execution stopped. Vertex AI training Job has failed.' \
                            'Check the Vertex AI logs for additional information.')
                    return

                time.sleep(10)
                print('Waiting for config files to be generated.')
                print('Waiting additional 10 seconds.')            

    vertex_ai.start_run(run_name)

    print('Creating artifacts.')
    # Generate metadata artifacts
    _create_artifacts(
        tfds_data_dir=tfds_data_dir,
        gin_config_path=gin_config_path,
        model_architecture_path=model_architecture_path,
        model_dir=model_dir,
        execution_name=execution_name,
        job_display_name=job_display_name,
        custom_job=custom_job,
        run_mode=run_mode)

    print('Artifacts were created. Training job is still running.')
    vertex_ai.end_run()


def _parse_metrics_from_tb_events(summary_dir, out_file):
    """ Parse TensorBoard events files and log results to csv
    
    Adapted from 
    https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/scripts/parse_tb.py
    """
    try:
        # Reading event directories
        subdirs = tf.io.gfile.listdir(summary_dir)
        summary_dirs = [os.path.join(summary_dir, d.rstrip('/')) for d in subdirs]

        # Parsing event files
        scores = None
        for d in summary_dirs:
            events = eval_utils.parse_events_files(d, True)
            task_metrics = eval_utils.get_eval_metric_values(
                events,
                task_name=os.path.basename(d))
            if scores:
                scores.update(task_metrics)
            else:
                scores = task_metrics

        if not scores:
            print(f"No evaluation events found in {summary_dir}")

        # Computing and formatting metrics
        df = eval_utils.scores_to_df(scores)
        df = eval_utils.compute_avg_glue(df)
        df = eval_utils.sort_columns(df)

        # Writing metrics to output file
        eval_utils.log_csv(df, output_file=out_file)
    except Exception as e:
        print(e)
        print('Failed to parse TensorBoard event files')
    

def parse_and_log_eval_metrics(
    summary_dir: str,
    run_name: str,
    vertex_ai: vertex_ai
) -> pd.DataFrame:
    """Parses the evaluation metrics of a trained model.

    Args:
        summary_dir (str):
            Required. Full path to a local filesystem folder with the files with 
            the evaluation metrics generated during training.
            Example:
                `/inference_eval`
        out_file (str):
            Required. Full path to the file that will be generated in this function.
            Example: 
                `/folder1/results.csv`
        run_name (str):
            Required. The user-defined name of the Vertex AI experiment run.
        vertex_ai (aiplatform):
            Required. Instance of google.cloud.aiplatform.
    
    Returns:
        (pandas.DataFrame):
            Returns the parsed results (evaluation metrics) from the 
            trained model. 
    """

    try:
        # create temporary file to stage results
        out_file = tempfile.NamedTemporaryFile()

        # Parse evaluation metrics
        _parse_metrics_from_tb_events(summary_dir, out_file.name)
        
        # Log metrics to Vertex AI Experiments
        results = pd.read_csv(out_file.name, sep=',')

        with vertex_ai.start_run(run_name, resume=True) as my_run:
            metrics = {}
            for k, v in results[-2:].drop('step', axis=1).to_dict().items():
                metrics[k + ': max, step'] = ', '.join([str(v) for v in v.values()])
            my_run.log_metrics(metrics)
        
        # Remove output file after parsing
        out_file.close()
    except Exception as e:
        print(e)
        print('Metrics were not logged to metadata.')
        return
    else:
        return results


def create_tensorboard(
    instance_name: str,
    region: str,
    project_id: str
) -> str:
    """Creates a managed Vertex AI tensorboard instance.

    Args:
        instance_name (str):
            Required. The use-defined name of the managed Vertex AI
            Tensorboard instance.
        region (str):
            Required. Region where the managed Vertex AI Tensorboard instance
            will be created.
        project_id (str):
            Required. Project ID where the managed Vertex AI Tensorboard instance
            will be created.
    Returns:
        (str):
            Returns the full ID of the managed Vertex AI Tensorboad.
    """

    process = subprocess.run(
        [
            'gcloud',
            'ai',
            'tensorboards',
            'create',
            f'--display-name={instance_name}',
            f'--region={region}',
            f'--project={project_id}'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    return process.stderr.split(sep=':')[-1].strip().replace('.', '')
