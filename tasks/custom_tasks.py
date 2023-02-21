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
"""This module maintains custom task definitions and registrations."""

import functools
import logging
import seqio
import t5.data
from t5.data import postprocessors
from t5.data import preprocessors
from t5.evaluation import metrics
import tensorflow as tf

from typing import Dict
from typing import Any
from typing import Optional

TaskRegistry = seqio.TaskRegistry


_S_PROMPT = '[S2S]'
_X_PROMPT = '[NLG]'
_R_PROMPT = '[NLU]'


DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}


TaskRegistry.add(
    "cnn_dailymail_custom",
    source=seqio.TfdsDataSource(tfds_name="cnn_dailymail:3.4.0"),
    preprocessors=[
        functools.partial(
            preprocessors.summarize,
            article_key="article",
            summary_key="highlights"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.rouge],
    output_features=DEFAULT_OUTPUT_FEATURES)


TaskRegistry.add(
    "xsum",
    source=seqio.TfdsDataSource(tfds_name="xsum:1.1.0"),
    preprocessors=[
        functools.partial(
            preprocessors.summarize,
            article_key="document",
            summary_key="summary"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.rouge],
    output_features=DEFAULT_OUTPUT_FEATURES)


def prepend_prompt(dataset: tf.data.Dataset,
                   output_features: seqio.preprocessors.OutputFeaturesType,
                   sequence_length: Optional[
                       seqio.preprocessors.SequenceLengthType] = None,
                   prompt_mode: str = "",
                   key: str = "inputs",
                   mode: str = "") -> tf.data.Dataset:
    """Prepends a prompt at the beginning of an input sequence."""
    del sequence_length
    if prompt_mode and mode:
        logging.info("Add prompt")
        prompt_tokens = output_features[key].vocabulary.encode_tf(prompt_mode)
        logging.info(prompt_tokens)
        logging.info(dataset)

        def add_to_inputs(x):
            x[key] = tf.concat([prompt_tokens, x[key]], axis=0)
            return x

        dataset = dataset.map(add_to_inputs)
    return dataset


for task_name, prompt_mode in zip(['xsum_s_prompt', 'xsum_x_prompt', 'xsum_r_prompt'],
                             [_S_PROMPT, _X_PROMPT, _R_PROMPT]):
    
    TaskRegistry.add(
        task_name,
        source=seqio.TfdsDataSource(tfds_name="xsum:1.1.0"),
        preprocessors=[
            functools.partial(
                preprocessors.summarize,
                article_key="document",
                summary_key="summary"),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            functools.partial(
                prepend_prompt,
                prompt_mode=prompt_mode,
                mode="_prompt"),
            seqio.preprocessors.append_eos_after_trim,
        ],
        metric_fns=[metrics.rouge],
        output_features=DEFAULT_OUTPUT_FEATURES)