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

FROM python:3.9 

RUN apt-get install apt-transport-https ca-certificates gnupg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-cli -y

RUN pip install -U numpy sentencepiece tensorflow==2.8.1 rouge_score t5

WORKDIR /
RUN git clone --branch=main https://github.com/google/flaxformer
WORKDIR flaxformer
RUN pip install '.[testing]'

WORKDIR /
RUN git clone --branch=main https://github.com/google-research/t5x
WORKDIR t5x
RUN pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

WORKDIR /
ADD tasks ./tasks

ENV PYTHONPATH=/tasks
ENTRYPOINT ["python", "./t5x/t5x/main.py"]
