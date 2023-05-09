# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Union
from urllib.parse import unquote, urlparse

import filelock
import paddle

from paddle3d.env import PRETRAINED_HOME, TMP_HOME
from paddle3d.utils.download import download_with_progress
from paddle3d.utils.logger import logger
from paddle3d.utils.xarfile import unarchive_with_progress


def load_pretrained_model_from_url(model: paddle.nn.Layer,
                                   url: str,
                                   overwrite: bool = False,
                                   verbose: bool = True):
    """
    """
    pretrained_model = unquote(url)
    savename = pretrained_model.split('/')[-1]
    savedir = os.path.join(PRETRAINED_HOME, savename.split('.')[0])

    os.makedirs(savedir, exist_ok=True)
    savepath = os.path.join(savedir, savename)

    if os.path.exists(savepath) and not overwrite:
        logger.warning(
            "There is a file with the same name locally, we directly load the local file"
        )
    else:
        if overwrite:
            logger.warning(
                "There is a file with the same name locally, we will delete the file."
            )
            os.remove(savepath)

        # Add file lock to prevent multi-process download
        with filelock.FileLock(os.path.join(TMP_HOME, savename)):
            if not os.path.exists(savepath):
                with logger.progressbar(
                        "download pretrained model from {}".format(url)) as bar:
                    for _, ds, ts in download_with_progress(url, savedir):
                        bar.update(float(ds) / ts)

        #TODO: unzip the file if it is a compress one

    load_pretrained_model_from_path(model, savepath, verbose=verbose)


def load_pretrained_model_from_path(model: paddle.nn.Layer,
                                    path: str,
                                    verbose: bool = True):
    """
    """
    para_state_dict = paddle.load(path)
    load_pretrained_model_from_state_dict(
        model, para_state_dict, verbose=verbose)


def load_pretrained_model_from_state_dict(model: paddle.nn.Layer,
                                          state_dict: dict,
                                          verbose: bool = True):
    """
    """
    model_state_dict = model.state_dict()
    keys = model_state_dict.keys()
    num_params_loaded = 0
    for k in keys:
        if k not in state_dict:
            if verbose:
                logger.warning("{} is not in pretrained model".format(k))
        elif list(state_dict[k].shape) != list(model_state_dict[k].shape):
            if verbose:
                logger.warning(
                    "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                    .format(k, state_dict[k].shape, model_state_dict[k].shape))
        else:
            model_state_dict[k] = state_dict[k]
            num_params_loaded += 1

    model.set_dict(model_state_dict)
    logger.info("There are {}/{} variables loaded into {}.".format(
        num_params_loaded, len(model_state_dict), model.__class__.__name__))


def load_pretrained_model(model: paddle.nn.Layer,
                          pretrained_model: Union[dict, str],
                          verbose: bool = True):
    """
    """
    if isinstance(pretrained_model, dict):
        load_pretrained_model_from_state_dict(
            model, pretrained_model, verbose=verbose)
    elif isinstance(pretrained_model, str):
        if urlparse(pretrained_model).netloc:
            load_pretrained_model_from_url(
                model, pretrained_model, verbose=verbose)
        elif os.path.exists(pretrained_model):
            load_pretrained_model_from_path(
                model, pretrained_model, verbose=verbose)
        else:
            raise ValueError(
                '{} is neither a valid path nor a valid URL.'.format(
                    pretrained_model))
    else:
        raise TypeError('Unsupported pretrained model type {}'.format(
            type(pretrained_model)))
