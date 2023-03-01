# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp
import inspect
import functools
import pickle

# TODO: Set cache directory in a global config module
CACHE_DIR = osp.abspath(osp.join('.cache', 'paddle_uapi'))

if not osp.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def get_cache_dir(*args, **kwargs):
    # `args` and `kwargs` reserved for extension
    return CACHE_DIR


def create_yaml_config_file(tag, noclobber=True):
    cache_dir = get_cache_dir()
    file_path = osp.join(cache_dir, f"{tag}.yml")
    if noclobber and osp.exists(file_path):
        raise FileExistsError
    # Overwrite content
    with open(file_path, 'w') as f:
        f.write("")
    return file_path


def persist(arg_save_dir=None, filename='.cache.pkl'):
    def _deco(func):
        if arg_save_dir is None:
            arg_name = None
        else:
            if not isinstance(arg_save_dir, (int, str)):
                raise TypeError
            sig = inspect.signature(func)
            if isinstance(arg_save_dir, int):
                if arg_save_dir >= len(sig.parameters):
                    raise ValueError
                arg_name = list(sig.parameters.keys())[arg_save_dir]
            else:
                # isinstance(arg_save_dir, str)
                if arg_save_dir not in sig.parameters.keys():
                    raise ValueError
                arg_name = arg_save_dir

        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            if arg_name is None:
                save_dir = CACHE_DIR
            else:
                bnd_args = sig.bind(*args, **kwargs)
                if arg_name not in bnd_args.arguments:
                    raise RuntimeError(f"`{arg_name}` not specified.")
                save_dir = bnd_args.arguments[arg_name]
            cache_file_path = osp.join(save_dir, filename)
            if osp.exists(cache_file_path):
                with open(cache_file_path, 'rb') as f:
                    ret = pickle.load(f)
            else:
                ret = func(*args, **kwargs)
                if ret:
                    with open(cache_file_path, 'wb') as f:
                        pickle.dump(ret, f)
            return ret

        return _wrapper

    return _deco
