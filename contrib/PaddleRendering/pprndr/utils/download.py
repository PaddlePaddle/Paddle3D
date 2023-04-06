#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License")
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
from typing import Generator
from urllib.parse import urlparse

import requests


def download(url: str, path: str = None) -> str:
    '''Download a file

    Args:
        url (str) : url to be downloaded
        path (str, optional) : path to store downloaded products, default is current work directory

    Examples:
        .. code-block:: python
            url = 'https://xxxxx.xx/xx.tar.gz'
            download(url, path='./output')
    '''
    for savename, _, _ in download_with_progress(url, path):
        ...
    return savename


def download_with_progress(url: str,
                           path: str = None) -> Generator[str, int, int]:
    '''Download a file and return the downloading progress -> Generator[filename, download_size, total_size]

    Args:
        url (str) : url to be downloaded
        path (str, optional) : path to store downloaded products, default is current work directory

    Examples:
        .. code-block:: python
            url = 'https://xxxxx.xx/xx.tar.gz'
            for filename, download_size, total_szie in download_with_progress(url, path='./output'):
                print(filename, download_size, total_size)
    '''
    path = os.getcwd() if not path else path
    if not os.path.exists(path):
        os.makedirs(path)

    parse_result = urlparse(url)
    savename = parse_result.path.split('/')[-1]
    savename = os.path.join(path, savename)

    res = requests.get(url, stream=True)
    download_size = 0
    total_size = int(res.headers.get('content-length'))
    with open(savename, 'wb') as _file:
        for data in res.iter_content(chunk_size=4096):
            _file.write(data)
            download_size += len(data)
            yield savename, download_size, total_size
