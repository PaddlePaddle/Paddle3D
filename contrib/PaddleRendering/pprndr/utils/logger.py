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

import contextlib
import functools
import logging
import sys
import threading
import time
from typing import Iterable

import colorlog

__all__ = ["logger"]

loggers = {}

log_config = {
    'DEBUG': {
        'level': 10,
        'color': 'purple'
    },
    'INFO': {
        'level': 20,
        'color': 'cyan'
    },
    'WARNING': {
        'level': 30,
        'color': 'yellow'
    },
    'ERROR': {
        'level': 40,
        'color': 'red'
    },
    'CRITICAL': {
        'level': 50,
        'color': 'bold_red'
    }
}


class Logger(object):
    """Default logger

    Args:
        name(str) : Logger name, default is 'PaddleNGP'
    """

    def __init__(self, name: str = None):
        name = 'PaddleNGP' if not name else name
        self.logger = logging.getLogger(name)

        for key, conf in log_config.items():
            logging.addLevelName(conf['level'], key)
            self.__dict__[key] = functools.partial(self.__call__, conf['level'])
            self.__dict__[key.lower()] = functools.partial(
                self.__call__, conf['level'])

        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.format)

        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        self._enabled = True

    @property
    def format(self):
        if sys.stdout.isatty():
            color_format = '%(log_color)s%(asctime)-15s%(reset)s - %(levelname)8s - %(message)s'
            log_colors = {
                key: conf['color']
                for key, conf in log_config.items()
            }
            return colorlog.ColoredFormatter(
                color_format, log_colors=log_colors)

        normal_format = '%(asctime)-15s - %(levelname)8s - %(message)s'
        return logging.Formatter(normal_format)

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def __call__(self, log_level: str, msg: str):
        if not self.enabled:
            return

        self.logger.log(log_level, msg)

    @contextlib.contextmanager
    def use_terminator(self, terminator: str):
        old_terminator = self.handler.terminator
        self.handler.terminator = terminator
        yield
        self.handler.terminator = old_terminator

    @contextlib.contextmanager
    def processing(self, msg: str, flush_interval: float = 0.1):
        """
        Continuously print a progress bar with rotating special effects.
        Args:
            msg(str): Message to be printed.
            flush_interval(float): Rotation interval. Default to 0.1.
        """
        end = False

        def _printer():
            index = 0
            flags = ['\\', '|', '/', '-']
            while not end:
                flag = flags[index % len(flags)]
                with self.use_terminator('\r'):
                    self.info('{}: {}'.format(msg, flag))
                time.sleep(flush_interval)
                index += 1

            self.info('{}'.format(msg))

        if sys.stdout.isatty():
            t = threading.Thread(target=_printer)
            t.daemon = True
            t.start()
        else:
            self.info('{}'.format(msg))

        yield
        end = True

    @contextlib.contextmanager
    def progressbar(self, msg: str, flush_interval: float = 0.1):
        self.info(msg)
        bar = ProgressBar(logger=self, flush_interval=flush_interval)
        yield bar
        bar._end = True
        bar.update(1)

    def range(self, stop: int, msg: str):
        with self.progressbar(msg) as bar:
            for idx in range(stop):
                bar.update(float(idx) / stop)
                yield idx

    def enumerate(self, iterable: Iterable, msg: str):
        totalnum = len(iterable)

        with self.progressbar(msg) as bar:
            for idx, item in enumerate(iterable):
                bar.update(float(idx) / totalnum)
                yield idx, item


class ProgressBar(object):
    """
    Progress bar printer
    Args:
        flush_interval(float): Flush rate of progress bar, default is 0.1.
    Examples:
        .. code-block:: python
            with ProgressBar('Download module') as bar:
                for i in range(100):
                    bar.update(i / 100)
            # with continuous bar.update, the progress bar in the terminal
            # will continue to update until 100%
            #
            # Download module
            # [##################################################] 100.00%
    """

    def __init__(self, logger: Logger, flush_interval: float = 0.1):
        self.logger = logger
        self.last_flush_time = time.time()
        self.flush_interval = flush_interval
        self._end = False

    def update(self, progress: float):
        """
        Update progress bar
        Args:
            progress: Processing progress, from 0.0 to 1.0
        """
        msg = '[{:<50}] {:.2f}%'.format('#' * int(progress * 50),
                                        progress * 100)
        need_flush = (time.time() - self.last_flush_time) >= self.flush_interval

        if (need_flush and sys.stdout.isatty()) or self._end:
            with self.logger.use_terminator('\r'):
                self.logger.info(msg)
            self.last_flush_time = time.time()

        if self._end:
            self.logger.info('')


logger = Logger()
