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

from typing import List, Union

__all__ = ["get_color"]

color_dict = {
    "white": [1.0, 1.0, 1.0],
    "black": [0.0, 0.0, 0.0],
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
}


def get_color(color: Union[str, list, tuple]) -> List:
    """
    Args:
        color (Union[str, list]): Color as a string or a rgb list

    Returns:
        TensorType[3]: Parsed color
    """
    if isinstance(color, str):
        color = color.lower()
        if color not in color_dict:
            raise ValueError(
                "{} is not a valid preset color. please use one of {}".format(
                    color, color_dict.keys()))
        return color_dict[color]
    elif isinstance(color, list):
        if len(color) != 3:
            raise ValueError(
                "Color should be 3 values (RGB) instead got {}".format(color))
        return color
