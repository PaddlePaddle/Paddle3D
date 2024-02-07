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

import inspect
from collections.abc import Sequence
from typing import Callable, Iterable, Union

from pprndr.utils.logger import logger

__all__ = [
    "DATASETS", "TRANSFORMS", "MODELS", "ENCODERS", "RAYSAMPLERS", "LAYERS",
    "FIELDS", "RENDERERS", "LR_SCHEDULERS", "METRICS", "WEIGHT_INITIALIZERS"
]


class ComponentManager:
    """Implement a manager class to add the new component properly.
    The component can be added as either class or function type.

    Args:
        name (str): The name of component.
        description (str): Description of Component Manager
    Returns:
        A callable object of ComponentManager.

    Examples 1:
        from pprndr.apis.manager import ComponentManager
        model_manager = ComponentManager()
        class AlexNet: ...
        class ResNet: ...
        model_manager.add_component(AlexNet)
        model_manager.add_component(ResNet)
        # Or pass a sequence alliteratively:
        model_manager.add_component([AlexNet, ResNet])
        print(model_manager.components_dict)
        # {'AlexNet': <class '__main__.AlexNet'>, 'ResNet': <class '__main__.ResNet'>}

    Examples 2:
        # Or an easier way, using it as a Python decorator, while just add it above the class declaration.
        from pprndr.apis.manager import ComponentManager
        model_manager = ComponentManager()
        @model_manager.add_component
        class AlexNet: ...
        @model_manager.add_component
        class ResNet: ...
        print(model_manager.components_dict)
        # {'AlexNet': <class '__main__.AlexNet'>, 'ResNet': <class '__main__.ResNet'>}
    """

    def __init__(self, *, name: str, description: str = ''):
        self._components_dict = dict()
        self._name = name
        self._description = description

    def __len__(self):
        return len(self._components_dict)

    def __repr__(self):
        name_str = self._name if self._name else self.__class__.__name__
        return "{}:{}".format(name_str, list(self._components_dict.keys()))

    def __getitem__(self, item: str):
        if item not in self._components_dict.keys():
            raise KeyError("{} does not exist in availabel {}".format(
                item, self))
        return self._components_dict[item]

    @property
    def components_dict(self) -> dict:
        return self._components_dict

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def _add_single_component(self, component: Callable):
        """
        Add a single component into the corresponding manager.
        Args:
            component (function|class): A new component.
        Raises:
            TypeError: When `component` is neither class nor function.
            KeyError: When `component` was added already.
        """

        # Currently only support class or function type
        if not (inspect.isclass(component) or inspect.isfunction(component)):
            raise TypeError(
                "Expect class/function type, but received {}".format(
                    type(component)))

        # Obtain the internal name of the component
        component_name = component.__name__

        # Check whether the component was added already
        if component_name in self._components_dict.keys():
            logger.warning(
                "{} exists already! It is now updated to {} !!!".format(
                    component_name, component))
            self._components_dict[component_name] = component

        else:
            # Take the internal name of the component as its key
            self._components_dict[component_name] = component

    def add_component(self, components: Union[Callable, Iterable[Callable]]
                      ) -> Union[Callable, Iterable[Callable]]:
        """
        Add component(s) into the corresponding manager.
        Args:
            components (function|class|list|tuple): Support four types of components.
        Returns:
            components (function|class|list|tuple): Same with input components.
        """

        # Check whether the type is a sequence
        if isinstance(components, Sequence):
            for component in components:
                self._add_single_component(component)
        else:
            component = components
            self._add_single_component(component)

        return components


DATASETS = ComponentManager(name="datasets")
TRANSFORMS = ComponentManager(name="transforms")
MODELS = ComponentManager(name="models")
FIELDS = ComponentManager(name="fields")
RAYSAMPLERS = ComponentManager(name="ray_samplers")
ENCODERS = ComponentManager(name="encoders")
LAYERS = ComponentManager(name="layers")
RENDERERS = ComponentManager(name="renderers")
LR_SCHEDULERS = ComponentManager(name="lr_schedulers")
METRICS = ComponentManager(name="metrics")
WEIGHT_INITIALIZERS = ComponentManager(name="weight_initializers")
