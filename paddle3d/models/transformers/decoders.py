# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# ------------------------------------------------------------------------
# Modified from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/decoder.py
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import copy

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.models.transformers.transformer import inverse_sigmoid


@manager.TRANSFORMER_DECODERS.add_component
class DetectionTransformerDecoder(nn.Layer):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 transformerlayers=None,
                 num_layers=None,
                 return_intermediate=False):
        super(DetectionTransformerDecoder, self).__init__()
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers
        self.num_layers = num_layers
        self.layers = nn.LayerList()
        for i in range(num_layers):
            layer_name = transformerlayers[i].pop('type_name')
            decoder_layer = manager.TRANSFORMER_DECODER_LAYERS.components_dict[
                layer_name]
            params = transformerlayers[i]
            self.layers.append(decoder_layer(**params))

        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                key,
                value,
                query_pos,
                reference_points,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []

        # np.save("d_query.npy", query.numpy())
        # np.save("d_value.npy", kwargs['value'].numpy())
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(
                [2])  # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                key,
                value,
                query_pos,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.transpose([1, 0, 2])
            # np.save("d_output_{}.npy".format(lid), output.numpy())

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                new_reference_points = paddle.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                    reference_points[..., :2])
                new_reference_points[..., 2:
                                     3] = tmp[..., 4:5] + inverse_sigmoid(
                                         reference_points[..., 2:3])

                reference_points = F.sigmoid(new_reference_points).detach()
                # np.save("d_new_reference_points_{}.npy".format(lid), reference_points.numpy())

            output = output.transpose([1, 0, 2])
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return paddle.stack(intermediate), paddle.stack(
                intermediate_reference_points)

        return output, reference_points
