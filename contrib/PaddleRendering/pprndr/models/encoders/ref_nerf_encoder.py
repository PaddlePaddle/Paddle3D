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

import math
from typing import Union

import numpy as np
import paddle
import paddle.nn as nn
from paddle.autograd import PyLayer

from pprndr.apis import manager
from pprndr.cameras.math_functionals import Gaussians

__all__ = ["IntegDirEncoder"]


@manager.ENCODERS.add_component
class IntegDirEncoder(nn.Layer):
    """Generate integrated directional encoding (IDE) function.

    This function returns a function that computes the integrated directional
    encoding from Equations 6-8 of https://arxiv.org/abs/2112.03907.

    Args:
        deg_view: number of spherical harmonics degrees to use.

    Returns:
        A function for evaluating integrated directional encoding.

     Raises:
        ValueError: if deg_view is larger than 5.
    """

    def __init__(
            self,
            deg_view: int,
    ):
        super(IntegDirEncoder, self).__init__()

        self.deg_view = deg_view
        if self.deg_view > 5:
            raise ValueError(
                "Only deg_view of at most 5 is numerically stable.")

        self.ml_array = self._get_ml_array(self.deg_view)
        l_max = 2**(self.deg_view - 1)
        self.mat = np.zeros((l_max + 1, self.ml_array.shape[1]))
        for i, (m, l) in enumerate(self.ml_array.T):
            for k in range(l - m + 1):
                self.mat[k, i] = self._sh_coeff(l, m, k)

        self.mat = paddle.to_tensor(self.mat, dtype="float32")
        self.ml_array = paddle.to_tensor(self.ml_array, dtype="int32")

    def _sh_coeff(self, l, m, k):
        """Compute spherical harmonic coefficients."""

        return np.sqrt((2.0 * l + 1.0) * np.math.factorial(l - m) /
                       (4.0 * np.pi * np.math.factorial(l + m))
                       ) * self._assoc_legendre_coeff(l, m, k)

    def _assoc_legendre_coeff(self, l, m, k):
        """Compute associated Legendre polynomial coefficients.

        Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
        (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

        Args:
            l: associated Legendre polynomial degree.
            m: associated Legendre polynomial order.
            k: power of cos(theta).

        Returns:
            A float, the coefficient of the term corresponding to the inputs.
        """
        return ((-1)**m * 2**l * np.math.factorial(l) / np.math.factorial(k) /
                np.math.factorial(l - k - m) * self._generalized_binomial_coeff(
                    0.5 * (l + k + m - 1.0), l))

    def _generalized_binomial_coeff(self, a, k):
        """Compute generalized binomial coefficients."""
        return np.prod(a - np.arange(k)) / np.math.factorial(k)

    def _get_ml_array(self, deg_view):
        """Create a list with all pairs of (l, m) values to use in the encoding."""
        ml_list = []
        for i in range(deg_view):
            l = 2**i
            # Only use nonnegative m values, later splitting real and imaginary parts.
            for m in range(l + 1):
                ml_list.append((m, l))

        ml_array = np.array(ml_list).T
        return ml_array

    def forward(self, xyz: paddle.Tensor, kappa_inv: paddle.Tensor):
        """Function returning integrated directional encoding (IDE).

        Args:
            xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
            kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
                Mises-Fisher distribution.

        Returns:
            An array with the resulting IDE.
        """

        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]
        z = paddle.where(
            paddle.abs(z) > 1e-10, z, paddle.to_tensor([1e-10],
                                                       dtype='float32'))

        xy_complex = paddle.complex(x, y)
        vmz = paddle.concat([z**i for i in range(self.mat.shape[0])], axis=-1)
        vmxy = paddle.concat([
            complex_pow.apply(xy_complex, m.item()) for m in self.ml_array[0, :]
        ],
                             axis=-1)
        sh_xyz = vmxy * paddle.matmul(vmz, self.mat)
        sigma = 0.5 * self.ml_array[1, :] * (self.ml_array[1, :] + 1)

        ide = sh_xyz * paddle.exp(-sigma * kappa_inv)
        ret = paddle.concat([paddle.real(ide), paddle.imag(ide)], axis=-1)

        return ret


class complex_pow(PyLayer):
    @staticmethod
    def forward(ctx, x, factor):
        y = paddle.to_tensor(paddle.ones(x.shape), dtype='complex64')
        x_copy = x.clone()
        f = int(factor)
        while f > 0:
            if f % 2 == 1:
                y *= x_copy
            x_copy *= x_copy
            f = f // 2
        ctx.save_for_backward(x, factor)

        return y

    @staticmethod
    def backward(ctx, dy):
        x, factor, = ctx.saved_tensor()
        y = paddle.to_tensor(paddle.ones(x.shape), dtype='complex64')
        x = paddle.complex(paddle.real(x), -paddle.imag(x))
        f = int(factor) - 1
        while f > 0:
            if f % 2 == 1:
                y *= x
            x *= x
            f = f // 2
        grad_x = dy * factor * y

        return grad_x
