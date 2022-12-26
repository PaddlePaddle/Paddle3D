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

import math
import warnings
from typing import Optional

import paddle


class Transform3d:
    """
    A Transform3d object encapsulates a batch of N 3D transformations, and knows
    how to transform points and normal vectors.

    This code is based on https://github.com/facebookresearch/pytorch3d/blob/46cb5aaaae0cd40f729fd41a39c0c9a232b484c0/pytorch3d/transforms/transform3d.py#L20
    """

    def __init__(self, dtype='float32', matrix=None):
        """
        Args:
            dtype: The data type of the transformation matrix.
                to be used if `matrix = None`.
            matrix: A tensor of shape (4, 4) or of shape (minibatch, 4, 4)
                representing the 4x4 3D transformation matrix.
                If `None`, initializes with identity using
                the specified `dtype`.
        """

        if matrix is None:
            self._matrix = paddle.eye(4, dtype=dtype).reshape([1, 4, 4])
        else:
            if len(matrix.shape) not in (2, 3):
                raise ValueError(
                    '"matrix" has to be a 2- or a 3-dimensional tensor.')
            if matrix.shape[-2] != 4 or matrix.shape[-1] != 4:
                raise ValueError(
                    '"matrix" has to be a tensor of shape (minibatch, 4, 4)')
            self._matrix = matrix.reshape([-1, 4, 4])

        self._transforms = []  # store transforms to compose
        self._lu = None

    def __len__(self):
        return self.get_matrix().shape[0]

    def compose(self, *others):
        """
        Return a new Transform3d with the tranforms to compose stored as
        an internal list.

        Args:
            *others: Any number of Transform3d objects

        Returns:
            A new Transform3d with the stored transforms
        """
        out = Transform3d()
        out._matrix = self._matrix.clone()
        for other in others:
            if not isinstance(other, Transform3d):
                msg = "Only possible to compose Transform3d objects; got %s"
                raise ValueError(msg % type(other))
        out._transforms = self._transforms + list(others)
        return out

    def get_matrix(self):
        """
        Return a matrix which is the result of composing this transform
        with others stored in self.transforms. Where necessary transforms
        are broadcast against each other.
        For example, if self.transforms contains transforms t1, t2, and t3, and
        given a set of points x, the following should be true:

        .. code-block:: python

            y1 = t1.compose(t2, t3).transform(x)
            y2 = t3.transform(t2.transform(t1.transform(x)))
            y1.get_matrix() == y2.get_matrix()

        Returns:
            A transformation matrix representing the composed inputs.
        """
        composed_matrix = self._matrix.clone()
        if len(self._transforms) > 0:
            for other in self._transforms:
                other_matrix = other.get_matrix()
                composed_matrix = _broadcast_bmm(composed_matrix, other_matrix)
        return composed_matrix

    def _get_matrix_inverse(self):
        """
        Return the inverse of self._matrix.
        """
        return paddle.inverse(self._matrix)

    def inverse(self, invert_composed: bool = False):
        """
        Returns a new Transform3D object that represents an inverse of the
        current transformation.

        Args:
            invert_composed:
                - True: First compose the list of stored transformations
                  and then apply inverse to the result. This is
                  potentially slower for classes of transformations
                  with inverses that can be computed efficiently
                  (e.g. rotations and translations).
                - False: Invert the individual stored transformations
                  independently without composing them.

        Returns:
            A new Transform3D object contaning the inverse of the original
            transformation.
        """

        tinv = Transform3d()

        if invert_composed:
            # first compose then invert
            tinv._matrix = paddle.inverse(self.get_matrix())
        else:
            # self._get_matrix_inverse() implements efficient inverse
            # of self._matrix
            i_matrix = self._get_matrix_inverse()

            # 2 cases:
            if len(self._transforms) > 0:
                # a) Either we have a non-empty list of transforms:
                # Here we take self._matrix and append its inverse at the
                # end of the reverted _transforms list. After composing
                # the transformations with get_matrix(), this correctly
                # right-multiplies by the inverse of self._matrix
                # at the end of the composition.
                tinv._transforms = [
                    t.inverse() for t in reversed(self._transforms)
                ]
                last = Transform3d()
                last._matrix = i_matrix
                tinv._transforms.append(last)
            else:
                # b) Or there are no stored transformations
                # we just set inverted matrix
                tinv._matrix = i_matrix

        return tinv

    def stack(self, *others):
        transforms = [self] + list(others)
        matrix = paddle.concat([t._matrix for t in transforms], axis=0)
        out = Transform3d()
        out._matrix = matrix
        return out

    def transform_points(self, points, eps: Optional[float] = None):
        """
        Use this transform to transform a set of 3D points. Assumes row major
        ordering of the input points.

        Args:
            points: Tensor of shape (P, 3) or (N, P, 3)
            eps: If eps!=None, the argument is used to clamp the
                last coordinate before peforming the final division.
                The clamping corresponds to:
                last_coord := (last_coord.sign() + (last_coord==0)) *
                paddle.clamp(last_coord.abs(), eps),
                i.e. the last coordinates that are exactly 0 will
                be clamped to +eps.

        Returns:
            points_out: points of shape (N, P, 3) or (P, 3) depending
            on the dimensions of the transform
        """
        points_batch = points.clone()
        if len(points_batch.shape) == 2:
            points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
        if len(points_batch.shape) != 3:
            msg = "Expected points to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % repr(points.shape))

        N, P, _3 = points_batch.shape
        ones = paddle.ones([N, P, 1], dtype=points.dtype)
        points_batch = paddle.concat([points_batch, ones], axis=2)

        composed_matrix = self.get_matrix()
        points_out = _broadcast_bmm(points_batch, composed_matrix)
        denom = points_out[..., 3:]  # denominator
        if eps is not None:
            denom_sign = denom.sign() + (denom == 0.0).cast(denom.dtype)
            denom = denom_sign * paddle.clip(denom.abs(), eps)
        points_out = points_out[..., :3] / denom

        # When transform is (1, 4, 4) and points is (P, 3) return
        # points_out of shape (P, 3)
        if points_out.shape[0] == 1 and len(points.shape) == 2:
            points_out = points_out.reshape(points.shape)

        return points_out

    def transform_normals(self, normals):
        """
        Use this transform to transform a set of normal vectors.

        Args:
            normals: Tensor of shape (P, 3) or (N, P, 3)

        Returns:
            normals_out: Tensor of shape (P, 3) or (N, P, 3) depending
            on the dimensions of the transform
        """
        if len(normals.shape) not in [2, 3]:
            msg = "Expected normals to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % (normals.shape, ))
        composed_matrix = self.get_matrix()

        # TODO: inverse is bad! Solve a linear system instead
        mat = composed_matrix[:, :3, :3]
        normals_out = _broadcast_bmm(normals,
                                     mat.transpose([0, 2, 1]).inverse())

        # When transform is (1, 4, 4) and normals is (P, 3) return
        # normals_out of shape (P, 3)
        if normals_out.shape[0] == 1 and len(normals.shape) == 2:
            normals_out = normals_out.reshape(normals.shape)

        return normals_out

    def translate(self, *args, **kwargs):
        return self.compose(Translate(*args, **kwargs))

    def clone(self):
        """
        Deep copy of Transforms object. All internal tensors are cloned
        individually.

        Returns:
            new Transforms object.
        """
        other = Transform3d()
        if self._lu is not None:
            other._lu = [elem.clone() for elem in self._lu]
        other._matrix = self._matrix.clone()
        other._transforms = [t.clone() for t in self._transforms]
        return other

    def to(self, copy: bool = False, dtype=None):
        """
        Match functionality of paddle.cast()

        Args:
          copy: Boolean indicator whether or not to clone self. Default False.
          dtype: If not None, casts the internal tensor variables
              to a given paddle.dtype.

        Returns:
          Transform3d object.
        """
        if not copy and self.dtype == dtype:
            return self
        other = self.clone()
        other._matrix = self._matrix.to(dtype=dtype)
        for t in other._transforms:
            t.to(copy=copy, dtype=dtype)
        return other


class Translate(Transform3d):
    def __init__(self, x, y=None, z=None, dtype='float32'):
        """
        Create a new Transform3d representing 3D translations.

        Option I: Translate(xyz, dtype='float32')
            xyz should be a tensor of shape (N, 3)

        Option II: Translate(x, y, z, dtype='float32')
            Here x, y, and z will be broadcast against each other and
            concatenated to form the translation. Each can be:
                - A python scalar
                - A paddle scalar
                - A 1D paddle tensor

        This code is based on https://github.com/facebookresearch/pytorch3d/blob/46cb5aaaae0cd40f729fd41a39c0c9a232b484c0/pytorch3d/transforms/transform3d.py#L525
        """
        super().__init__()
        xyz = _handle_input(x, y, z, dtype, "Translate")
        N = xyz.shape[0]

        mat = paddle.eye(4, dtype=dtype)
        mat = mat.reshape([1, 4, 4]).tile([N, 1, 1])
        mat[:, 3, :3] = xyz
        self._matrix = mat

    def _get_matrix_inverse(self):
        """
        Return the inverse of self._matrix.
        """
        inv_mask = paddle.ones([1, 4, 4], dtype=self._matrix.dtype)
        inv_mask[0, 3, :3] = -1.0
        i_matrix = self._matrix * inv_mask
        return i_matrix


class Rotate(Transform3d):
    """
    This code is based on https://github.com/facebookresearch/pytorch3d/blob/46cb5aaaae0cd40f729fd41a39c0c9a232b484c0/pytorch3d/transforms/transform3d.py#L615
    """

    def __init__(self, R, dtype='float32', orthogonal_tol: float = 1e-5):
        """
        Create a new Transform3d representing 3D rotation using a rotation
        matrix as the input.

        Args:
            R: a tensor of shape (3, 3) or (N, 3, 3)
            orthogonal_tol: tolerance for the test of the orthogonality of R

        """
        super().__init__()
        if len(R.shape) == 2:
            R = R[None]
        if R.shape[-2:] != [3, 3]:
            msg = "R must have shape (3, 3) or (N, 3, 3); got %s"
            raise ValueError(msg % repr(R.shape))
        R = R.cast(dtype=dtype)
        _check_valid_rotation_matrix(R, tol=orthogonal_tol)
        N = R.shape[0]
        mat = paddle.eye(4, dtype=dtype)
        mat = mat.reshape([1, 4, 4]).tile([N, 1, 1])
        mat[:, :3, :3] = R
        self._matrix = mat

    def _get_matrix_inverse(self):
        """
        Return the inverse of self._matrix.
        """
        return self._matrix.transpose([0, 2, 1])


def _handle_coord(c, dtype):
    """
    Helper function for _handle_input.

    Args:
        c: Python scalar, paddle scalar, or 1D paddle.tensor

    Returns:
        c_vec: 1D paddle.tensor
    """
    if not paddle.is_tensor(c):
        c = paddle.to_tensor(c, dtype=dtype)
    if len(c.shape) == 0:
        c = c.reshape([1])
    return c


def _handle_input(x, y, z, dtype, name: str, allow_singleton: bool = False):
    """
    Helper function to handle parsing logic for building transforms. The output
    is always a tensor of shape (N, 3), but there are several types of allowed
    input.

    Case I: Single Matrix
        In this case x is a tensor of shape (N, 3), and y and z are None. Here just
        return x.

    Case II: Vectors and Scalars
        In this case each of x, y, and z can be one of the following
            - Python scalar
            - Torch scalar
            - Torch tensor of shape (N, 1) or (1, 1)
        In this case x, y and z are broadcast to tensors of shape (N, 1)
        and concatenated to a tensor of shape (N, 3)

    Case III: Singleton (only if allow_singleton=True)
        In this case y and z are None, and x can be one of the following:
            - Python scalar
            - Torch scalar
            - Torch tensor of shape (N, 1) or (1, 1)
        Here x will be duplicated 3 times, and we return a tensor of shape (N, 3)

    Returns:
        xyz: Tensor of shape (N, 3)

    This code is based on https://github.com/facebookresearch/pytorch3d/blob/46cb5aaaae0cd40f729fd41a39c0c9a232b484c0/pytorch3d/transforms/transform3d.py#L716
    """
    # If x is actually a tensor of shape (N, 3) then just return it
    if paddle.is_tensor(x) and len(x.shape) == 2:
        if x.shape[1] != 3:
            msg = "Expected tensor of shape (N, 3); got %r (in %s)"
            raise ValueError(msg % (x.shape, name))
        if y is not None or z is not None:
            msg = "Expected y and z to be None (in %s)" % name
            raise ValueError(msg)
        return x

    if allow_singleton and y is None and z is None:
        y = x
        z = x

    # Convert all to 1D tensors
    xyz = [_handle_coord(c, dtype) for c in [x, y, z]]

    # Broadcast and concatenate
    sizes = [c.shape[0] for c in xyz]
    N = max(sizes)
    for c in xyz:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = "Got non-broadcastable sizes %r (in %s)" % (sizes, name)
            raise ValueError(msg)
    xyz = [c.expand(N) for c in xyz]
    xyz = paddle.stack(xyz, axis=1)
    return xyz


def _broadcast_bmm(a, b):
    """
    Batch multiply two matrices and broadcast if necessary.

    Args:
        a: paddle tensor of shape (P, K) or (M, P, K)
        b: paddle tensor of shape (N, K, K)

    Returns:
        a and b broadcast multipled. The output batch dimension is max(N, M).

    To broadcast transforms across a batch dimension if M != N then
    expect that either M = 1 or N = 1. The tensor with batch dimension 1 is
    expanded to have shape N or M.

    This code is based on https://github.com/facebookresearch/pytorch3d/blob/46cb5aaaae0cd40f729fd41a39c0c9a232b484c0/pytorch3d/transforms/transform3d.py#L802
    """
    if len(a.shape) == 2:
        a = a[None]
    if len(a) != len(b):
        if not ((len(a) == 1) or (len(b) == 1)):
            msg = "Expected batch dim for bmm to be equal or 1; got %r, %r"
            raise ValueError(msg % (a.shape, b.shape))
        if len(a) == 1:
            a = a.expand(len(b), -1, -1)
        if len(b) == 1:
            b = b.expand(len(a), -1, -1)
    return a.bmm(b)


def _check_valid_rotation_matrix(R, tol: float = 1e-7):
    """
    Determine if R is a valid rotation matrix by checking it satisfies the
    following conditions:

    ``RR^T = I and det(R) = 1``

    Args:
        R: an (N, 3, 3) matrix

    Returns:
        None

    Emits a warning if R is an invalid rotation matrix.

    This code is based on https://github.com/facebookresearch/pytorch3d/blob/46cb5aaaae0cd40f729fd41a39c0c9a232b484c0/pytorch3d/transforms/transform3d.py#L831
    """
    N = R.shape[0]
    eye = paddle.eye(3, dtype=R.dtype)
    eye = eye.reshape([1, 3, 3]).expand([N, -1, -1])
    orthogonal = paddle.allclose(R.bmm(R.transpose([0, 2, 1])), eye, atol=tol)
    det_R = paddle.linalg.det(R)
    no_distortion = paddle.allclose(det_R, paddle.ones_like(det_R))
    if not (orthogonal and no_distortion):
        msg = "R is not a valid rotation matrix"
        warnings.warn(msg)
    return
