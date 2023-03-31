from typing import List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn.functional as F


def get_dimensions(img: paddle.Tensor) -> List[int]:
    """Returns the dimensions of an image as [channels, height, width].
    Args:
        img (Tensor): The image to be checked.
    Returns:
        List[int]: The image dimensions.
    """
    channels = 1 if img.ndim == 2 else img.shape[-3]
    height, width = img.shape[-2:]
    return [channels, height, width]


def _get_inverse_affine_matrix(center: List[float],
                               angle: float,
                               translate: List[float],
                               scale: float,
                               shear: List[float],
                               inverted: bool = True) -> List[float]:
    """
    This fuction refers to https://github.com/pypaddle/vision/blob/main/paddlevision/transforms/functional.py#L992
    """

    # Helper method to compute inverse matrix for affine transformation

    # Pillow requires inverse affine transformation matrix:
    # Affine matrix is : M = T * C * RotateScaleShear * C^-1
    #
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RotateScaleShear is rotation with scale and shear matrix
    #
    #       RotateScaleShear(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1

    def radians(angle):
        pi = paddle.full([1], np.pi)
        degree = pi / 180. * angle
        return degree

    # rot = math.radians(angle)
    # sx = math.radians(shear[0])
    # sy = math.radians(shear[1])
    rot = radians(angle)
    sx = radians(shear[0])
    sy = radians(shear[1])

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    # a = math.cos(rot - sy) / math.cos(sy)
    # b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    # c = math.sin(rot - sy) / math.cos(sy)
    # d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)
    a = paddle.cos(rot - sy) / paddle.cos(sy)
    b = -paddle.cos(rot - sy) * paddle.tan(sx) / paddle.cos(sy) - paddle.sin(
        rot)
    c = paddle.sin(rot - sy) / paddle.cos(sy)
    d = -paddle.sin(rot - sy) * paddle.tan(sx) / paddle.cos(sy) + paddle.cos(
        rot)

    if inverted:
        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        matrix = [d, -b, 0.0, -c, a, 0.0]
        matrix = [x / scale for x in matrix]
        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
        matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += cx
        matrix[5] += cy
    else:
        matrix = [a, b, 0.0, c, d, 0.0]
        matrix = [x * scale for x in matrix]
        # Apply inverse of center translation: RSS * C^-1
        matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
        matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
        # Apply translation and center : T * C * RSS * C^-1
        matrix[2] += cx + tx
        matrix[5] += cy + ty

    return matrix


def _gen_affine_grid(
        theta: paddle.Tensor,
        w: int,
        h: int,
        ow: int,
        oh: int,
) -> paddle.Tensor:
    # https://github.com/pytorch/pytorch/blob/74b65c32be68b15dc7c9e8bb62459efbfbde33d8/aten/src/ATen/native/
    # AffineGridGenerator.cpp#L18
    # Difference with AffineGridGenerator is that:
    # 1) we normalize grid values after applying theta
    # 2) we can normalize by other image size, such that it covers "extend" option like in PIL.Image.rotate

    d = 0.5
    base_grid = paddle.empty([1, oh, ow, 3], dtype=theta.dtype)
    x_grid = paddle.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, num=ow)
    base_grid[..., 0] = x_grid
    y_grid = paddle.linspace(
        -oh * 0.5 + d, oh * 0.5 + d - 1, num=oh).unsqueeze([-1])
    base_grid[..., 1] = y_grid
    base_grid[..., 2] = 1

    tmp_tensor = paddle.to_tensor([0.5 * w, 0.5 * h],
                                  dtype=theta.dtype,
                                  place=paddle.CPUPlace())
    rescaled_theta = theta.transpose([0, 2, 1]) / paddle.to_tensor(tmp_tensor)
    output_grid = base_grid.reshape([1, oh * ow, 3]).bmm(rescaled_theta)
    return output_grid.reshape([1, oh, ow, 2])


def _cast_squeeze_in(img: paddle.Tensor, req_dtypes: List[paddle.dtype]
                     ) -> Tuple[paddle.Tensor, bool, bool, paddle.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(axis=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.cast(req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def _cast_squeeze_out(img: paddle.Tensor, need_cast: bool, need_squeeze: bool,
                      out_dtype: paddle.dtype) -> paddle.Tensor:
    if need_squeeze:
        img = img.squeeze(axis=0)

    if need_cast:
        if out_dtype in (paddle.uint8, paddle.int8, paddle.int16, paddle.int32,
                         paddle.int64):
            # it is better to round before cast
            img = paddle.round(img)
        img = img.cast(out_dtype)

    return img


def _apply_grid_transform(
        img: paddle.Tensor, grid: paddle.Tensor, mode: str,
        fill: Optional[Union[int, float, List[float]]]) -> paddle.Tensor:

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(
        img, [grid.dtype])

    if img.shape[0] > 1:
        # Apply same grid to a batch of images
        grid = grid.expand(
            [img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3]])

    # np.save('grid_img.npy', img.numpy())
    img = F.grid_sample(
        img, grid, mode=mode, padding_mode="zeros", align_corners=False)
    # np.save('grid_sample.npy', img.numpy())
    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)

    return img


def _rotate(img: paddle.Tensor,
            matrix: List[float],
            interpolation: str = "nearest") -> paddle.Tensor:
    ow, oh = img.shape[-1], img.shape[-2]
    w, h = img.shape[-1], img.shape[-2]
    dtype = img.dtype if paddle.is_floating_point(img) else paddle.float32
    theta = paddle.concat(matrix).reshape([1, 2, 3])
    # grid will be generated on the same device as theta and img
    grid = _gen_affine_grid(theta, w=w, h=h, ow=ow, oh=oh)
    # np.save("grid.npy", grid.numpy())

    img = _apply_grid_transform(img, grid, interpolation, fill=None)
    return img


def rotate(img: paddle.Tensor,
           angle: float,
           interpolation: str = "nearest",
           center: Optional[List[int]] = None):
    center_f = [0.0, 0.0]
    if center is not None:
        _, height, width = get_dimensions(img)
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [
            1.0 * (c - s * 0.5) for c, s in zip(center, [width, height])
        ]

    # due to current incoherence of rotation angle direction between affine and rotate implementations
    # we need to set -angle.
    matrix = _get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0,
                                        [0.0, 0.0])

    # np.save('origin_img.npy', img.numpy())
    img = _rotate(img, matrix=matrix, interpolation=interpolation)
    return img


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def nan_to_num(x, nan=0.0, posinf=None, neginf=None, name=None):
    """
    Replaces NaN, positive infinity, and negative infinity values in input tensor.
    """
    # NOTE(tiancaishaonvjituizi): it seems that paddle handles the dtype of python float number
    # incorrectly, so we have to explicitly contruct tensors here
    posinf_value = paddle.full_like(x, float("+inf"))
    neginf_value = paddle.full_like(x, float("-inf"))
    nan = paddle.full_like(x, nan)
    assert x.dtype in [paddle.float32, paddle.float64]
    is_float32 = x.dtype == paddle.float32
    if posinf is None:
        posinf = (np.finfo(np.float32).max
                  if is_float32 else np.finfo(np.float64).max)
    posinf = paddle.full_like(x, posinf)
    if neginf is None:
        neginf = (np.finfo(np.float32).min
                  if is_float32 else np.finfo(np.float64).min)
    neginf = paddle.full_like(x, neginf)
    x = paddle.where(paddle.isnan(x), nan, x)
    x = paddle.where(x == posinf_value, posinf, x)
    x = paddle.where(x == neginf_value, neginf, x)
    return x
