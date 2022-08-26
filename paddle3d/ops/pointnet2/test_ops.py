import numpy as np
import paddle
from paddle.utils.cpp_extension import load

pointnet2_ops = load(
    name='pointnet2_ops',
    sources=[
        # 'ball_query_gpu.cu',
        # 'ball_query.cc',
        'group_points_gpu.cu',
        'group_points.cc',
        'gather_points_gpu.cu',
        'gather_points.cc',
        #  'sampling_gpu.cu', 'sampling.cc',
        # 'box_utils_gpu.cu',
        # 'box_utils.cc'
    ])

np.random.seed(0)

# ========================= test ball_query ==============================
# b, n, m, radius, nsample = 2, 16384, 4096, 0.8, 16
# xyz = np.random.randn(b, n, 3)
# new_xyz = np.random.randn(b, m, 3)

# xyz = paddle.to_tensor(xyz, dtype='float32')
# new_xyz = paddle.to_tensor(new_xyz, dtype='float32')
# print(xyz[0, 0, :])
# print(new_xyz[0, 0, :])

# idx = pointnet2_ops.ball_query(new_xyz, xyz, radius, nsample)
# print(idx)

# ======================== test group_operation ==========================
b, c, n, npoints, nsample = 2, 3, 16384, 4096, 16
points = np.random.randn(b, c, n)
points = paddle.to_tensor(points, dtype='float32')
points.stop_gradient = False
idx = np.random.randint(0, n, size=[b, npoints, nsample])
idx = paddle.to_tensor(idx, dtype='int32')
out = pointnet2_ops.group_operation(points, idx)
print(out[0, 0, 0, :])

loss = out.sum()
print(loss)
loss.backward()

grad = points.grad.numpy()
th_grad = np.load('/workspace/temp/th_grad.npy')
diff_mean = np.abs(grad - th_grad).mean()
print(diff_mean)

# ======================== test gather_operation ===========================
# b, c, n, npoints = 2, 3, 16384, 4096
# points = np.random.randn(b, c, n)
# points = paddle.to_tensor(points, dtype='float32')
# points.stop_gradient=False
# idx = np.random.randint(0, n, size=[b, npoints])
# idx = paddle.to_tensor(idx, dtype='int32')
# out = pointnet2_ops.gather_operation(points, idx)
# print(out[0, 0, :])

# loss = out.sum()
# print(loss)
# loss.backward()
# grad = points.grad.numpy()
# th_grad = np.load('/workspace/temp/th_grad.npy')
# diff_mean = np.abs(grad - th_grad).mean()
# print(diff_mean)

# ======================== test fps ===========================
# b, c, n, npoints = 2, 3, 16384, 4096
# points = np.random.randn(b, n, c)
# points = paddle.to_tensor(points, dtype='float32')
# sampled_idx = pointnet2_ops.farthest_point_sample(points, npoints)
# print(sampled_idx.sum())

# =================== test points_in_boxes_gpu ================
# b, n, npoints = 2, 4096, 64
# points = np.random.randn(b, n, 3)
# boxes = np.random.randn(b, npoints, 7)
# points = paddle.to_tensor(points, dtype='float32')
# boxes = paddle.to_tensor(boxes, dtype='float32')
# box_idx_of_points = pointnet2_ops.points_in_boxes_gpu(boxes, points)
# print(box_idx_of_points.sum())
