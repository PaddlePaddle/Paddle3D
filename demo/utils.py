import cv2
import numba
import numpy as np
import mayavi.mlab as mlab
# import open3d as o3d
# import matplotlib.pyplot as plt


from paddle3d.transforms.target_generator import encode_label


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.
        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref
        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]
        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)
        velodyne coord:
        front x, left y, up z
        rect/ref camera coord:
        right x, down y, front z
        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam['R'], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


def get_ratio(ori_img_size, output_size, down_ratio=(4, 4)):
    return np.array([[
        down_ratio[1] * ori_img_size[1] / output_size[1],
        down_ratio[0] * ori_img_size[0] / output_size[0]
    ]], np.float32)


def get_img(img_path):
    img = cv2.imread(img_path)
    origin_shape = img.shape
    img = cv2.resize(img, (1280, 384))

    target_shape = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / 255.0
    img = np.subtract(img, np.array([0.485, 0.456, 0.406]))
    img = np.true_divide(img, np.array([0.229, 0.224, 0.225]))
    img = np.array(img, np.float32)

    img = img.transpose(2, 0, 1)
    img = img[None, :, :, :]

    return img, origin_shape, target_shape


def total_pred_by_conf_to_kitti_records(
        total_pred, conf, class_names=["Car", "Cyclist", "Pedestrian"]):
    """convert total_pred to kitti_records"""
    kitti_records_list = []
    for p in total_pred:
        if p[-1] > conf:
            p = list(p)
            p[0] = class_names[int(p[0])]
            # default, to kitti_records formate
            p.insert(1, 0.0)
            p.insert(2, 0)
            kitti_records_list.append(p)
    kitti_records = np.array(kitti_records_list)

    return kitti_records


def total_imgpred_by_conf_to_kitti_records(total_pred,
                                           conf,
                                           class_names=[
                                               "Car", "Cyclist", "Pedestrian"
                                           ]):
    """convert total_pred to kitti_records"""
    kitti_records_list = []
    for idx in range(len(total_pred['confidences'])):
        box2d = total_pred['bboxes_2d'][idx]
        box3d = total_pred['bboxes_3d'][idx]
        label = total_pred['labels'][idx]
        cnf = total_pred['confidences'][idx]
        if cnf > conf:
            p = []
            p.append(class_names[int(label)])
            # default, to kitti_records formate
            p.extend([0.0, 0.0, 0.0])
            p.extend(box2d)
            p.extend([box3d[5], box3d[4], box3d[3], box3d[0], box3d[1], box3d[2], box3d[-1]])
            kitti_records_list.append(p)
    kitti_records = np.array(kitti_records_list)

    return kitti_records


def make_imgpts_list(bboxes_3d, K):
    """to 8 points on image"""
    # external parameters do not transform
    rvec = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tvec = np.array([[0.0], [0.0], [0.0]])

    imgpts_list = []
    for box3d in bboxes_3d:

        locs = np.array(box3d[0:3])
        rot_y = np.array(box3d[6])

        height, width, length = box3d[3:6]
        _, box2d, box3d = encode_label(K, rot_y,
                                       np.array([length, height, width]), locs)

        if np.all(box2d == 0):
            continue

        imgpts, _ = cv2.projectPoints(box3d.T, rvec, tvec, K, 0)
        imgpts_list.append(imgpts)

    return imgpts_list


def draw_mono_3d(img, imgpts_list):
    """draw smoke result to photo"""
    connect_line_id = [
        [1, 0],
        [2, 7],
        [3, 6],
        [4, 5],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
        [0, 7],
        [7, 6],
        [6, 5],
        [5, 0],
    ]

    img_draw = img.copy()

    for imgpts in imgpts_list:
        for p in imgpts:
            p_x, p_y = int(p[0][0]), int(p[0][1])
            cv2.circle(img_draw, (p_x, p_y), 1, (0, 255, 0), -1)
        for i, line_id in enumerate(connect_line_id):

            p1 = (int(imgpts[line_id[0]][0][0]), int(imgpts[line_id[0]][0][1]))
            p2 = (int(imgpts[line_id[1]][0][0]), int(imgpts[line_id[1]][0][1]))

            if i <= 3:  # body
                color = (255, 0, 0)
            elif i <= 7:  # head
                color = (0, 0, 255)
            else:  # tail
                color = (255, 255, 0)

            cv2.line(img_draw, p1, p2, color, 1)

    cv2.imshow('output', img_draw)
    cv2.waitKey(0)
    cv2.destroyWindow('output')


def read_point(file_path, num_point_dim):
    points = np.fromfile(file_path, np.float32).reshape(-1, num_point_dim)
    points = points[:, :4]
    return points


@numba.jit(nopython=True)
def _points_to_voxel(points, voxel_size, point_cloud_range, grid_size, voxels,
                     coords, num_points_per_voxel, grid_idx_to_voxel_idx,
                     max_points_in_voxel, max_voxel_num):
    num_voxels = 0
    num_points = points.shape[0]
    # x, y, z
    coord = np.zeros(shape=(3,), dtype=np.int32)

    for point_idx in range(num_points):
        outside = False
        for i in range(3):
            coord[i] = np.floor(
                (points[point_idx, i] - point_cloud_range[i]) / voxel_size[i])
            if coord[i] < 0 or coord[i] >= grid_size[i]:
                outside = True
                break
        if outside:
            continue
        voxel_idx = grid_idx_to_voxel_idx[coord[2], coord[1], coord[0]]
        if voxel_idx == -1:
            voxel_idx = num_voxels
            if num_voxels >= max_voxel_num:
                continue
            num_voxels += 1
            grid_idx_to_voxel_idx[coord[2], coord[1], coord[0]] = voxel_idx
            coords[voxel_idx, 0:3] = coord[::-1]
        curr_num_point = num_points_per_voxel[voxel_idx]
        if curr_num_point < max_points_in_voxel:
            voxels[voxel_idx, curr_num_point] = points[point_idx]
            num_points_per_voxel[voxel_idx] = curr_num_point + 1

    return num_voxels


def hardvoxelize(points, point_cloud_range, voxel_size, max_points_in_voxel,
                 max_voxel_num):
    num_points, num_point_dim = points.shape[0:2]
    point_cloud_range = np.array(point_cloud_range)
    voxel_size = np.array(voxel_size)
    voxels = np.zeros((max_voxel_num, max_points_in_voxel, num_point_dim),
                      dtype=points.dtype)
    coords = np.zeros((max_voxel_num, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros((max_voxel_num,), dtype=np.int32)
    grid_size = np.round((point_cloud_range[3:6] - point_cloud_range[0:3]) /
                         voxel_size).astype('int32')

    grid_size_x, grid_size_y, grid_size_z = grid_size

    grid_idx_to_voxel_idx = np.full((grid_size_z, grid_size_y, grid_size_x),
                                    -1,
                                    dtype=np.int32)

    num_voxels = _points_to_voxel(points, voxel_size, point_cloud_range,
                                  grid_size, voxels, coords,
                                  num_points_per_voxel, grid_idx_to_voxel_idx,
                                  max_points_in_voxel, max_voxel_num)

    voxels = voxels[:num_voxels]
    coords = coords[:num_voxels]
    num_points_per_voxel = num_points_per_voxel[:num_voxels]

    return voxels, coords, num_points_per_voxel


def preprocess(file_path, num_point_dim, point_cloud_range, voxel_size,
               max_points_in_voxel, max_voxel_num):
    points = read_point(file_path, num_point_dim)
    voxels, coords, num_points_per_voxel = hardvoxelize(
        points, point_cloud_range, voxel_size, max_points_in_voxel,
        max_voxel_num)

    return voxels, coords, num_points_per_voxel


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def compute_box_3d(obj):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj[-1])

    # 3d bounding box dimensions
    l = obj[4];
    w = obj[3];
    h = obj[5];

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] - obj[1];
    corners_3d[1, :] = corners_3d[1, :] - obj[2];
    corners_3d[2, :] = corners_3d[2, :] + obj[0];

    return np.transpose(corners_3d)


def draw_lidar(pc, color=None, fig=None, bgcolor=(0, 0, 0), pts_scale=1, pts_mode='point', pts_color=None):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    '''
    if fig is None: fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:, 2]
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color, color=pts_color, mode=pts_mode, colormap='gnuplot',
                  scale_factor=pts_scale, figure=fig)

    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)

    # draw axis
    axes = np.array([
        [2., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 2., 0.],
    ], dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)

    # draw fov (todo: update to real sensor spec.)
    fov = np.array([  # 45 degree
        [20., 20., 0., 0.],
        [20., -20., 0., 0.],
    ], dtype=np.float64)

    mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                figure=fig)
    mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                figure=fig)

    # draw square region
    TOP_Y_MIN = -20
    TOP_Y_MAX = 20
    TOP_X_MIN = 0
    TOP_X_MAX = 40
    TOP_Z_MIN = -2.0
    TOP_Z_MAX = 0.4

    x1 = TOP_X_MIN
    x2 = TOP_X_MAX
    y1 = TOP_Y_MIN
    y2 = TOP_Y_MAX
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)

    # mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def draw_gt_boxes3d(gt_boxes3d, fig, color=(1, 1, 1), line_width=1, draw_text=True, text_scale=(1, 1, 1),
                    color_list=None):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    '''
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text: mlab.text3d(b[4, 0], b[4, 1], b[4, 2], '%d' % n, scale=text_scale, color=color, figure=fig)
        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)
    # mlab.show(1)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return


def show_lidar_with_boxes(pc_velo, objects, scores, calib):
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) '''

    # print(('All point num: ', pc_velo.shape[0]))
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))

    draw_lidar(pc_velo, fig=fig)

    num_bbox3d, bbox3d_dims = objects.shape
    for box_idx in range(num_bbox3d):
        # filter fake results: score = -1
        if scores[box_idx] <= 0.3:
            continue
        obj = objects[box_idx]
        # Draw 3d bounding box
        box3d_pts_3d = compute_box_3d(obj)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)

        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
    mlab.show()


def pts2bev(pts):
    side_range = (-75, 75)
    fwd_range = (0, 75)
    height_range = (-2, 5)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    f_filter = np.logical_and(x > fwd_range[0], x < fwd_range[1])
    s_filter = np.logical_and(y > side_range[0], y < side_range[1])
    h_filter = np.logical_and(z > height_range[0], z < height_range[1])
    filter = np.logical_and(f_filter, s_filter)
    filter = np.logical_and(filter, h_filter)
    indices = np.argwhere(filter).flatten()
    x, y, z = x[indices], y[indices], z[indices]

    res = 0.25
    x_img = (-y / res).astype(np.int32)
    y_img = (-x / res).astype(np.int32)
    x_img = x_img - int(np.floor(side_range[0]) / res)
    y_img = y_img + int(np.floor(fwd_range[1]) / res)

    pixel_value = [255, 255, 255]

    x_max = int((side_range[1] - side_range[0]) / res) + 1
    y_max = int((fwd_range[1] - fwd_range[0]) / res) + 1

    im = np.zeros([y_max, x_max, 3], dtype=np.uint8)
    im[y_img, x_img] = pixel_value

    return im[:, :]


def show_bev_with_boxes(pc_velo, objects, scores, calib):
    bev_im = pts2bev(pc_velo)
    num_bbox3d, bbox3d_dims = objects.shape
    for box_idx in range(num_bbox3d):
        # filter results
        if scores[box_idx] <= 0.3:
            continue
        obj = objects[box_idx]
        # Draw bev bounding box
        box3d_pts_3d = compute_box_3d(obj)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)

        bpts = box3d_pts_3d_velo[:4, :2]
        bpts = bpts[:, [1, 0]]
        print((int(bpts[0, 0] * 4), int(bpts[0, 1] * 4 + 300)), (int(bpts[1, 0] * 4), int(bpts[1, 1] * 4 + 300)))

        cv2.line(bev_im,
                 (int(-bpts[0, 0] * 4 + 300), int(300 - bpts[0, 1] * 4)),
                 (int(-bpts[1, 0] * 4 + 300), int(300 - bpts[1, 1] * 4)),
                 (0, 0, 255), 2)
        cv2.line(bev_im,
                 (int(-bpts[1, 0] * 4 + 300), int(300 - bpts[1, 1] * 4)),
                 (int(-bpts[2, 0] * 4 + 300), int(300 - bpts[2, 1] * 4)),
                 (0, 0, 255), 2)
        cv2.line(bev_im,
                 (int(-bpts[2, 0] * 4 + 300), int(300 - bpts[2, 1] * 4)),
                 (int(-bpts[3, 0] * 4 + 300), int(300 - bpts[3, 1] * 4)),
                 (0, 0, 255), 2)
        cv2.line(bev_im,
                 (int(-bpts[3, 0] * 4 + 300), int(300 - bpts[3, 1] * 4)),
                 (int(-bpts[0, 0] * 4 + 300), int(300 - bpts[0, 1] * 4)),
                 (0, 0, 255), 2)
    cv2.imshow('bev', bev_im)
    cv2.waitKey(0)
    cv2.destroyWindow('bev')
