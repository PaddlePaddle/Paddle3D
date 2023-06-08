import cv2
import numpy as np
from .coord_util import image_undistort


class Standard_camera:
    def __init__(self,
                 cameraA_intrinsic,
                 cameraA2ego_matrix,
                 imageA_shape,
                 cameraB_intrinsic,
                 cameraB2ego_matrix,
                 imageB_shape,
                 cameraA_distortion=None,
                 cameraB_distortion=None):
        '''
        :param cameraA_intrinsic: parameters of Virtual camera
        :param cameraA2ego_matrix: parameters of Virtual camera
        :param cameraA_distortion: distortion parameters of Virtual camera
        :param imageA_shape: image shape of Virtual camera
        :param cameraB_intrinsic: parameters of current camera
        :param cameraB2ego_matrix: parameters of current camera
        :param cameraB_distortion: distortion parameters of current camera
        :param imageB_shape: image shape of current camera
        '''
        ''' cameraA parameter '''
        self.cameraA_intrinsic = cameraA_intrinsic
        self.cameraA2ego_matrix = cameraA2ego_matrix
        self.cameraA_distortion = cameraA_distortion
        self.imageA_shape = imageA_shape
        ''' cameraB parameter '''
        self.cameraB_intrinsic = cameraB_intrinsic
        self.cameraB2ego_matrix = cameraB2ego_matrix
        self.cameraB_distortion = cameraB_distortion
        self.imageB_shape = imageB_shape

    def project_B2A(self, imageB, height=0):
        '''
        :param imageB:
        :return:
        '''
        u1, u2 = int(0.6 * self.imageB_shape[0]), int(
            0.8 * self.imageB_shape[0])  #
        v1, v2 = int(0.33 * self.imageB_shape[1]), int(
            0.66 * self.imageB_shape[1])  #
        base_points = np.array([(u1, v1, 1), (u1, v2, 1), (u2, v1, 1),
                                (u2, v2, 1)])
        res = self.get_project_matrix(base_points, height)
        if self.cameraB_distortion is not None:  #
            imageB = image_undistort(imageB, self.cameraB_intrinsic,
                                     self.cameraB_distortion)
        imageB_in_A = cv2.warpPerspective(
            imageB, res, (self.imageA_shape[1], self.imageA_shape[0]))
        if self.cameraA_distortion is not None:  #
            imageB_in_A = imageB_in_A  # distort_image
        return imageB_in_A

    def get_matrix(self, height=0):
        u1, u2 = int(0.6 * self.imageB_shape[0]), int(
            0.8 * self.imageB_shape[0])
        v1, v2 = int(0.33 * self.imageB_shape[1]), int(
            0.66 * self.imageB_shape[1])
        base_points = np.array([(u1, v1, 1), (u1, v2, 1), (u2, v1, 1),
                                (u2, v2, 1)])
        res = self.get_project_matrix(base_points, height)
        return res

    def get_project_matrix(self, base_points, height=0):
        """
        :param base_points: selected four points on image of Virtual camera
        :return: homography matrix
        """
        ''' base_points project to ego_points_B '''
        ego_points_B = self.imageview2ego(
            base_points.T, self.cameraB_intrinsic,
            np.linalg.inv(self.cameraB2ego_matrix), height)
        ''' '''
        ego2cameraA_matrix = np.linalg.inv(self.cameraA2ego_matrix)
        cameraB_points_in_A = np.dot(ego2cameraA_matrix[:3, :3], ego_points_B) + \
                              ego2cameraA_matrix[:3, 3].reshape(3, 1)
        image_points_B_in_A_ = self.cameraA_intrinsic @ cameraB_points_in_A
        image_points_B_in_A = image_points_B_in_A_ / image_points_B_in_A_[2]
        '''  '''
        point1 = base_points[:, :2].astype(np.float32)
        point2 = image_points_B_in_A[:2].T.astype(np.float32)
        matrix_B2A = cv2.getPerspectiveTransform(point1, point2)
        return matrix_B2A

    def imageview2ego(self, image_view_points, camera_intrinsic,
                      ego2camera_matrix, height):
        '''
        :param image_view_points: np.ndarray 3*n [[u,v,1],,,,]
        :param camera_intrinsic: np.ndarray 3*3
        :param ego2camera_matrix: nd.ndarray 4*4 ego -> camera
        :param height: int defalut = 0
        :return: ndarray 3*n [[x_ego,y_ego,hegiht],,,]
        '''
        camera_intrinsic_inv = np.linalg.inv(camera_intrinsic)
        R_inv = ego2camera_matrix[:3, :3].T
        T = ego2camera_matrix[:3, 3]
        mat1 = np.dot(np.dot(R_inv, camera_intrinsic_inv), image_view_points)
        mat2 = np.dot(R_inv, T)
        Zc = (height + mat2[2]) / mat1[2]
        points_ego = Zc * mat1 - np.expand_dims(mat2, 1)
        return points_ego
