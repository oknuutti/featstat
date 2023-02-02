import math
from functools import lru_cache

import numpy as np
import quaternion  # adds to numpy  # noqa # pylint: disable=unused-import

from featstat.algo import tools


class Camera:
    def __init__(self, width, height, x_fov=None, y_fov=None, sensor_size_mm=None, focal_length_mm=None,
                 dist_coefs=None, cam_mx=None):
        self.width = width      # in pixels
        self.height = height    # in pixels
        self.x_fov = x_fov      # in deg
        self.y_fov = y_fov      # in deg

        if x_fov is None and cam_mx is not None:
            cam_mx = np.array(cam_mx)
            self.x_fov = x_fov = math.degrees(math.atan(self.width / cam_mx[0, 0] / 2) * 2)
            self.y_fov = y_fov = math.degrees(math.atan(self.height / cam_mx[1, 1] / 2) * 2)

        # camera matrix estimated using cv2.calibrateCamera
        if cam_mx is not None:
            self._cam_mx = np.array(cam_mx)
        else:
            self._cam_mx = self.intrinsic_camera_mx()

        self.focal_length = None
        self.sensor_width = None
        self.sensor_height = None
        self.f_stop = None
        self.aperture = None
        self.dist_coefs = ([0.]*5) if dist_coefs is None else dist_coefs

        assert sensor_size_mm is None or x_fov is None or focal_length_mm is None, \
            'give only two of sensor_size, focal_length and fov'

        if sensor_size_mm is not None:
            sw, sh = sensor_size_mm  # in mm
            self.sensor_width = sw
            self.sensor_height = sh
            if x_fov is not None:
                self.focal_length = min(sw / math.tan(math.radians(x_fov) / 2) / 2,
                                        sh / math.tan(math.radians(y_fov) / 2) / 2)

        if focal_length_mm is not None:
            self.focal_length = focal_length_mm    # in mm
            if sensor_size_mm is not None:
                self.x_fov = math.degrees(math.atan(sensor_size_mm[0] / focal_length_mm / 2) * 2)
                self.y_fov = math.degrees(math.atan(sensor_size_mm[1] / focal_length_mm / 2) * 2)
            else:
                self.sensor_width = math.tan(math.radians(x_fov) / 2) * 2 * focal_length_mm
                self.sensor_height = math.tan(math.radians(y_fov) / 2) * 2 * focal_length_mm

        if self.focal_length is not None:
            self.pixel_size = 1e3 * min(self.sensor_width / self.width, self.sensor_height / self.width)  # in um

    @property
    def cam_mx(self):
        return self.intrinsic_camera_mx()

    @cam_mx.setter
    def cam_mx(self, cam_mx):
        self._cam_mx = cam_mx

    @property
    def inv_cam_mx(self):
        return self.inv_intrinsic_camera_mx()

    def intrinsic_camera_mx(self):
        if self._cam_mx is None:
            return Camera._intrinsic_camera_mx(self.width, self.height, self.x_fov, self.y_fov)
        else:
            return Camera._intrinsic_camera_mx_2(tuple(self._cam_mx.flatten()))

    @staticmethod
    def _intrinsic_camera_mx(width, height, x_fov, y_fov):
        # opencv cam +z axis, -y up

        x = width / 2
        y = height / 2
        fl_x = x / math.tan(math.radians(x_fov) / 2)
        fl_y = y / math.tan(math.radians(y_fov) / 2)
        return np.array([[fl_x, 0, x],
                         [0, fl_y, y],
                         [0, 0, 1]], dtype=np.float32)

    @staticmethod
    @lru_cache(maxsize=3)
    def _intrinsic_camera_mx_2(cam_mx, inverse=False):
        cam_mx = np.array(cam_mx, dtype=np.float32).reshape((3, 3))
        return np.linalg.inv(cam_mx) if inverse else cam_mx

    def inv_intrinsic_camera_mx(self):
        if self._cam_mx is None:
            return Camera._inv_intrinsic_camera_mx(self.width, self.height, self.x_fov, self.y_fov)
        else:
            return Camera._intrinsic_camera_mx_2(tuple(self._cam_mx.flatten()), inverse=True)

    @staticmethod
    @lru_cache(maxsize=3)
    def _inv_intrinsic_camera_mx(w, h, xfov, yfov):
        return np.linalg.inv(Camera._intrinsic_camera_mx(w, h, xfov, yfov))

    def to_unit_sphere(self, pts2d, undistort=True):
        if undistort and self.dist_coefs is not None:
            pts2d = Camera._undistort(pts2d, self.intrinsic_camera_mx(), self.dist_coefs)
        pts2d += 0.5
        pts2dh = np.hstack((pts2d, np.ones((len(pts2d), 1))))
        v = self.inv_cam_mx.dot(pts2dh.T).T
        return tools.normalize_mx(v)

    def calc_img_xy(self, x, y, z, distort=True):
        """ x, y, z are in opencv cam frame where +z axis, -y up, return image coordinates  """
        K = self.intrinsic_camera_mx()[:2, :]
        xd, yd = x / z, y / z

        # DISTORT
        if distort and self.dist_coefs is not None:
            xd, yd = self.distort(np.array([[xd, yd]]))

        ix, iy = K.dot(np.array([xd, yd, 1]))
        return ix, iy

    def calc_img_R(self, R, distort=True):
        """
        R is a matrix where each row is a point in camera frame,
        returns a matrix where each row corresponds to points in image space """
        R = R / R[:, 2:]

        # DISTORT
        if distort and self.dist_coefs is not None:
            R = self.distort(R).squeeze()
            R = np.hstack([R, np.ones((len(R), 1))])

        K = self.intrinsic_camera_mx()[:2, :].T
        iR = R.dot(K)
        return iR

    def project(self, pts3d):
        return self._project(pts3d, self.intrinsic_camera_mx(), self.dist_coefs)

    def undistort(self, P):
        if len(P) > 0 and self.dist_coefs is not None:
            return Camera._undistort(P, self.intrinsic_camera_mx(), self.dist_coefs)
        return P

    def distort(self, P):
        if self.dist_coefs is not None:
            dP = Camera._distort_own(P, self.dist_coefs, cam_mx=self.intrinsic_camera_mx(),
                                     inv_cam_mx=None if P.shape[1] == 3 else self.inv_intrinsic_camera_mx())
            return dP[:, :2] / dP[:, 2:] if dP.shape[1] == 3 else dP
        return P

    @staticmethod
    def _project(P, K, dist_coefs):
        import cv2
        return cv2.projectPoints(P.reshape((-1, 3)), #np.hstack([P, np.ones((len(P), 1))]).reshape((-1, 1, 4)),
                                 np.array([0, 0, 0], dtype=np.float32), np.array([0, 0, 0], dtype=np.float32),
                                 K, np.array(dist_coefs),
                                 jacobian=False)[0].squeeze()

    @staticmethod
    def _undistort(P, cam_mx, dist_coefs, old_cam_mx=None):
        import cv2
        if len(P) > 0:
            if 0:
                # with this can tweak stopping criteria, the other version stops after 5 iterations
                pts = cv2.undistortPointsIter(P.reshape((-1, 1, 2)), cam_mx if old_cam_mx is None else old_cam_mx,
                                              np.array(dist_coefs), None, cam_mx,
                                              (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001))
            else:
                pts = cv2.undistortPoints(P.reshape((-1, 1, 2)), cam_mx if old_cam_mx is None else old_cam_mx,
                                          np.array(dist_coefs), None, cam_mx)
            return pts
        return P

    @staticmethod
    def _distort_own(P, dist_coefs, cam_mx=None, inv_cam_mx=None):
        """
        return distorted coords from undistorted ones based on dist_coefs
        if inv_cam_mx given, assume P are image coordinates instead of coordinates in camera frame
        """
        # from https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=projectpoints
        # TODO: investigate if and how s3 and s4 are used, above page formulas dont include them
        #       - also this documentation includes parameters tau_x and tau_y:
        #         https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga27865b1d26bac9ce91efaee83e94d4dd

        if inv_cam_mx is not None:
            P = np.hstack((P, np.ones((len(P), 1), dtype=P.dtype))).dot(inv_cam_mx[:2, :].T)
        else:
            P = P[:, :2] / P[:, 2:]

        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = np.pad(dist_coefs, (0, 12 - len(dist_coefs)), 'constant')

        R2 = np.sum(P[:, 0:2] ** 2, axis=1).reshape((-1, 1))
        R4 = R2 ** 2
        R6 = R4 * R2 if k3 or k6 else 0
        XY = np.prod(P, axis=1).reshape((-1, 1))
        KR = (1 + k1 * R2 + k2 * R4 + k3 * R6) / (1 + k4 * R2 + k5 * R4 + k6 * R6)

        Xdd = P[:, 0:1] * KR \
              + (2 * p1 * XY if p1 else 0) \
              + (p2 * (R2 + 2 * P[:, 0:1] ** 2) if p2 else 0) \
              + (s1 * R2 if s1 else 0) \
              + (s2 * R4 if s2 else 0)

        Ydd = P[:, 1:2] * KR \
              + (p1 * (R2 + 2 * P[:, 1:2] ** 2) if p1 else 0) \
              + (2 * p2 * XY if p2 else 0) \
              + (s1 * R2 if s1 else 0) \
              + (s2 * R4 if s2 else 0)

        img_P = np.hstack((Xdd, Ydd, np.ones((len(Xdd), 1), dtype=P.dtype)))
        if cam_mx is not None:
            img_P = img_P.dot(cam_mx[:2, :].T)
        return img_P
