import argparse
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats

from featstat.algo import tools
from featstat.algo.model import Camera
from featstat.algo.odo.simple import SimpleOdometry

logger = tools.get_logger("main")


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Run visual odometry on a set of images')
    parser.add_argument('--data', '-d', required=True, metavar='DATA', help='path to data')
    parser.add_argument('--verbosity', '-v', type=int, default=2,
                        help='verbosity level (0-4, 0-1: text only, 2:+debug imgs, 3: +keypoints, 4: +poses)')
    parser.add_argument('--debug-out', '-o', metavar='OUT', help='path to the debug output folder')

    parser.add_argument('--cam-w', required=True, type=int, help='cam resolution width')
    parser.add_argument('--cam-h', required=True, type=int, help='cam resolution height')
    parser.add_argument('--cam-dist', type=float, nargs='*', help='cam distortion coeffs')
    parser.add_argument('--cam-fl-x', required=True, type=float, help='cam focal length x')
    parser.add_argument('--cam-fl-y', type=float, help='cam focal length y')
    parser.add_argument('--cam-pp-x', type=float, help='cam principal point x')
    parser.add_argument('--cam-pp-y', type=float, help='cam principal point y')
    parser.add_argument('--half-scale', action='store_true', help='downscale images by 2 (default: False)')

    parser.add_argument('--first-frame', '-f', type=int, default=0, help='first frame (default: 0; -1: hardcoded value)')
    parser.add_argument('--last-frame', '-l', type=int, help='last frame (default: None; -1: hardcoded end)')
    parser.add_argument('--skip', '-s', type=int, default=1, help='use only every xth as a keyframe (default: 1)')
    parser.add_argument('--detection-win-px', type=int, default=7,
                        help='feature detection window in pixels (default: 7)')
    args = parser.parse_args()

    cam_dist_coefs = [0.0] * max(5, np.where(np.array(args.cam_dist) != 0)[0][-1] + 1)
    for i, c in enumerate(args.cam_dist):
        cam_dist_coefs[i] = c

    fl_y = args.cam_fl_y or args.cam_fl_x
    pp_x = args.cam_pp_x or args.cam_w / 2
    pp_y = args.cam_pp_y or args.cam_h / 2
    sc = 2 if args.half_scale else 1
    cam = Camera(
        args.cam_w // sc,
        args.cam_h // sc,
        dist_coefs=cam_dist_coefs,
        cam_mx=[[args.cam_fl_x / sc, 0., pp_x / sc],
                [0., fl_y / sc, pp_y / sc],
                [0., 0., 1.]],
    )
    odo = SimpleOdometry(cam, max_repr_err=1.5, max_ba_fun_eval=5000, min_2d2d_inliers=16, min_inliers=10,
                         verbose=args.verbosity, logger=logger)
    odo._track_image_height = 600

    if args.debug_out:
        odo._track_save_path = args.debug_out

    min_init_kp_count = round(1.333 * odo.min_2d2d_inliers / odo.inlier_ratio_2d2d)
    kp_ids, next_kp_id, old_img, old_kp_obs = [None] * 4
    ba_errs = []
    odo_states = []
    i, ini_fail_count, reset_count = 0, 0, 0

    def ba_err_logger(frame_id, per_frame_ba_errs):
        per_frame_ba_errs = np.stack((per_frame_ba_errs[:, 0],
                                      np.linalg.norm(per_frame_ba_errs[:, 1:4], axis=1),
                                      np.linalg.norm(per_frame_ba_errs[:, 4:7], axis=1) / np.pi * 180), axis=1)
        ba_errs.append([frame_id, *np.nanmean(per_frame_ba_errs, axis=0)])
    odo.ba_err_logger = ba_err_logger

    image_loader = create_image_loader(args.data, args.first_frame, args.last_frame, scale=1/sc)

    for orig_img in image_loader:
        if len(orig_img.shape) == 3 and orig_img.shape[2] > 1:
            img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        else:
            img = orig_img
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
        is_keyframe = (i % args.skip) == 0

        if is_keyframe:
            msg = "=== Keyframe #%d added ===" % (len(odo.state.keyframes) + 1,)
            logger.info(" " * len(msg))
            logger.info("=" * len(msg))
            logger.info(msg)
        else:
            msg = "--- new frame ---"
            logger.info(" " * len(msg))
            logger.info(msg)

        if i == 0:
            logger.info("")
            logger.info(f"Trying to detect initial features (min {min_init_kp_count} required)")
            kp_obs = detect_features(img, min_kp_dist=args.detection_win_px)
            next_kp_id = len(kp_obs)
            kp_ids = np.arange(next_kp_id)
            odo.first_frame(kp_ids, kp_obs)
        else:
            kp_obs, rem_mask = track_features(odo, old_img, old_kp_obs, img, None)
            rem_kp_ids = kp_ids[rem_mask]
            kp_ids, kp_obs = remove_features(kp_ids, kp_obs, rem_kp_ids)
            odo.expire_kps(rem_kp_ids)

            pos, ori, rem_kp_ids = odo.next_frame(kp_ids, kp_obs, is_keyframe=is_keyframe)

            n_3d_pts = np.sum([p.active for p in odo.state.map3d.values()])
            logger.info(f"active 3d points: {n_3d_pts}")

            if pos is None or n_3d_pts < odo.min_inliers:
                if odo.state.initialized or ini_fail_count > 3 \
                        or len(kp_ids) < odo.min_2d2d_inliers / odo.inlier_ratio_2d2d:
                    logger.info("Odometry failed, resetting...")
                    ini_fail_count = 0
                    reset_count += 1
                    odo_states.append(odo.state)
                    odo.reset()
                    i = -1
                else:
                    logger.info("Failed to init, %d attempts left" % (4 - ini_fail_count))
                    ini_fail_count += 1
            else:
                kp_ids, kp_obs = remove_features(kp_ids, kp_obs, rem_kp_ids)

                if is_keyframe:
                    new_kp_obs = detect_features(img, kp_obs, min_kp_dist=args.detection_win_px)
                    new_kp_ids = np.arange(next_kp_id, next_kp_id + len(new_kp_obs))
                    odo.new_kps(new_kp_ids, new_kp_obs)

                    kp_ids = np.concatenate((kp_ids, new_kp_ids), axis=0)
                    kp_obs = np.concatenate((kp_obs, new_kp_obs), axis=0)
                    next_kp_id += len(new_kp_obs)

        if is_keyframe and args.verbosity > 1:
            if len(odo.state.keyframes) > 0:
                odo.state.keyframes[-1].image = img
                odo.state.keyframes[-1].img_sc = 1.0
                odo.state.keyframes[-1].orig_image = orig_img

            if len(odo.state.keyframes) > 1:
                rf, nf = odo.state.keyframes[-2:]
                odo._cv_draw_pts3d(nf, label=None, shape=(odo._track_image_height,) * 2)
                odo._draw_bottom_bar()
                odo._draw_tracks(nf, ref_frame=rf, pause=False)

        i += 1
        old_img, old_kp_obs = img, kp_obs

    # calculate final stats
    kp_ids, succ_rate, track_len, repr_err, keyframe_count = [], [], [], [], []
    for state in odo_states + [odo.state]:
        keyframe_count.append(len(state.keyframes))
        if len(state.keyframes) >= 20:
            odo.state = state
            _pos, _ori, _kp_ids, _kp_3d, _succ_rate, _track_len, _repr_err, _repr_err_ids = odo.tracking_stats()
            kp_ids.append(_kp_ids)
            succ_rate.append(_succ_rate)
            track_len.append(_track_len)
            repr_err.append(_repr_err)

    if len(kp_ids) == 0:
        logger.info("Not enough successfully estimated poses to calculate feature tracking statistics.")
        return

    all_kps = np.array([len(kp)/sr for kp, sr in zip(kp_ids, succ_rate)])
    succ_rate = np.sum(np.array(succ_rate) * all_kps) / np.sum(all_kps)     # weighted average
    kp_ids, track_len, repr_err = map(np.concatenate, (kp_ids, track_len, repr_err))

    n_bins = 6
    hist = np.bincount(track_len, minlength=n_bins)
    hist[n_bins-1] = np.sum(hist[n_bins-1:])
    track_counts = hist[1:n_bins].copy()
    hist = 100 * hist[1:n_bins] / np.sum(hist[1:n_bins])
    hist_lbls = list(map(str, range(0, n_bins-2))) + ['%d or more' % (n_bins-2,)]

    logger.info("")
    logger.info("")
    logger.info("=== RESULTS ===")
    logger.info("")
    logger.info("Odometry reset count due to VO failure: %d" % reset_count)
    logger.info("Consecutive successful keyframes (max, median, mean): %d, %.1f, %.1f" % (
        np.max(keyframe_count), np.median(keyframe_count), np.mean(keyframe_count)))
    logger.info("Number of 3d points estimated: %d" % (len(kp_ids),))
    logger.info("Success rate: %.3f%%" % (succ_rate*100,))
    logger.info("Track lengths (%s): %s" % (
        ', '.join(hist_lbls), ', '.join(['%d' % p for p in track_counts]),))
    logger.info("Track length %% (%s): %s" % (
        ', '.join(hist_lbls), ', '.join(['%.2f' % p for p in hist]),))
    logger.info("Track length percentiles (15.9, 50, 84.1): %s" % (
        ', '.join(['%.1f' % p for p in np.percentile(track_len-1, (15.9, 50, 84.1))]),))  # gaussian -sigma, mode, sigma
    logger.info("Re-projection error percentiles (50, 84.1, 97.7): %s" % (
        ', '.join(['%.3f' % p for p in np.percentile(repr_err, (50, 84.1, 97.7))]),))   # gaussian mode, sigma, 2*sigma

    plt.figure(1)
    plt.bar(hist_lbls, hist)
    plt.title("Track length distribution")

    plt.figure(2)
    err_de = stats.gaussian_kde(repr_err)
    x = np.arange(0., 5, .01)
    plt.plot(x, err_de(x))
    plt.title("Re-projection error density curve")
    plt.show()


def create_image_loader(path, first_frame, last_frame, scale=1.0):

    def video_loader():
        cap = cv2.VideoCapture(path)
        ret, f_id = True, 0

        while cap.isOpened() and ret:
            f_id += 1
            ret, img = cap.read()
            if first_frame is not None and f_id < first_frame:
                continue
            elif last_frame is not None and f_id > last_frame:
                break
            if scale != 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale)
            yield img
        cap.release()

    def image_loader():
        for f_id, img_file in enumerate(sorted(os.listdir(path))):
            img = cv2.imread(os.path.join(path, img_file), cv2.IMREAD_UNCHANGED)
            if first_frame is not None and f_id < first_frame:
                continue
            elif last_frame is not None and f_id > last_frame:
                break
            if scale != 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale)
            yield img

    return video_loader() if os.path.isfile(path) else image_loader()


def detect_features(img, prev_kps=None, min_kp_dist=7, max_kps=500, refine=True, harris=False):
    # dont detect close to current keypoints
    prev_kps = [] if prev_kps is None else prev_kps
    max_new_kps = max(0, max_kps - len(prev_kps))
    if max_new_kps == 0:
        np.zeros((0, 2), dtype='f4')

    h, w = img.shape
    mask = 255 * np.ones((h, w), dtype=np.uint8)
    for x, y in prev_kps:
        y0, y1 = max(0, int(y) - min_kp_dist), min(h, int(y) + min_kp_dist)
        x0, x1 = max(0, int(x) - min_kp_dist), min(w, int(x) + min_kp_dist)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 0

    # detector
    det = cv2.GFTTDetector_create(**{
            'maxCorners': max_new_kps,
            'qualityLevel': 0.02,           # default 0.01, was 0.02
            'minDistance': min_kp_dist,     # default 1
            'blockSize': 7,                 # default 3, was 7
            'useHarrisDetector': harris,    # default False, was False
            'k': 0.02,                      # default 0.04, was 0.02, related to Harris detector only,
                                            #   smaller k allows more line-like features
        })
    kps = det.detect(img, mask=mask)
    kp_obs = np.array([k.pt for k in kps], dtype='f4').reshape((-1, 1, 2))

    if refine:
        win_size = (5, 5)
        zero_zone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
        kp_obs = cv2.cornerSubPix(img, kp_obs, win_size, zero_zone, criteria)

    logger.info("New keypoints detected: %d" % len(kp_obs))
    return kp_obs.reshape((-1, 2))


def track_features(odo, img0, kp0, img1, kp1_ini=None):
    new_kp2d, mask, err = cv2.calcOpticalFlowPyrLK(img0, img1, kp0, kp1_ini, **{
        'winSize': (32, 32),  # was (32, 32)
        'maxLevel': 4,  # was 4
        'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.2),  # was: ..., 10, 0.05)
        'minEigThreshold': 0.0001,
    }, **({} if kp1_ini is None else {'flags': cv2.OPTFLOW_USE_INITIAL_FLOW}))
    if new_kp2d is None:
        return np.zeros_like(kp0), np.ones(len(kp0), dtype=bool)
    assert len(kp0) == len(new_kp2d), 'new kp arr diff size than prev arr'
    assert len(kp0) == len(mask), 'mask arr diff size than prev arr'

    # extra sanity check on tracked points, set mask to false if keypoint quality too poor
    kp0_norm = odo.cam.undistort(kp0)
    new_kp2d_norm = odo.cam.undistort(new_kp2d)
    mask = odo.check_features(kp0, kp0_norm, new_kp2d, new_kp2d_norm, mask)

    logger.info('Tracking: %d/%d' % (int(np.sum(mask)), len(mask)))
    return new_kp2d, np.logical_not(mask.flatten())


def remove_features(kp_ids, kp_obs, rem_kp_ids):
    assert len(kp_ids) == len(kp_obs), 'kp_ids and kp_obs need to have same length'
    if rem_kp_ids is None or len(rem_kp_ids.shape) == 0 or len(rem_kp_ids) == 0:
        return kp_ids, kp_obs

    rem_kp_ids = set(rem_kp_ids)
    idxs = [i for i, id in enumerate(kp_ids) if id not in rem_kp_ids]
    return kp_ids[idxs], kp_obs[idxs, :]


if __name__ == '__main__':
    main()
