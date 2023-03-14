import os
import argparse
import warnings

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats

from featstat.algo import tools
from featstat.algo.model import Camera
from featstat.algo.odo.simple import SimpleOdometry

logger = tools.get_logger("main")

"""
 example:
   python3 -m featstat.trackingperf_ev --data=$HOME/data/day2-spot-0deg-side-30rpm/recording_2023-02-09_17-53-33.csv \
                                       -v=1 --cam-w=640 --cam-h=480 --cam-fl-x=5.512e+02 --cam-fl-y=5.512e+02 \
                                       --cam-pp-x=3.165e+02 --cam-pp-y=2.408e+02 \
                                       --cam-dist="-9.771e-02 2.311e-01 9.028e-04 4.551e-04 0.0" \
                                       --tracking-interval=1 --keyframe-interval=0.05
"""


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Run visual odometry on an event camera stream')
    parser.add_argument('--data', '-d', required=True, metavar='DATA', help='path to data')
    parser.add_argument('--haste-exec', default='haste/build/tracking_app_file', help='path to haste executable,'
                        'to run on WSL, use "wsl -d <distro> /path/to/featstat/haste/build/tracking_app_file"')
    parser.add_argument('--verbosity', '-v', type=int, default=2,
                        help='verbosity level (0-4, 0-1: text only, 2:+debug imgs, 3: +keypoints, 4: +poses)')
    parser.add_argument('--debug-out', '-o', metavar='OUT', help='path to the debug output folder')

    parser.add_argument('--cam-w', required=True, type=int, help='cam resolution width')
    parser.add_argument('--cam-h', required=True, type=int, help='cam resolution height')
    parser.add_argument('--cam-dist', default="0 0 0 0 0", help='cam distortion coeffs')
    parser.add_argument('--cam-fl-x', required=True, type=float, help='cam focal length x')
    parser.add_argument('--cam-fl-y', type=float, help='cam focal length y')
    parser.add_argument('--cam-pp-x', type=float, help='cam principal point x')
    parser.add_argument('--cam-pp-y', type=float, help='cam principal point y')

    parser.add_argument('--first-event', '-f', type=float, default=0, help='first event timestamp (default: 0.0)')
    parser.add_argument('--last-event', '-l', type=float, help='last event timestamp (default: None)')
    parser.add_argument('--skip', '-s', type=int, default=1, help='use only every xth event (default: 1)')
    parser.add_argument('--tracking-interval', type=float, default=1,
                        help='batch tracking interval in seconds (default: 1)')
    parser.add_argument('--max-track-dt', type=float, default=0.0075,
                        help='max tolerance in seconds when finding HASTE tracked keypoint location (default: 0.1)')
    parser.add_argument('--keyframe-interval', type=float, default=1, help='keyframe interval in seconds (default: 1)')
    parser.add_argument('--detection-win-n', type=int, default=2000,
                        help='feature detection window in events (default: 2000)')
    parser.add_argument('--detection-win-px', type=int, default=7,
                        help='feature detection window in pixels (default: 7)')
    args = parser.parse_args()
    args.cam_dist = list(map(float, args.cam_dist.split(" ")))

    logger.info("Initializing...")

    cam_dist_coefs = [0.0] * max(5, np.where(np.array(args.cam_dist) != 0)[0][-1] + 1)
    for i, c in enumerate(args.cam_dist):
        cam_dist_coefs[i] = c

    fl_y = args.cam_fl_y or args.cam_fl_x
    pp_x = args.cam_pp_x or args.cam_w / 2
    pp_y = args.cam_pp_y or args.cam_h / 2
    cam = Camera(
        args.cam_w,
        args.cam_h,
        dist_coefs=cam_dist_coefs,
        cam_mx=[[args.cam_fl_x, 0., pp_x],
                [0., fl_y, pp_y],
                [0., 0., 1.]],
    )
    odo = SimpleOdometry(cam, max_repr_err=1.5, max_ba_fun_eval=5000, min_2d2d_inliers=16, min_inliers=10,
                         verbose=args.verbosity, logger=logger)
    odo._track_image_height = 600

    if args.debug_out:
        odo._track_save_path = args.debug_out

    min_init_kp_count = round(1.333 * odo.min_2d2d_inliers / odo.inlier_ratio_2d2d)
    kp_ids, old_kp_obs = [None] * 2
    next_kp_id = 0
    ba_errs = []
    odo_states = []

    def ba_err_logger(frame_id, per_frame_ba_errs):
        per_frame_ba_errs = np.stack((per_frame_ba_errs[:, 0],
                                      np.linalg.norm(per_frame_ba_errs[:, 1:4], axis=1),
                                      np.linalg.norm(per_frame_ba_errs[:, 4:7], axis=1) / np.pi * 180), axis=1)
        ba_errs.append([frame_id, *np.nanmean(per_frame_ba_errs, axis=0)])
    odo.ba_err_logger = ba_err_logger

    event_loader = EventLoader(args.data, args.first_event, args.last_event)
    feature_track_cache = FeatureTrackCache(exec=args.haste_exec, max_dt=args.max_track_dt)

    i, t0, ini_fail_count, reset_count = 0, event_loader.ts, 0, 0
    try:
        while True:
            esurf = None

            if len(odo.state.keyframes) == 0:
                logger.info("")
                logger.info(f"Trying to detect initial features (min {min_init_kp_count} required, t={t0:.3f})")
                kp_ids, kp_obs, next_kp_id, t1, esurf = detect_features(feature_track_cache, event_loader, t0,
                                                                        next_kp_id, args.detection_win_n,
                                                                        args.tracking_interval, cam,
                                                                        min_kp_dist=args.detection_win_px,
                                                                        min_kps=min_init_kp_count)
                if len(kp_ids) > min_init_kp_count:
                    odo.first_frame(kp_ids, kp_obs)
                    t1 = t0 + args.keyframe_interval
            else:
                logger.info("")
                logger.info(f"New keyframe (t={t0:.3f})")

                kp_obs, rem_mask = track_features(odo, feature_track_cache, kp_ids, kp_obs, t0)

                rem_kp_ids = kp_ids[rem_mask]
                feature_track_cache.remove(rem_kp_ids)
                kp_ids, kp_obs = remove_features(kp_ids, kp_obs, rem_kp_ids)
                odo.expire_kps(rem_kp_ids)

                pos, ori, rem_kp_ids = odo.next_frame(kp_ids, kp_obs, is_keyframe=True)

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
                        t1 = t0
                    else:
                        logger.info("Failed to init, %d attempts left" % (4 - ini_fail_count))
                        ini_fail_count += 1
                        t1 = t0 + args.keyframe_interval
                else:
                    feature_track_cache.remove(rem_kp_ids)
                    kp_ids, kp_obs = remove_features(kp_ids, kp_obs, rem_kp_ids)

                    # handle newly detected keypoints
                    new_kp_ids, new_kp_obs, next_kp_id, _, esurf = detect_features(feature_track_cache, event_loader,
                                                                                   t0, next_kp_id, args.detection_win_n,
                                                                                   args.tracking_interval, cam,
                                                                                   min_kp_dist=args.detection_win_px,
                                                                                   prev_kps=kp_obs)
                    odo.new_kps(new_kp_ids, new_kp_obs)

                    kp_ids = np.concatenate((kp_ids, new_kp_ids), axis=0)
                    kp_obs = np.concatenate((kp_obs, new_kp_obs), axis=0)
                    t1 = t0 + args.keyframe_interval

            if args.verbosity > 1:
                if esurf is None:
                    events_d, _ = event_loader.load_interval(t0, n=args.detection_win_n)
                    esurf = event_surface(events_d, (cam.height, cam.width))

                v0 = np.nanmin(esurf)
                v1 = np.nanmax(esurf)
                v0 = 0 if np.isnan(v0) else v0
                v1 = 1 if np.isnan(v1) else v1
                img = ((esurf - v0) / (v1 - v0) * 255).astype(np.uint8)

                if len(odo.state.keyframes) > 0:
                    odo.state.keyframes[-1].image = img
                    odo.state.keyframes[-1].img_sc = 1.0
                    odo.state.keyframes[-1].orig_image = img

                if len(odo.state.keyframes) > 1:
                    rf, nf = odo.state.keyframes[-2:]
                    if nf.pose.post is not None:
                        odo._cv_draw_pts3d(nf, label=None, shape=(odo._track_image_height,) * 2)
                        odo._draw_bottom_bar()
                        odo._draw_tracks(nf, ref_frame=rf, pause=False)

                cv2.imshow('event surface', img)
                cv2.waitKey(100)

            event_loader.expire_older(t1)
            old_kp_ids, old_kp_obs, t0 = kp_ids, kp_obs, t1
    except EventLoaderOutOfRange as e:
        pass

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


class EventLoaderOutOfRange(Exception):
    pass


class EventLoader:
    def __init__(self, path, ts, te=None):
        raw = np.loadtxt(path, dtype=np.uint32, delimiter=',')
        self._t = (raw[:, 3] - raw[0, 3]).astype(np.float32) / 1e6
        self._xy = raw[:, 0:2].astype(np.uint16)
        self._b = raw[:, 2].astype(bool)
        (self._t, self._xy, self._b), _ = self.load_interval(ts, t1=te)
        self.ts = self._t[0]
        self.te = self._t[-1]
        logger.info("Loaded %d events (ts=%.3f, te=%.3f)" % (len(self._t), self.ts, self.te))

    def load_interval(self, t0, t1=None, dt=None, n=None, margin=1.0):
        assert (t1 is None) + (dt is None) + (n is None) >= 2, 'only give one of t1, dt, or n'
        idx0 = np.argmax(self._t > t0)
        idx1, _t1 = None, t1

        if dt is not None:
            _t1 = t0 + dt
        elif n is not None:
            if idx0 + n > len(self._t):
                raise EventLoaderOutOfRange(f"{idx0} + {n} > {len(self._t)}")
            idx1 = idx0 + n
            _t1 = self._t[idx1-1]

        if _t1 is None:
            idx1 = len(self._t)
            _t1 = self._t[-1]
        elif idx1 is None:
            idx1 = np.argmax(self._t >= _t1)

        ts, te = self._t[idx0], self._t[idx1-1]
        if t0 > te or idx1 < idx0:
            raise EventLoaderOutOfRange(f"{t0:.9f} > {te:.9f} or {idx1} < {idx0}")
        if ts < t0 or _t1 < ts:
            raise EventLoaderOutOfRange(f"{ts:.9f} < {t0:.9f} or {_t1:.9f} < {ts:.9f}")
        if (t1 is not None or dt is not None) and _t1 < te - margin:
            raise EventLoaderOutOfRange(f"{_t1:.9f} < {te:.9f} - {margin:.9f}")
        if n is not None and idx1 - idx0 < n:
            raise EventLoaderOutOfRange(f"{idx1} - {idx0} < {n}")

        return (self._t[idx0:idx1], self._xy[idx0:idx1, :], self._b[idx0:idx1]), _t1

    def expire_older(self, t):
        idx0 = max(0, np.argmax(self._t >= t) - 1)
        self._t = self._t[idx0:]
        self._xy = self._xy[idx0:, :]
        self._b = self._b[idx0:]


class FeatureTrackCache:
    def __init__(self, exec, max_dt):
        self.exec = exec
        self.max_dt = max_dt
        self._cache = {}

    def add(self, kp_ids, t, kp_obs):
        unique_ids = np.unique(kp_ids)
        for kp_id in unique_ids:
            self._cache[kp_id] = (t[kp_ids == kp_id], kp_obs[kp_ids == kp_id, :])

    def get_obs(self, kp_ids, t):
        kp_obs = np.ones((len(kp_ids), 2), dtype='f4') * np.nan
        for i, kp_id in enumerate(kp_ids):
            if kp_id in self._cache:
                idx = np.argmin(np.abs(self._cache[kp_id][0] - t))
                if np.abs(self._cache[kp_id][0][idx] - t) < self.max_dt:
                    kp_obs[i, :] = self._cache[kp_id][1][idx, :]
        return kp_obs

    def remove(self, kp_ids):
        for kp_id in kp_ids:
            if kp_id in self._cache:
                del self._cache[kp_id]


def detect_features(feature_track_cache, event_loader, t0, next_kp_id, detection_win_n, tracking_interval, cam,
                    prev_kps=None, min_kp_dist=7, max_kps=500, min_kps=None, harris=True):
    events_d, t1 = event_loader.load_interval(t0, n=detection_win_n)
    events_t, t2 = event_loader.load_interval(t0, dt=tracking_interval)
    h, w = cam.height, cam.width
    mask = 255 * np.ones((h, w), dtype=np.uint8)
    _prev_kps = [] if prev_kps is None else prev_kps.copy()
    kp_obs = []
    max_new_kps = max_kps - len(_prev_kps)

    # detector
    det = cv2.GFTTDetector_create(**{
            'maxCorners': max_new_kps,
            'qualityLevel': 0.04,           # default 0.01, was 0.04
            'minDistance': min_kp_dist,     # default 1
            'blockSize': 5,                 # default 3, was 5
            'useHarrisDetector': harris,    # default False, was True
            'k': 0.02,                      # default 0.04, was 0.02, related to Harris detector only,
                                            #   smaller k allows more line-like features
        })

    for n in (detection_win_n//4, detection_win_n//2, detection_win_n):
        esurf = event_surface((events_d[0][:n], events_d[1][:n, :]), (h, w))
        img = (~np.isnan(esurf)).astype(np.uint8) * 255

        if max_new_kps > 0:
            # dont detect close to current keypoints
            for x, y in _prev_kps:
                y0, y1 = max(0, int(y) - min_kp_dist), min(h, int(y) + min_kp_dist)
                x0, x1 = max(0, int(x) - min_kp_dist), min(w, int(x) + min_kp_dist)
                if x1 > x0 and y1 > y0:
                    mask[y0:y1, x0:x1] = 0

            kps = det.detect(img, mask=mask)
            _prev_kps = np.array([k.pt for k in kps], dtype='f4').reshape((-1, 2))
            kp_obs.append(_prev_kps)
            max_new_kps -= _prev_kps.shape[0]

    kp_obs = np.concatenate(kp_obs, axis=0)
    n_detected = kp_obs.shape[0]
    next_kp_id = next_kp_id + n_detected
    kp_ids = np.arange(next_kp_id - n_detected, next_kp_id)

    if n_detected >= (min_kps or 0):
        # call HASTE event tracking code
        calc_feature_tracks(feature_track_cache, events_t, t0, kp_ids, kp_obs, cam)
        kp_obs = feature_track_cache.get_obs(kp_ids, t1)
        I = ~np.isnan(kp_obs[:, 0])
        kp_ids, kp_obs = kp_ids[I], kp_obs[I, :]
        logger.info("New keypoints detected and successfully tracked: %d/%d" % (len(kp_obs), n_detected))
    else:
        logger.info("No new keypoints detected")

    return kp_ids, kp_obs, next_kp_id, t1, esurf


def calc_feature_tracks(feature_track_cache, events, t0, kp_ids, kp_obs, cam):
    # write events to file (format: t x y p)
    with open('tmp-events.txt', 'w') as f:
        for t, xy, p in zip(*events):
            f.write('%f %d %d %d\n' % (t, xy[0], xy[1], int(p)))

    # write cam intrinsics to file (single-line file with format: fx fy cx cy k1 k2 p1 p2 k3)
    with open('tmp-cam.txt', 'w') as f:
        dist = np.zeros(5)
        for i, c in enumerate(cam.dist_coefs[:5]):
            dist[i] = c
        f.write('%f %f %f %f %f %f %f %f %f\n' % (
            cam.cam_mx[0, 0], cam.cam_mx[1, 1], cam.cam_mx[0, 2], cam.cam_mx[1, 2], *dist))

    # write keypoints to file (format: t,x,y,theta,id)
    with open('tmp-kps.txt', 'w') as f:
        for kp_id, xy in zip(kp_ids, kp_obs):
            f.write('%f,%d,%d,%f,%d\n' % (t0, xy[0], xy[1], 0.0, kp_id))

    if os.path.exists('tmp-tracks.txt'):
        os.unlink('tmp-tracks.txt')

    # run HASTE
    #   tracker_type=correlation|haste_correlation|haste_correlation_star|haste_difference|haste_difference_star
    #   somewhat working ones: correlation (best), haste_correlation
    os.system(f'{feature_track_cache.exec} --events_file=tmp-events.txt '
              '                            --camera_params_file=tmp-cam.txt '
              f'                           --camera_size={cam.width}x{cam.height} '
              '                            --centered_initialization=false '
              '                            --tracker_type=correlation '
              '                            --seeds_file=tmp-kps.txt '
              '                            --output_file=tmp-tracks.txt '
              '                            --visualize=false'
              '                            --minloglevel=3')    # only show critical errors

    # read tracks from file (format: t,x,y,theta,id)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'loadtxt: Empty input file: "tmp-tracks.txt"')
        data = np.loadtxt('tmp-tracks.txt', dtype='f4', delimiter=',')

    if len(data) == 0:
        return

    data = np.atleast_2d(data)
    kp_ids = data[:, 4].astype('i4')
    t, xy = data[:, 0], data[:, 1:3]
    feature_track_cache.add(kp_ids, t, xy)


def event_surface(events, shape):
    esurf = np.ones(shape, dtype='f4') * np.nan
    for t, xy in zip(events[0], events[1]):
        esurf[xy[1], xy[0]] = t
    return esurf


def track_features(odo, feature_track_cache, kp_ids, old_kp_obs, t0):
    new_kp_obs = feature_track_cache.get_obs(kp_ids, t0)
    mask = ~np.isnan(new_kp_obs[:, 0])

    old_kp2d_norm = odo.cam.undistort(old_kp_obs)
    new_kp2d_norm = odo.cam.undistort(new_kp_obs)

    # extra sanity check on tracked points, set mask to false if keypoint quality too poor
    mask = odo.check_features(old_kp_obs.reshape((-1, 1, 2)), old_kp2d_norm.reshape((-1, 1, 2)),
                              new_kp_obs.reshape((-1, 1, 2)), new_kp2d_norm.reshape((-1, 1, 2)), mask)

    fails, total = np.sum(~mask), mask.size
    logger.info('Tracking: %d/%d' % (total - fails, total))
    return new_kp_obs, ~mask.flatten()


def remove_features(kp_ids, kp_obs, rem_kp_ids):
    assert len(kp_ids) == len(kp_obs), 'kp_ids and kp_obs need to have same length'
    if rem_kp_ids is None or len(rem_kp_ids.shape) == 0 or len(rem_kp_ids) == 0:
        return kp_ids, kp_obs

    rem_kp_ids = set(rem_kp_ids)
    idxs = [i for i, id in enumerate(kp_ids) if id not in rem_kp_ids]
    return kp_ids[idxs], kp_obs[idxs, :]


if __name__ == '__main__':
    main()
