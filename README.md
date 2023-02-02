## Feature tracking statistics 
The codebase here focuses on calculating feature tracking performance metrics such as tracking length,
reprojection error and the rate of successfully triangulating tracked features.

The core is an implementation of Visual Odometry (VO) based on non-linear optimization (Bundle Adjustment, i.e. BA) 
implemented using Scipy. By default, the BA window size is set to a very high number (1,000,000) so that 
no marginalization is needed. The VO implementation support marginalization, however, it might impact the statistics as
it was not considered for this use.

VO is run frame-by-frame so that features that cannot be triangulated or that start to have too high reprojection errors
can be discarded, thus making it possible to measure geometrically valid tracking length.

Keyframe determination, feature detection and tracking are done by the client code, of which the script 
`trackingperf.py` is an example. The client code uses methods of `featstat.algo.odo.simple.SimpleOdometry` to
interact with the underlying VO and BA.

The codebase could also be used for calculating the performance of descriptor-based feature matching.

## Installation / dependencies

Create a conda environment and install additional packages using pip:

```
conda create -n featstat -c conda-forge python=3.10 pip numpy numba quaternion scipy opencv python-dateutil tqdm matplotlib
conda activate featstat
pip install opencv-contrib-python
```

An example camera calibration with a 8x6 checkerboard, each cell 44.4 mm wide, only use every 4th frame,
only estimate the first two of the distortion parameters (k1, k2) and redo calibration with those frames removed 
that had a reprojection error higher than 25 pixels:
```
python featstat\calibrate.py --path=data\calib-images -x=7 -y=5 -s=44.4 --skip=4 -n=2 --max-err=25
```

An example evaluation of Lukas-Kanade-Tomasi feature tracking using a set of frames from a video:
```
python featstat\trackingperf.py --data=data\example-video.mp4 --cam-w=1920 --cam-h=1080 \
                                --cam-fl-x=1575.5 --cam-pp-x=992.2 --cam-pp-y=522.2 \
                                --cam-dist "-0.10388" "0.07001" \
                                --first-frame=1600 --last-frame=1700 --skip=5
```

For full set of arguments, see source code or run scripts with the `--help` argument.

To evaluate a different feature tracking method, implement your own python script using `trackingperf.py` as an example.
