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
interact with the underlying VO and BA. There's also support for event-based feature tracking, `trackingperf_ev.py` is
an example of such.

The codebase could also be used for calculating the performance of descriptor-based feature matching.

## Installation / dependencies

You can install the package and dependencies using pip:
```
pip install git+https://github.com/oknuutti/featstat
```

Alternatively, you can clone the repository and then create a conda environment:
```
conda create -n featstat -c conda-forge python=3.10 pip numpy numba quaternion scipy opencv python-dateutil tqdm matplotlib
conda activate featstat
pip install opencv-contrib-python
```

If cloned, then can also use event-based feature tracking (`trackingperf_ev.py`) via
[HASTE](https://github.com/ialzugaray/haste) (please cite the authors if you use this for research). Feature detection
is done based on [Vasco et al., 2016](https://doi.org/10.1109/IROS.2016.7759610), where Harris corner detector is used
on a binarized event surface calculated using a fixed number of events.

To use HASTE, you need to build it from source:
```
# Install dependencies
sudo apt install build-essential cmake libgflags-dev libgoogle-glog-dev libopencv-dev

# Clone featstat and HASTE, which is a submodule
git clone https://github.com/oknuutti/featstat
cd featstat
git submodule update --init --recursive

# Build HASTE
mkdir haste/build && cd haste/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGUI=false		
make
```

If you use WSL, you can make a symlink to the Windows folder instead having to clones. However, in that case, 
update the submodules from the Linux side, otherwise there will be problems. You can still run `trackingperf_ev.py`
from the Windows-side by giving `--haste-exec="wsl ~/featstat/haste/build/tracking_app_file"` as an argument to it.


## Example Usage

You can calibrate your frame-based camera with `calibrate.py`. An example camera calibration with a 8x6 checkerboard,
each cell 44.4 mm wide, only use every 4th frame, only estimate the first two of the distortion parameters (k1, k2) and
redo calibration with those frames removed that had a reprojection error higher than 25 pixels:
```
python -m featstat.calibrate --path=data\calib-images -x=7 -y=5 -s=44.4 --skip=4 -n=2 --max-err=25
```

Here's an example evaluation of Lukas-Kanade-Tomasi feature tracking using a set of frames from a video:
```
python -m featstat.trackingperf --data=data\example-video.mp4 --cam-w=1920 --cam-h=1080 \
                                --cam-fl-x=1575.5 --cam-pp-x=992.2 --cam-pp-y=522.2 \
                                --cam-dist "-0.10388" "0.07001" \
                                --first-frame=1600 --last-frame=1700 --skip=5
```

Here's another example for event-based feature tracking with HASTE using events from a csv file:
```
python -m featstat.trackingperf_ev --haste-exec="wsl  ~/featstat/haste/build/tracking_app_file" \
                                   --data="data/recorded-events.csv" \
                                   --cam-w=640 --cam-h=480 \
                                   --cam-fl-x=5.512e+02 --cam-fl-y=5.512e+02 \
                                   --cam-pp-x=3.165e+02 --cam-pp-y=2.408e+02 \
                                   --cam-dist="-9.771e-02 2.311e-01 9.028e-04 4.551e-04 0.0" \
                                   --tracking-interval=0.5 --keyframe-interval=0.05556 \
                                   --first-event=5.556 --last-event=11.11 \
                                   -v=2
```

For full set of arguments, see source code or run scripts with the `--help` argument.

To evaluate a different feature tracking method, implement your own python script using `trackingperf.py` or 
`trackingperf_ev.py` as examples.
