#!/bin/bash
python3 -m featstat.trackingperf_ev --data="$HOME/data/day2-spot-0deg-side-30rpm/recording_2023-02-09_17-53-33.csv" \
                                    -v 2 --cam-w=640 --cam-h=480 --cam-fl-x=5.512e+02 --cam-fl-y=5.512e+02 \
                                    --cam-pp-x=3.165e+02 --cam-pp-y=2.408e+02 \
                                    --cam-dist "-9.771e-02 2.311e-01 9.028e-04 4.551e-04 0.0" \
                                    --tracking-interval=1 --keyframe-interval=0.03 \
				                            --first-event=3.0 --last-event=6.0
