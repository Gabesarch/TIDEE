#!/bin/sh

echo cloning rearrangement repos and setting task files...
git clone https://github.com/allenai/ai2thor-rearrangement.git
mv ai2thor-rearrangement rearrangement
cp utils/two_phase_tidee_base.py rearrangement/baseline_configs/two_phase/