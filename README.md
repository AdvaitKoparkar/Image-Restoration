# PDE_Inpainting
Image Inpainting

## Progress
- [x] Base code for loading and mask drawing -- lib/loader.py lib/draw.py
- [x] Harmonic Solver -- models/harmonic.py
- [x] TV-L1 Solver -- models/tv.py
- [x] Exponential Solver -- models/exponential.py
- [x] Metrics Plot and Save -- plots saved in results
- [x] Animation -- video saved in results

## Files to run
* test.py - runs all three solvers on data
* plot.py - generates and saves plots

## Dependencies
* OpenCV
* tqdm
* numpy, scipy, matplotlib
* sklearn
