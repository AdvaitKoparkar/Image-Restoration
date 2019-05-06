# PDE_Inpainting
Image Inpainting

## Progress
- [x] Base code for loading and mask drawing -- lib/loader.py lib/draw.py
- [x] Harmonic Solver -- models/harmonic.py
- [x] TV-L1 Solver -- models/tv.py
- [x] Exponential Solver -- models/exponential.py
- [x] Metrics Plot and Save -- plots saved in results
- [x] Animation -- video saved in results

## Demo
![comb](https://user-images.githubusercontent.com/21837899/57224618-09d3b800-6fd8-11e9-9456-23301fff069a.gif)

## Results
![loss](https://user-images.githubusercontent.com/21837899/57225082-418f2f80-6fd9-11e9-8a4b-73ac6271c947.png)
![res](https://user-images.githubusercontent.com/21837899/57225083-418f2f80-6fd9-11e9-8f05-d449139ba53f.png)
![res1](https://user-images.githubusercontent.com/21837899/57225084-418f2f80-6fd9-11e9-94d7-5dcadf17dfe5.png)

## Files to run
* test.py - runs all three solvers on data
* plot.py - generates and saves plots

## Dependencies
* OpenCV
* tqdm
* numpy, scipy, matplotlib
* sklearn
