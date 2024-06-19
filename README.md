**********************************************
# Image Analysis
**********************************************

## Description
Repo for the Data Intensive MPhil 2024 Image Analysis coursework.
Please see `report/main.pdf` for the full report.

Script outputs will be saved the `./outputs` directory.
## Installation
Set up
```
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/a8_img_assessment/vj279.git

cd vj279

conda env create -f environment.yml

conda activate vj279_image_analysis

docker build -t vj279_image_analysis .
```

## Usage

All scripts are designed to be run from the command line, the following are docker commands, if running outside docker make sure to activate the conda environment `ia_vj279` first.

All outputs are saved to the outputs directory.
### Running the scripts
Q1a - Lung CT segmentation
```
docker run -it -v $(pwd):/image_analysis ia_vj279 /bin/bash -c "source activate ia_vj279 && python src/q1a.py data/CT.png"
```
Q1b - Noisy Flower segmentation
```
docker run -it -v $(pwd):/image_analysis ia_vj279 /bin/bash -c "source activate ia_vj279 && python src/q1b.py data/noisy_flower.jpg"
```

Q1c - Coin segmentation
```
docker run -it -v $(pwd):/image_analysis ia_vj279 /bin/bash -c "source activate ia_vj279 && python src/q1c.py data/coins.png"
```

Q2a - Line Fitting
```
docker run -it -v $(pwd):/image_analysis ia_vj279 /bin/bash -c "source activate ia_vj279 && python src/q2a.py"
```

Q2b - Compressed Sensing Reconstruction
```
docker run -it -v $(pwd):/image_analysis ia_vj279 /bin/bash -c "source activate ia_vj279 && python src/q2b.py"
```

Q2c - Sparse Wavelet Reconstruction
```
docker run -it -v $(pwd):/image_analysis ia_vj279 /bin/bash -c "source activate ia_vj279 && python src/q2c.py data/river.jpg"
```

Q3a - Gradient Descent Convergence Rate
```
docker run -it -v $(pwd):/image_analysis ia_vj279 /bin/bash -c "source activate ia_vj279 && python src/q3a.py"    
```

Q3b - Learned Gradient Descent Optimisation
- Not runnable in the current environment, requires a GPU and Astra-Toolbox
- Currently having issue installing Astra-Toolbox in the docker container.

See script `src/coursework_LGD_filled_vj.ipynb` for a fully runnable notebook, simply upload to colab and run.

## Contributing

## License
Please see license.md

## Author
Vishal Jain
2024-03-12
