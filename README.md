**********************************************
* image_analysis
**********************************************

## Description
Repo for the Data Intensive MPhil 2024 coursework.

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
## Contributing

## License

## Author
Vishal Jain
2024-06-09
