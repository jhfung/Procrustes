# Procrustes

This repository contains the code to solve the generalized Procrustes problem 
for aligning point clouds, as well as the notebooks and scripts needed to 
reproduce the figures and results in our papers

- [Resampling and averaging coordinates on data](https://arxiv.org/abs/2408.01379) ("paper1")
- [Subsampling, aligning, and averaging circular coordinates in recurrent time series](https://arxiv.org/abs/2412.18515) ("paper2").

## Installation

Clone this repository to your work directory:

    git clone https://github.com/jhfung/Procrustes.git

The module can then be imported using:

    import Procrustes
    import Procrustes2
    import ResamplingCoords

You will need to Python installed, along with the packages `numpy`, `scipy`, and
`scikit-learn`.  Additionally, if you want to work with circular coordinates 
from persistent cohomology, `ripser` is required.  

## Usage

See the various Jupyter notebooks for example usage.  