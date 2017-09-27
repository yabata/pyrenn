# pyrenn

pyrenn is a [recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network) toolbox for Python and Matlab.

[![Documentation Status](https://readthedocs.org/projects/pyrenn/badge/?version=latest)](https://pyrenn.readthedocs.org/en/latest/)  [![DOI](https://zenodo.org/badge/18757/yabata/pyrenn.svg)](https://zenodo.org/badge/latestdoi/18757/yabata/pyrenn)

## Features

  * pyrenn allows to create a wide range of (recurrent) neural network configurations
  * It is very easy to create, train and use neural networks
  * It uses the [Levenberg–Marquardt algorithm](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) (a second-order Quasi-Newton optimization method) for training, which is much faster than first-order methods like [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent). In the matlab version additionally the [Broyden–Fletcher–Goldfarb–Shanno algorithm](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) is implemented
  * The python version is written in pure python and numpy and the matlab version in pure matlab (no toolboxes needed)
  * [Real-Time Recurrent Learning (RTRL) algorithm](http://www.mitpressjournals.org/doi/abs/10.1162/neco.1989.1.2.270#.VpDullJ1F3Q) and [Backpropagation Through Time (BPTT) algorithm](https://en.wikipedia.org/wiki/Backpropagation_through_time) are implemented and can be used to implement further training algorithms 
  * It comes with various examples which show how to create, train and use the neural network



## Installation

### Install with pip (python only)

From your command line, run:

```bash
pip install pyrenn
```

### Install manually
1. [download](https://github.com/yabata/pyrenn/archive/master.zip) or clone (with [git](http://git-scm.com/)) this repository to a directory of your choice.
2.	
	* Python: Copy the `pyrenn.py` file in the `python` folder to a directory which is already in python's search path or add the `python` folder to python's search path (sys.path) ([how to](http://stackoverflow.com/questions/17806673/where-shall-i-put-my-self-written-python-packages/17811151#17811151))
	* Matlab: Add the `matlab` folder to Matlab's search path ([how to](http://www.mathworks.com/help/matlab/matlab_env/add-remove-or-reorder-folders-on-the-search-path.html))
    
## Get Started
1. Run the given examples in the `examples` folder.
2. Read the [documenation](http://pyrenn.readthedocs.org) and create your own neural network


## Copyright

Copyright (C) 2016  Dennis Atabay

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
