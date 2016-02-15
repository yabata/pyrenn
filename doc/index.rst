.. pyrenn documentation master file, created by
   sphinx-quickstart on Mon Jan 11 11:43:04 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. module:: pyrenn

pyrenn: A recurrent neural network toolbox for python and matlab
================================================================

:Maintainer: Dennis Atabay, <dennis.atabay@tum.de>
:Organization: `Institute for Energy Economy and Application Technology`_,
               Technische Universität München
:Version: |version|
:Date: |today|
:Copyright:
  This documentation is licensed under a `Creative Commons Attribution 4.0 
  International`__ license.

.. __: http://creativecommons.org/licenses/by/4.0/


.. image:: https://zenodo.org/badge/18757/yabata/pyrenn.svg
   :target: https://zenodo.org/badge/latestdoi/18757/yabata/pyrenn

Contents
--------

This documentation contains the following pages:

.. toctree::
   :maxdepth: 1

   create
   train
   use
   save
   examples
   

Features
--------
* pyrenn allows to create a wide range of (recurrent) neural network configurations
* It is very easy to create, train and use neural networks
* It uses the `Levenberg–Marquardt algorithm`_ (a second-order Quasi-Newton optimization method) for training, which is much faster than first-order methods like `gradient descent`_. In the matlab version additionally the `Broyden–Fletcher–Goldfarb–Shanno algorithm`_ is implemented
* The python version is written in pure python and numpy and the matlab version in pure matlab (no toolboxes needed)
* `Real-Time Recurrent Learning (RTRL) algorithm`_ and `Backpropagation Through Time (BPTT) algorithm`_ are implemented and can be used to implement further training algorithms 
* It comes with various examples which show how to create, train and use the neural network


Get Started
-----------

1. `download`_ or clone (with `git`_) this repository to a directory of your choice.
2.	* Python: Copy the ``pyrenn.py`` file in the ``python`` folder to a directory which is already in python's search path or add the ``python`` folder to python's search path (sys.path) (`how to`__)
	* Matlab: Add the ``matlab`` folder to Matlab's search path (`how to`_)
3. Run the given examples in the `examples` folder.
4. Create your own neural network.


.. __: http://stackoverflow.com/questions/17806673/where-shall-i-put-my-self-written-python-packages/17811151#17811151) 
.. _how to: http://www.mathworks.com/help/matlab/matlab_env/add-remove-or-reorder-folders-on-the-search-path.html


  
Dependencies (Python)
------------

* `numpy`_ for mathematical operations 
* `pandas`_ only for using the examples 


   
.. _Institute for Energy Economy and Application Technology: http://www.ewk.ei.tum.de/
.. _numpy: http://www.numpy.org/
.. _pandas: https://pandas.pydata.org
.. _Levenberg–Marquardt algorithm: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
.. _gradient descent: https://en.wikipedia.org/wiki/Gradient_descent
.. _Broyden–Fletcher–Goldfarb–Shanno algorithm: https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
.. _Real-Time Recurrent Learning (RTRL) algorithm: http://www.mitpressjournals.org/doi/abs/10.1162/neco.1989.1.2.270#.VpDullJ1F3Q
.. _Backpropagation Through Time (BPTT) algorithm: https://en.wikipedia.org/wiki/Backpropagation_through_time
.. _download: https://github.com/yabata/pyrenn/archive/master.zip
.. _git: http://git-scm.com/
