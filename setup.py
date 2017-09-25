# coding=utf-8
"""Setup file for distutils / pypi."""
try:
    from ez_setup import use_setuptools
    use_setuptools()
except ImportError:
    pass

from setuptools import setup, find_packages


setup(
    name='pyrenn',
    version='0.1',
    package_dir={'': 'python'},
    py_modules=['pyrenn'],
    license='GPL',
    author='Dennis Atabay',
    url='http://github.com/yabata/pyrenn',
    description='A recurrent neural network toolbox for Python and Matlab.',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
