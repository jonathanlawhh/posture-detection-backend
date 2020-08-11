from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name='Sitting Posture Identifier API',
    ext_modules=cythonize("main.pyx"),
    zip_safe=False,
    include_dirs=[numpy.get_include()]
)
