from setuptools import setup, Extension
from Cython.Build import cythonize
ext_modules = [Extension("cadd", ["cadd.pyx", "add.c"])]
setup(ext_modules=cythonize(ext_modules))
