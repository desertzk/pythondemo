from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [Extension("cadd", ["cadd.pyx"], libraries=["dl"])]
setup(
    ext_modules=cythonize(ext_modules),
)