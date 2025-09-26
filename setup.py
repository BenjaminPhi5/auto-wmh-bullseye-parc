 # setup for the trustworthai python library with required packages

from setuptools import find_packages, setup

setup(
    name="wmhparc",
    version="1.0",
    packages=find_packages(),

    install_requires = [
        "SimpleITK==2.4.0",
        "antspyx==0.5.4",
        "jupyterlab==4.3.4",
        "matplotlib==3.10.0",
        "matplotlib-inline==0.1.7",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scipy==1.14.1",
    ]
)
