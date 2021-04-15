from setuptools import setup, find_packages

setup(
    name="SegLossBias",
    version="0.1",
    author="Bingyuan Liu",
    description="Code for the paper :"
    "The hidden label-marginal biases of segmentation losses",
    packages=find_packages(),
    python_requries=">=3.8",
    install_requires=[
        # Please install pytorch-related libraries and opencv by yourself based on your environment
    ],
)
