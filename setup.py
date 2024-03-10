#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Ship classification project using geometric deep learning",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
)

# author="",
# author_email="",
# url="https://github.com/user/project",