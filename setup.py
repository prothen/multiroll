#!/bin/env python
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="multiroll",
    version="0.0.8",
    author="Philipp Rothenhäusler",
    author_email="philipp.rothenhaeusler@gmail.com",
    description="The multi-agent rollout Python implementation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prothen/multiroll.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
