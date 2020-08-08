#!/bin/env python
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rollout",
    version="0.0.8",
    author="Philipp Rothenhäusler",
    author_email="philipp.rothenhaeusler@gmail.com",
    description="A multi-agent rollout implementation targeted on the Flatland Neurips 2020 challenge.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
