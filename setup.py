import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pycle",
    version="1.2",
    author="Vincent Schellekens",
    author_email="vincent.schellekens@uclouvain.be",
    description="Python toolbox for Compressive Learning (machine learning from a sketch)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/schellekensv/pycle",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
