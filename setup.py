import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="faix-carusyte", # Replace with your own username
    version="0.0.1",
    author="carusyte",
    author_email="carusyte@163.com",
    description="Finance data estimation using Tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agux/faix",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)