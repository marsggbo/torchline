from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name="torchline", # Replace with your own username
    version="0.2.4.3",
    author="marsggbo",
    author_email="csxinhe@comp.hkbu.edu.hk",
    description="A framework for easy to use Pytorch",
    long_description='''
    The ML developer can easily use this framework to implement your ideas. Our framework is built based on pytorch_lightning, 
    and the structures is inspired by detectron2''',
    long_description_content_type="text/markdown",
    url="https://github.com/marsggbo/torchline",
    packages=find_packages(exclude=("tests", "projects")),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
