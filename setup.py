import pathlib
from typing import List, Tuple
from setuptools import find_packages, setup
import codecs
import os.path

def read(rel_path: str) -> str:
    """Read text from a file.

    Based on https://github.com/pypa/pip/blob/main/setup.py#L7.

    Args:
        rel_path (str): Relative path to the target file.

    Returns:
        str: Text from the file.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def requirements(rel_path: str) -> Tuple[List[str], List[str]]:
    """Parse pip-formatted requirements file.

    Args:
        rel_path (str): Path to a requirements file.

    Returns:
        Tuple[List[str], List[str]]: Extra package index URLs and setuptools-compatible package specifications.
    """
    packages = read(rel_path).splitlines()
    result = []
    for pkg in packages:
        if pkg.strip().startswith("#") or not pkg.strip():
            continue
        result.append(pkg)
    return result


requirements_file = "requirements.txt"
requirements_ = requirements(requirements_file)

setup(
    name="ullme",
    version="0.0.1",
    description="ULLME: A Unified Framework for Large Language Model Embeddings with Generation-Augmented Learning",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nlp-uoregon/ullme",
    author="NLP Group at the University of Oregon",
    author_email="hieum@uoregon.edu",
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements_,
)

