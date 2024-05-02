import os
import re
from setuptools import setup, find_packages


current_file = os.path.dirname(os.path.realpath(__file__))


def get_version():
    with open(os.path.join(current_file, 'fa4gcf', '__init__.py'), 'r') as init_file:
        init_content = init_file.read()

    version = re.search(r'(?<=__version__ = \").*(?=\")', init_content)[0]
    return version


install_requires = [
    "recbole>=1.2.0",
    "numba",
    "wandb",
    "igraph",
    "optuna",
    "seaborn>=0.11.2",
    "torch",
    "torch-geometric",
    "torch-scatter",
    "torch-sparse"
]

setup_requires = []

extras_require = {}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: GPU :: NVIDIA CUDA",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
]

long_description = """
FA4GCF is a framework that extends the codebase of GNNUERS, an approach that leverages edge-level perturbations to
provide explanations of consumer unfairness and mitigate the latter as well. FA4GCF operates on GNN-based recommender
systems as GNNUERS, but the base approach was extensively modularized, re-adapted, and reworked to offer a simple
interface for including new tools. We focus on the adoption of GNNUERS for consumer unfairness mitigation, which
leverages a set of policies that sample the user or item set to restrict the perturbation process on specific and
valuable portions of the graph. In such a scenario, the graph perturbations are conceived only as edge additions,
i.e. user-item interactions. The edges are added only to the disadvantaged group (enjoying a lower recommendation
utility), such that the resulting overall utility of the group matches that of the advantaged group.
"""

setup(
    name="fa4gcf",
    version=get_version(),  # please remember to edit gnnuers/__init__.py in response, once updating the version
    description="A library to augment the graph of GNN-based recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jackmedda/FA4GCF",
    author="jackmedda",
    author_email="jackm.medda@gmail.com",
    packages=[package for package in find_packages() if package.startswith("fa4gcf")],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False,
    classifiers=classifiers,
)
