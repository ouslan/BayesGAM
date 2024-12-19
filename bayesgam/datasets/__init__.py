"""
GAM datasets
"""
# -*- coding: utf-8 -*-

from bayesgam.datasets.load_datasets import mcycle
from bayesgam.datasets.load_datasets import coal
from bayesgam.datasets.load_datasets import faithful
from bayesgam.datasets.load_datasets import wage
from bayesgam.datasets.load_datasets import trees
from bayesgam.datasets.load_datasets import default
from bayesgam.datasets.load_datasets import cake
from bayesgam.datasets.load_datasets import hepatitis
from bayesgam.datasets.load_datasets import toy_classification
from bayesgam.datasets.load_datasets import head_circumference
from bayesgam.datasets.load_datasets import chicago
from bayesgam.datasets.load_datasets import toy_interaction

__all__ = [
    'mcycle',
    'coal',
    'faithful',
    'trees',
    'wage',
    'default',
    'cake',
    'hepatitis',
    'toy_classification',
    'head_circumference',
    'chicago',
    'toy_interaction',
]
