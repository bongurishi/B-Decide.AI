"""
NLP package for B-Decide AI
Now includes batch explanation generation with parallel processing
"""

from .explainer import ChurnExplainer
from .batch_explainer import BatchChurnExplainer

__all__ = ['ChurnExplainer', 'BatchChurnExplainer']
