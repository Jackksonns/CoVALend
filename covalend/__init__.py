"""
CoVALend: A Value-Alignment Pipeline for Compliant Micro-Lending
with Minimal-Intrusion Tabular-Text Fusion

This package implements the CoVALend model described in the research paper:
"CoVALend: A Value-Alignment Pipeline for Compliant Micro-Lending 
with Minimal-Intrusion Tabular-Text Fusion"

Author: Hongyi Zhou, Yalan Tan, Shouqiang Liu
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Hongyi Zhou"
__email__ = "h.zhou.23@abdn.ac.uk"

from covalend.models.finance_encoder import FinanceEncoder
from covalend.models.scsa import SCSA
from covalend.models.covalend_model import CovalendModel
from covalend.models.config import CovalendConfig

__all__ = [
    "FinanceEncoder",
    "SCSA",
    "CovalendModel",
    "CovalendConfig",
    "__version__",
]
