# -*- coding: utf-8 -*-

__author__ = "yajiez"
__copyright__ = "yajiez"

from .base import approximate_size
from .base import make_listing
from .base import make_divider
from .base import strfsec
from .base import download
from .base import unzip
from .base import Spinner

from .dataset import load_spam
from .dataset import load_esl_mixture

from .logger import get_logger

from .plotting import get_available_mpl_style_names
from .plotting import get_mpl_style
from .plotting import set_mpl_style
from .plotting import add_bar_value
from .plotting import add_barh_value
from .plotting import plot_pca_result

from . import dataframe
from .dataframe import prepare_latex_table

from .fastprogress import is_notebook
from .fastprogress import is_terminal
from .fastprogress import master_bar
from .fastprogress import progress_bar
