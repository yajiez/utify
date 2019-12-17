"""Utility functions and Monkeypatch Magics for Pandas DataFrame and Series
"""
import io
import logging
import os
from collections import defaultdict
from functools import partial
from typing import Collection

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import sqlalchemy as sqlalchemy
from IPython.display import display
from wasabi import Printer

from .base import make_listing
from .fastprogress import master_bar, progress_bar

msg = Printer()
logger = logging.getLogger(__name__)


def prepare_latex_table(table, save_file=None, overwrite=False, caption=None, label=None, fontsize='small'):
    """Prepare a ready to use latex table from pandas

    Args:
        table (str): latex table string produced by Pandas
        save_file (str): filepath to save the result
        overwrite (bool): if True, overwrite the table if exists
        caption (str): Optional caption of the table
        label (str): Optional label of the table
        fontsize (str): 'normal', 'large', 'x-large', 'small', 'x-small', 'tiny', etc.

    Returns:
        str: a ready to use latex table string
    """
    prefix = r'\begin{table}' + '\n' + r'\centering' + '\n' + r'\{}'.format(fontsize) + '\n'
    suffix = r'\end{table}' + '\n'
    if caption or label:
        tab_cap_label = ''
        if caption:
            tab_cap_label += r'\caption{' + caption + '}\n'
        if label:
            tab_cap_label += r'\label{' + label + '}\n'
        prefix += tab_cap_label
    latex_str = prefix + table + suffix
    if save_file:
        save_file = os.path.abspath(save_file)
        if os.path.exists(save_file) and (not overwrite):
            logger.warning("The target file already exists. Use overwrite=True to update.")
        else:
            with open(save_file, 'w+') as f:
                f.write(latex_str)
            logger.info(f"Table saved into {save_file}")
    return latex_str


# ============================== Pandas Series =====================================
def series_minmax(s: pd.Series):
    """Return a tuple of min and max"""
    return s.min(), s.max()


def normalize(s, **kwargs):
    return s / s.sum(**kwargs)


# ============================ Monkeypatch Magics ===================================
pd.Series.minmax = series_minmax
pd.Series.normalize = normalize


# ============================ Pandas DataFrame ====================================
def df_column_dtypes(df, dtype: str = None, target: str = None, max_cardinality: int = 10):
    """Return the columns grouped by their dtypes or for a specific dtype"""
    numeric_cols = []
    cat_cols = []
    for col in df.columns:
        if col == target:
            continue

        if (df[col].dtype in (int, float)) and (df[col].nunique() > max_cardinality):
            numeric_cols.append(col)
        else:
            cat_cols.append(col)

    dtypes = dict(numeric=numeric_cols, categorical=cat_cols)
    return dtypes[dtype] if dtype else dtypes


get_numeric_columns = partial(df_column_dtypes, dtype='numeric')
get_categorical_columns = partial(df_column_dtypes, dtype='categorical')


def parse_date_cols(df, cols, fmt="%d/%m/%Y"):
    """Parse the date columns in the given DataFrame

    Args:
        - df: pandas.DataFrame
        - cols: date columns to parse
        - fmt: format of the date str in cols

    Return:
        pd.DataFrame: each column in cols has been converted into datetime format
    """
    if isinstance(cols, str):
        cols = [cols]
    for col in cols:
        df[col] = pd.to_datetime(df[col], format=fmt)
    return df


def pd_to_psql(df, uri, tablename, if_exists="fail"):
    """Load pandas dataframe into a sql table using native postgres COPY FROM.

    Args:
        df (dataframe): pandas dataframe
        uri (str): postgres psycopg2 sqlalchemy database uri
        tablename (str): table to store data in
        if_exists (str): {‘fail’, ‘replace’, ‘append’}, default ‘fail’. See `pandas.to_sql()` for details

    Returns:
        bool: True if loader finished
    """
    if "psycopg2" not in uri:
        raise ValueError("psycopg2 uri is required for bulk import.")

    sql_engine = sqlalchemy.create_engine(uri)
    sql_cnxn = sql_engine.raw_connection()
    cursor = sql_cnxn.cursor()

    df[:0].to_sql(tablename, sql_engine, if_exists=if_exists, index=False)

    fbuf = io.StringIO()
    df.to_csv(fbuf, index=False, header=False, sep="\t")
    fbuf.seek(0)
    cursor.copy_from(fbuf, tablename, sep="\t", null="")
    sql_cnxn.commit()
    cursor.close()


def df_show(df: pd.DataFrame, display_shape=True, show_dtypes=True, show_index_dtype=True, show_missing_ratio=True, show_rows=True, n_rows=3, random=False):
    """Show useful information of a DataFrame

    Args:
        df (pd.DataFrame):
        display_shape (bool):
        show_dtypes (bool):
        show_index_dtype (bool):
        show_missing_ratio (bool):
        show_rows (bool):
        n_rows (int):
        random (bool):

    Returns:
        None

    """
    assert isinstance(df, pd.DataFrame), "df has to be an instance of pandas.DataFrame"
    if display_shape:
        msg.good("This DataFrame has {} rows and {} columns.".format(*df.shape))
    if show_index_dtype:
        msg.good(f"Index dtype: {df.index.dtype}")
    if show_dtypes:
        dtypes = df.dtypes.to_dict()
        msg.good("Column dtypes:")
        msg.text(make_listing(dtypes.items(), func=lambda i: f"{i[0]}: {i[1]}"))
    if show_missing_ratio:
        na_ratio = df.isna().mean().round(3).loc[lambda x: x > 0]
        if len(na_ratio) > 0:
            msg.good("Missing ratio:")
            msg.text(make_listing(na_ratio.items(), func=lambda i: f"{i[0]}: {i[1]}"))
    if show_rows:
        msg.good(f"Display the first {n_rows} rows:")
        if not random:
            display(df.head(n_rows))
        else:
            display(df.sample(n_rows))


def df_missing_ratio(
        df, plot=False, horizontal=False, to_frame=True, transpose=True, **plot_kwargs
):
    """Show missing ratios for all the columns of a DataFrame

    Args:
        df:
        plot:
        horizontal:
        to_frame:
        transpose:
        **plot_kwargs:

    Returns:

    """
    assert isinstance(df, pd.DataFrame)
    na_ratio = df.isna().mean().loc[lambda x: x > 0]
    if len(na_ratio) > 0:
        if "title" not in plot_kwargs:
            plot_kwargs["title"] = "Missing Ratio for Each Column"
        if plot and horizontal:
            na_ratio.plot.barh(**plot_kwargs)
            plt.xlim(0, 1)
        if plot and (not horizontal):
            na_ratio.plot.bar(**plot_kwargs)
            plt.ylim(0, 1)
        if to_frame:
            na_ratio = na_ratio.to_frame("missing_ratio")
        if to_frame and transpose:
            na_ratio = na_ratio.T
        return na_ratio
    else:
        msg.good("Congrats! There is no missing value in your DataFrame.")


def show_sparsity(matrix):
    """Show density and sparsity for binary matrix.

    Args:
        matrix:
    """
    matrix_density = matrix.mean().mean()
    print("Matrix density:", matrix_density)
    print("Matrix sparsity:", 1 - matrix_density)


def df_plot(
        df: pd.DataFrame,
        plot_cols=None,
        figure_types=None,
        sort_by_pct=True,
        show_pct=True,
        top_k=None,
        figsize=(9, 6),
        return_axs=True,
        title=None,
        xlabel=None,
        ylabel=None,
        **plot_kwargs,
):
    """Utility functions for quick plot

    Args:
        df:
        plot_cols:
        figure_types:
        sort_by_pct:
        show_pct:
        top_k:
        figsize:
        return_axs:
        title:
        xlabel:
        ylabel:
        **plot_kwargs:

    Returns:

    """
    assert plot_cols is not None, "You must provide the columns to plot."
    assert isinstance(plot_cols, Collection), "plot_cols should be a Collection."
    axs = []
    if figure_types:
        assert len(plot_cols) == len(figure_types)
    for col in plot_cols:
        n_unique = df[col].nunique()
        if not top_k:
            assert (
                    n_unique < 30
            ), "Only support columns with less than 30 unique values without using top_k."
        val_cnts = df[col].value_counts(sort=False)
        if sort_by_pct:
            val_cnts = val_cnts.sort_values(ascending=False)
        if top_k and top_k < n_unique:
            val_cnts = val_cnts.sort_values(ascending=False)
            val_cnts_top_k = val_cnts.iloc[:top_k]
            val_cnts_top_k.index = val_cnts_top_k.index.astype(str)
            val_cnts_top_k["Others"] = val_cnts.iloc[top_k:].sum()
            val_cnts = val_cnts_top_k
        display(val_cnts.to_frame().T)
        _, ax = plt.subplots(figsize=figsize)
        if show_pct:
            val_cnts = val_cnts / val_cnts.sum()
        val_cnts.plot.bar(ax=ax, **plot_kwargs)
        col_format_str = col.title().replace("_", " ")
        ax.set_title(title or f"Bar Chart: {col_format_str}")
        ax.set_xlabel(xlabel or col_format_str)
        ax.set_ylabel(ylabel or "No. of Observations")
        if show_pct:
            ax.set_ylabel(ylabel or "Pct. of Observations")
            for p in ax.patches:
                width, height = p.get_width(), p.get_height()
                x, y = p.get_xy()
                ax.annotate(
                    "{:.2%}".format(height), (x + 0.05 * width, y + height + 0.005)
                )
        axs.append(ax)
    if return_axs and (plot_cols is not None):
        return axs


def add_negative_examples(
        df,
        input_cols,
        sample_cols,
        neg_sampleing_ratio=2,
        shuffle=True,
        vocabs=None,
        target_col="label",
):
    """Negative Sampling
    """
    assert set(sample_cols) < set(input_cols)
    sample_cols_pos = {input_cols.index(col): col for col in sample_cols}
    choices = {
        col: vocabs[col] if vocabs and (col in vocabs) else df[col].unique()
        for col in sample_cols
    }
    df_dict = df[input_cols].copy().assign(**{target_col: 1}).set_index(input_cols).to_dict()[target_col]
    mbar = master_bar(range(neg_sampleing_ratio))
    result = defaultdict(list)
    for r in mbar:
        pbar = progress_bar(df_dict.keys(), parent=mbar)
        mbar.child.comment = f"negative sampling"
        for k in pbar:
            # Positive example
            if r == 0:
                for i, col in enumerate(input_cols):
                    result[col].append(k[i])
                result[target_col].append(1)
            # Negative example
            while True:
                # Generate a new random key
                random_key = tuple(
                    np.random.choice(choices[sample_cols_pos[idx]])
                    if idx in sample_cols_pos
                    else k[idx]
                    for idx in range(len(k))
                )
                # Add negative sample if the key doesn't appear in the data
                if random_key not in df_dict:
                    for i, col in enumerate(input_cols):
                        result[col].append(random_key[i])
                    result[target_col].append(0)
                    break
        mbar.first_bar.comment = f"round {r + 1} finished."

    result_df = pd.DataFrame(result)
    if shuffle:
        result_df = result_df.sample(len(result_df))
    return result_df.reset_index(drop=True)


# ============================ Monkeypatch Magics ===================================
pd.DataFrame.column_dtypes = df_column_dtypes
pd.DataFrame.show = df_show
pd.DataFrame.show_sparsity = show_sparsity
pd.DataFrame.qplot = df_plot
pd.DataFrame.missing_ratio = df_missing_ratio
pd.DataFrame.parse_date_cols = parse_date_cols

# Monkeypatch to support pandas -> arrow/feather conversion of nullable Integer type in `from_pandas`
# This will be removed when the stable release of pandas supports this feature
pd.arrays.IntegerArray.__arrow_array__ = lambda self, dtype: pa.array(self._data, mask=self._mask, type=dtype)
