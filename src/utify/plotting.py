import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL.Image import Image

logger = logging.getLogger(__name__)


def add_bar_value(ax, fmt=None, offset_x=0.15, offset_y=0.005, **kwargs):
    """Add the value of each bar above it in a bar plot

    Args:
        ax (mpl.axes.Axes): a matplotlib Axes object returned by most plot functions
        fmt (str): string format for the value, e.g. .2f for real number or .1% for percentage
        offset_x (float): used to calculate the offset along the xaxis based on the bar width
        offset_y (float): used to calculate the offset along the yaxis based on the max value of yaxis
        **kwargs: additional key words parameters passed to ax.text

    Returns:
        mpl.axes.Axes: an updated axes with bar value added
    """
    fmt = "{:" + fmt + "}" if fmt else "{}"
    ymax = ax.get_ylim()[1]
    for p in ax.patches:
        w, h = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.text(x + w * offset_x, y + h + ymax * offset_y, fmt.format(h), **kwargs)
    return ax


def add_barh_value(ax, fmt=None, offset_x=0.005, offset_y=0.15, **kwargs):
    """Add the value of each bar on the right in a barh plot

    Args:
        ax (mpl.axes.Axes): a matplotlib Axes object returned by most plot functions
        fmt (str): string format for the value, e.g. .2f for real number or .1% for percentage
        offset_x (float): used to calculate the offset along the xaxis based on the max value of xaxis
        offset_y (float): used to calculate the offset along the yaxis based on the bar height
        **kwargs: additional key words parameters passed to ax.text

    Returns:
        mpl.axes.Axes: an updated axes with bar value added to the right
    """
    fmt = "{:" + fmt + "}" if fmt else "{}"
    xmax = ax.get_xlim()[1]
    for p in ax.patches:
        w, h = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.text(w + xmax * offset_x, y + h * offset_y, fmt.format(w), **kwargs)
    return ax


def show_image(image, label=None, ax=None, figsize=None, show_axis=False, cmap=None, data_format='channels_first'):
    """Show image in either PIL format or in a numpy array format

    Args:
        image (PIL or np.ndarray): if image is a numpy array, the number of dimensions must be 2 or 3
        label (str): optional string to be used as title of the image
        ax (mpl.axes.Axes): Axes object to draw the plot onto, otherwise create a new Axes.
        figsize (tuple): a tupe of width, height
        show_axis (bool): switch on or off the axis
        cmap (str or mpl.colormap): colormap to select colors, If string, load colormap from matplotlib.
        data_format (str): 'channels_first' or 'channels_last', the number of channels must be 1, 3 or 4

    Returns:
        mpl.axes.Axes: the Axes object with the image drawn on it.
    """
    assert data_format in ('channels_first', 'channels_last')
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if not show_axis:
        ax.set_axis_off()

    if isinstance(image, Image):
        ax.imshow(image)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            ax.imshow(image, cmap=cmap or 'gray')
        elif image.ndim == 3:
            n_channel = image.shape[0] if data_format == 'channels_first' else image.shape[-1]
            if n_channel == 1:
                ax.imshow(image.squeeze(), cmap=cmap or 'gray')
            elif n_channel in (3, 4):
                if data_format == 'channels_first':
                    image = np.transpose(image, (1, 2, 0))
                ax.imshow(image)
            else:
                raise ValueError("The number of channel has to be 1, 3, or 4.")
        else:
            raise ValueError("""
                Image dimension can only be 2 or 3.
                Use show_images if you passed multiple images.
            """)
    else:
        raise ValueError("image has to be a PIL.Image or numpy array")

    if label is not None:
        ax.set_title(str(label))

    return ax


def plot_pca_result(explained_variance_ratio, markers=None, title=None, xlabel=None, ylabel=None, poi=None, loc='best', figsize=(7, 5), ax=None):
    """Plot the PCA explained variance ratio vs. number of components

    Args:
        explained_variance_ratio (list): a list of explained variance ratio for all the components
        markers (list): a list of number of componments to show markers
        title (str): optional title of the figure
        xlabel (str): optional xaxis label of the figure
        ylabel (str): optional yaxis label of the figure
        poi (int): optional additional legend for a specific number of components (i.e. position of interest)
        loc (int or str): optional parameters for the position of the legend
        figsize (tuple): a tuple of int for the figure size
        ax: a matplotlib Axes

    Returns:
        axes
    """
    explained_variance_ratio = np.array(explained_variance_ratio)
    n_comps = len(explained_variance_ratio)
    total_variance_ratio = sum(explained_variance_ratio)
    if markers is None:
        markers = []
        if n_comps > 10:
            markers.extend(list(range(1, 10)))
        else:
            markers = list(range(1, n_comps))
        if n_comps > 50:
            markers.extend(list(range(10, 50, 5)))
        else:
            markers.extend(list(range(10, n_comps, 5)))
        if n_comps > 300:
            markers.extend(list(range(50, 300, 50)))
        else:
            markers.extend(list(range(50, n_comps, 50)))
        if n_comps > 500:
            markers.extend(list(range(300, 500, 50)))
            markers.extend(list(range(500, n_comps, 50)))
        else:
            markers.extend(list(range(300, n_comps, 50)))
        markers.append(n_comps)

    if poi is not None:
        assert isinstance(poi, int) and (poi <= n_comps)
        if poi not in markers:
            markers.append(poi)
            markers.sort()

    logger.debug(f"List of components to plot: {markers}")
    pca_result = {(i + 1): ratio for i, ratio in enumerate(explained_variance_ratio.cumsum())}
    pca_result_subset = {k: v for k, v in pca_result.items() if k in markers}
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.plot(list(pca_result_subset.keys()), list(pca_result_subset.values()), '--bo')
    ax.set_title(title or f'Explained Variance Ratio {round(total_variance_ratio, 3)} using {n_comps} Components')
    ax.set_xlabel(xlabel or 'N Components')
    ax.set_ylabel(ylabel or 'Explained Variance Ratio')
    ax.set_ylim(0, 1.03)
    if poi:
        ax.vlines(poi, 0.15, pca_result_subset[poi], linestyles='--')
        ax.hlines(
            pca_result_subset[poi], 0, poi, linestyles='--',
            label=f'explained variance ratio using {poi} Components: {round(pca_result_subset[poi], 2):.0%}'
        )
        ax.legend(loc=loc)
    return ax


# ============================== Matplotlib Styles ==============================
mpl_style_serif = {
    'mathtext.fontset': 'cm',
    'font.family':      'serif',
    'font.serif':       'Iowan Old Style, serif'
}

mpl_style_sans = {
    'mathtext.fontset': 'cm',
    'font.family':      'sans-serif',
    'font.sans-serif':  'Google Sans, sans-serif'
}

available_mpl_styles = {
    'serif': mpl_style_serif,
    'sans':  mpl_style_sans
}


def get_available_mpl_style_names():
    """Return a list of names for the predifined matplotlib styles"""
    return list(available_mpl_styles.keys())


def get_mpl_style(name):
    """Return the key value pairs of the parameters for the given style name

    Args:
        name (str): name of a predefined matplotlib style

    Returns:
        dict: properties of the style with the given name
    """
    assert name in available_mpl_styles, f"{name} is not a defined style name."
    return available_mpl_styles[name]


def set_mpl_style(name):
    """Set the matplotlib style with the predefined properties of the given name

    Args:
        name (str):  name of a predefined matplotlib style

    Returns:
        None
    """
    style = get_mpl_style(name)
    plt.rcParams.update(style)
    plt.rcParams['pdf.fonttype'] = 42
    logger.debug(f"The following properties have been updated: {', '.join(style.keys())}")
