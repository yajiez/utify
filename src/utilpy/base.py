# -*- coding: utf-8 -*-


def strfsec(seconds: int, ndigits=0):
    """Format the time in seconds into a time string

    Args:
        seconds (int): time in seconds
        ndigits (int): number of digits to round the seconds

    Returns:
        str: time string with a nice format
    """
    units = [("day", 24 * 60 * 60), ("hour", 60 * 60), ("minute", 60)]
    res, timestr = seconds, ''
    for unit, value in units:
        if res >= value:
            num = int(res // value)
            timestr += f"{num} {unit}{'s' if num > 1 else ''} "
            res = res % value
    timestr += f"{round(res, ndigits) if ndigits > 0 else int(res)} second{'s' if res > 1 else ''}"
    return timestr


def make_listing(iterable, indent=2, prefix='+', func=str):
    """Convert an iterable to a formatted listing.

    Args:
        iterable: an iterable of items to make the listing
        indent: space size before each item
        prefix: a prefix string to show before each item
        func: use this to customize the appearance of each item

    Returns:

    """
    sep = '\n' + ' ' * indent + prefix + ' '
    return sep + sep.join(func(x) for x in iterable)


def make_divider(text='', char='=', line_max=100, show=False):
    """Print a divider with a headline:
    ============================ Headline here ===========================

    Args:
        text (unicode): text of headline. If empty, only the chars are printed.
        char (unicode): Line character to repeat, e.g. =.
        line_max (int): max number of chars per line
        show (bool): Whether to print or not.

    Returns:
        str or None: if show, print the divider otherwise return the divider text
    """
    if len(char) != 1:
        raise ValueError(
            "Divider chars need to be one character long. "
            "Received: {}".format(char)
        )

    chars = char * (int(round((line_max - len(text))) / 2) - 2)
    text = " {} ".format(text) if text else ""
    text = f"\n{chars}{text}{chars}"

    if len(text) < line_max:
        text = text + char * (line_max - len(text))
    if show:
        print(text)
    else:
        return text
