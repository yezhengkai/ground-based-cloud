import warnings
import matplotlib as mpl


def get_rcParams(new_params=None, update=True, figsize="l"):
    """Get the specific rcParams used by this package.
    Parameters
    ----------
    new_params : dict, default None
        Dictionary of matplotlib's rcParams to manipulate plot setting.
    update : bool, dafault True
        The current rcPamams is updated when True,
        and the rcParams dictionary is returned when False.
    figsize : str, tuple or list, default 'l'
        's': squared paper size
            (4.8*4.8) inch * 300 dpi => (1440*1440) pixel
        'p': portrait paper size
            (4.8*6.4) inch * 300 dpi => (1440*1920) pixel
        'l': landscape paper size
            (6.4*4.8) inch * 300 dpi => (1920*1440) pixel
        tuple or list with 2 element:
            (width, height)
    Returns
    -------
    params : dict
        Dictionary of rcParams.
    """

    # 's': square
    # 'p': portrait
    # 'l': landscape
    if figsize == "s":
        w, h = 4.8, 4.8
    elif figsize == "p":
        w, h = 4.8, 6.4
    elif figsize == "l":
        w, h = 6.4, 4.8
    elif isinstance(figsize, (list, tuple)):
        if len(figsize) != 2:
            warnings.warn(
                "The length of figsize is wrong. Use default figsize", Warning
            )
            w, h = 6.4, 4.8
        else:
            w, h = figsize
    else:
        warnings.warn("figsize is wrong. Use default figsize", Warning)
        w, h = 6.4, 4.8

    # https://stackoverflow.com/questions/47782185/increment-matplotlib-string-font-size
    # sizes = ['xx-small','x-small','small','medium',
    #           'large','x-large','xx-large']
    params = {
        "axes.titlesize": "x-large",
        "axes.labelsize": "large",
        "figure.dpi": 100.0,
        "font.family": ["sans-serif"],
        "font.size": 12,
        "figure.figsize": [w, h],
        "figure.titlesize": "x-large",
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.fontsize": "medium",
        "mathtext.fontset": "stix",
        "savefig.dpi": 300.0,
        "xtick.labelsize": "small",
        "ytick.labelsize": "small",
    }

    if isinstance(new_params, dict):
        # https://stackoverflow.com/questions/8930915/append-dictionary-to-a-dictionary
        params.update(new_params)

    if update is True:
        mpl.rcParams.update(params)
        return
    else:
        return params
