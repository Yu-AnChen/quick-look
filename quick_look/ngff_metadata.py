import ome_zarr.io
import zarr
import numpy as np


def update_pixel_size(multiscale_metadata):
    ori = {**multiscale_metadata[0]}
    axes = [
        ax if ax['type'] != 'space'
        else {**ax, **{'unit': 'micrometer'}}
        for ax in ori['axes']
    ]
    updated = {
        **ori,
        **{'axes': axes}
    }
    return [updated]


def add_channel_metadata(
    path,
    channel_names,
    channel_colors=None,
    channel_contrasts=None,
):
    store = ome_zarr.io.parse_url(path, mode="a").store
    root = zarr.group(store=store)
    n_channels, _, _ = root[0].shape

    # channel names
    # list[str]
    if len(channel_names) != n_channels:
        print(
            f"Adding channel names aborted. {n_channels} channels in {path}"
            f" but {len(channel_names)} channel names were provided."
        )
    channel_names = [
        {'label': n} for n in channel_names
    ]

    # channel colors
    # list[str]
    default_colors = [
        '0000ff',
        '00ff00',
        'ffffff',
        'ff0000',
        'ff9900',
    ]
    if channel_colors is None:
        channel_colors = [
            default_colors[i%len(default_colors)] 
            for i in range(n_channels)
        ]
    assert len(channel_colors) == n_channels
    channel_colors = [
        {'color': c} for c in channel_colors
    ]

    # channel contrast
    # list[tuple(float, float)]
    dtype = root[0].dtype
    if channel_contrasts is None:
        if np.issubdtype(dtype, np.integer):
            dmin, dmax = np.iinfo(dtype).min, np.iinfo(dtype).max
            channel_contrasts = [{'window': {'start': dmin, 'end': dmax}}] * n_channels
        else:
            channel_contrasts = [None] * n_channels
    else:
        channel_contrasts = [
            {'window': {'start': dmin, 'end': dmax}}
            for dmin, dmax in channel_contrasts
        ]
    assert len(channel_contrasts) == n_channels

    channels = []
    for name, color, contrast in zip(
        channel_names, channel_colors, channel_contrasts
    ):
        if contrast is None:
            contrast = {}
        channels.append(
            {**name, **color, **contrast, **{'active': False}}
        )
    root.attrs["omero"] = {"channels": channels}
    return channels
