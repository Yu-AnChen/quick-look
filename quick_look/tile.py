import functools
import json
import math
import pathlib
import warnings

import dask.array as da
import joblib
import numpy as np
import ome_zarr.io
import ome_zarr.scale
import ome_zarr.writer
import tqdm
import zarr
from ashlar import reg

from . import ngff_metadata as nmetadata


def ignore_warnings_stage_position_unit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=reg.DataWarning,
                message="Stage coordinates' measurement unit is undefined"
            )
            return func(*args, **kwargs)
    return wrapper


@ignore_warnings_stage_position_unit
def tile_rcpnl(
    img_path,
    out_path,
    overwrite=False,
    positions_mode='trim',
    min_pixel_size=1,
    pyramid_downscale_factor=8,
    pyramid_level_max_size=1024,
    **metadata
):
    img_path = pathlib.Path(img_path)
    out_path = pathlib.Path(out_path)
    print('Processing', img_path.name)
    if out_path.exists() & (not overwrite):
        print(f"Aborted. {out_path} exists and `overwrite` is `False`")
        return

    assert positions_mode in ['trim', 'tile', 'stage']

    reader = reg.BioformatsReader(str(img_path))
    _metadata = reader.metadata
    num_channels = _metadata.num_channels
    tile_shape = _metadata.size
    pixel_size = _metadata.pixel_size
    dtype = _metadata.pixel_dtype

    metadata_positions = _metadata.positions

    tile_downsize_factor = 1
    if pixel_size < min_pixel_size:
        # use factor of 2, this shouldn't be necessary
        tile_downsize_factor = 2 ** np.ceil(
            np.log2(min_pixel_size / pixel_size)
        ).astype(int)
        print(
            f"pixel size {pixel_size} is smaller than `min_pixel_size` ({min_pixel_size});"
            f" will downsize the image by factor of {tile_downsize_factor}"
        )
    
    pixel_size *= tile_downsize_factor

    positions, tile_shape = _process_tile_positions(
        metadata_positions / tile_downsize_factor,
        positions_mode=positions_mode,
        tile_shape=np.ceil(tile_shape / tile_downsize_factor).astype(int)
    )

    mosaic_shape = (num_channels, *(positions.max(axis=0) + tile_shape))

    root = _make_ngff(
        out_path,
        shape=mosaic_shape,
        tile_shape=tile_shape,
        dtype=dtype,
        pixel_size=pixel_size,
        pyramid_downscale_factor=pyramid_downscale_factor,
        pyramid_level_max_size=pyramid_level_max_size,
        **metadata
    )

    print('Writing to', out_path)
    n_jobs = min(num_channels, joblib.cpu_count())
    _ = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
        joblib.delayed(_tile_channel)(
            reader,
            channel,
            root,
            positions,
            tile_downsize_factor=tile_downsize_factor,
            tile_shape=tile_shape,
            pyramid_downscale_factor=pyramid_downscale_factor,
            verbose=verbose
        )
        for channel, verbose in zip(range(num_channels), [True]+(num_channels-1)*[False]) 
    )
    return root


def _tile_channel(
    reader,
    channel,
    root,
    positions,
    tile_downsize_factor=1,
    tile_shape=None,
    pyramid_downscale_factor=8,
    verbose=False
):
    if tile_shape is None:
        tile_shape = reader.read(0, 0).shape
    tile_height, tile_width = tile_shape
    downscale_factor = pyramid_downscale_factor
    enum = enumerate(positions)
    if verbose:
        enum = enumerate(tqdm.tqdm(positions))
    intensity_maxs = np.zeros(len(positions))
    for idx, pp in enum:
        _img = reader.read(idx, channel)[::tile_downsize_factor, ::tile_downsize_factor]
        _img = _img[:tile_height, :tile_width]
        intensity_maxs[idx] = np.max(_img)
        for i, (_, aa) in enumerate(root.arrays()):
            rs, cs = np.floor(pp / downscale_factor**i).astype(int)
            img = _img[::downscale_factor**i, ::downscale_factor**i]
            h, w = img.shape
            aa[channel, rs:rs+h, cs:cs+w] = img
    return np.percentile(intensity_maxs, 90)


def _process_tile_positions(positions, positions_mode, tile_shape):
    
    mode = positions_mode
    positions = np.floor(positions - positions.min(axis=0)).astype(int)
    assert positions_mode in ['trim', 'tile', 'stage']
    rp, cp = positions.T
    if mode == 'stage':
        return positions, tile_shape
    
    # assume overlaps must < 50 % of tile shape
    # tile positions must form a dense grid
    rstep = rp[rp > 0.5 * tile_shape[0]].min()
    cstep = cp[cp > 0.5 * tile_shape[1]].min()

    if mode == 'trim':
        # use tiff compatible tile size
        tile_shape = np.round(np.divide([rstep, cstep], 16)).astype(int) * 16
    grid_positions = np.round(positions / [rstep, cstep]).astype(int) * np.array(tile_shape)
    return grid_positions, tile_shape


def _num_levels(base_shape, downscale_factor=8, max_size=1024):
    factor = max(base_shape) / max_size
    return math.ceil(math.log(factor, downscale_factor)) + 1


def _make_ngff(
    path,
    shape,
    tile_shape=None,
    dtype='uint16',
    pixel_size=1,
    pyramid_downscale_factor=8,
    pyramid_level_max_size=1024,
    **metadata
):
    store = ome_zarr.io.parse_url(path, mode="w").store
    root = zarr.group(store=store, overwrite=True)

    n_levels = _num_levels(
        shape,
        downscale_factor=pyramid_downscale_factor,
        max_size=pyramid_level_max_size
    )
    
    downscale_factor = pyramid_downscale_factor

    if tile_shape is None: tile_shape = (1024, 1024)

    data = da.zeros(shape, dtype=dtype, chunks=(1, *tile_shape))
    pyramid = [
        data[:, ::downscale_factor**i, ::downscale_factor**i] for i in range(n_levels)
    ]
    chunks = [level.chunksize for level in pyramid]
    coordinate_transformations = [
        [{
            'scale': [
                1,
                pixel_size * downscale_factor**i,
                pixel_size * downscale_factor**i
            ],
            'type': 'scale',

        }]
        # translation is needed for overlaying images at different resolution in
        # napari

        # NOTE: this is not the case anymore for napari v0.6.6 & napari-ome-zarr
        # v0.6.1. Commenting the following out
        # {
        #     'translation': [
        #         1,
        #         .5 * pixel_size * downscale_factor**i,
        #         .5 * pixel_size * downscale_factor**i
        #     ],
        #     'type': 'translation'
        # }]
        for i in range(n_levels)
    ]

    ome_zarr.writer.write_multiscale(
        pyramid=pyramid,
        group=root,
        axes='cyx',
        coordinate_transformations=coordinate_transformations,
        storage_options=[dict(chunks=cc) for cc in chunks],
        compute=False,
        **metadata
    )

    root.attrs['multiscales'] = nmetadata.update_pixel_size(
        root.attrs['multiscales']
    )
    return root


def _rcjob_channel_names(rcjob_path):
    with open(rcjob_path) as f:
        markers = json.load(f)['scanner']['assay']['biomarkers']
    return [
        '-'.join(mm.split('-')[:-1]) for mm in markers
    ]


def test():
    img_path = '/Users/yuanchen/Dropbox (HMS)/ashlar-dev-data/ashlar-rotation-data/3/LSP12961@20220309_150112_606256.rcpnl'
    return tile_rcpnl(
        img_path,
        out_path='/Users/yuanchen/projects/quick-look/quick_look/.dev/LSP12961@20220309_150112_606256-1.ome.zarr',
        overwrite=True,
        positions_mode='trim',
        min_pixel_size=1,
        pyramid_downscale_factor=4,
        pyramid_level_max_size=2048,
        metadata={'software': 'zzz', 'method': 'trim'},
        otherMetadata='b'
    )
    '''
    trim
    /
    ├── 0 (4, 17408, 13376) uint16
    ├── 1 (4, 2176, 1672) uint16
    └── 2 (4, 272, 209) uint16


    tile
    /
    ├── 0 (4, 18360, 14080) uint16
    ├── 1 (4, 2295, 1760) uint16
    └── 2 (4, 287, 220) uint16

    stage
    /
    ├── 0 (4, 17386, 13474) uint16
    ├── 1 (4, 2174, 1685) uint16
    └── 2 (4, 272, 211) uint16
    '''