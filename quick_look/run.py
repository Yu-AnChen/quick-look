import datetime
import pathlib
import sys
import time

from . import ngff_metadata, tile, util, path_configs
from . import __version__ as _version


def _process_path(filepath):
    filepath = util.get_path(filepath)
    rcpnl, rcjob = None, None
    if not filepath.exists(): return
    if filepath.is_dir():
        rcpnl = next(filepath.glob('*.rcpnl'), None)
        rcjob = next(filepath.glob('*.rcjob'), None)
    if filepath.suffix == '.rcpnl':
        rcpnl = filepath
    return rcpnl, rcjob


def _to_log(log_path, img_path, img_shape, time_diff):
    pathlib.Path(log_path).parent.mkdir(exist_ok=True, parents=True)
    with open(log_path, 'a') as log_file:
        log_file.write(
            f"{datetime.timedelta(seconds=time_diff)} | {img_path.name} | {img_shape} \n"
        )


def process_directory(
    input_dir,
    output_dir,
    process_kwargs=None,
    inspect_focus=False
):
    input_dir = util.get_path(input_dir)
    if not input_dir.exists():
        print(f"Input directory does not exists: {input_dir}")
        return
    output_dir = util.get_path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    slides = []
    for filepath in input_dir.iterdir():
        slides.append(_process_path(filepath))
    slides = filter(lambda x: x[0], slides)

    if process_kwargs is None:
        process_kwargs = {}
    for rcpnl, rcjob in slides:
        img_path = rcpnl
        out_path = output_dir / rcpnl.name.replace('.rcpnl', '.ome.zarr')
        start_time = time.perf_counter()

        root = tile.tile_rcpnl(
            img_path, out_path, **process_kwargs,
            metadata={'software': 'quick-look', 'version': _version}
        )
        if root is None:
            continue

        channel_names = [f"Channel {i}" for i in range(len(root['0']))]
        if rcjob is not None:
            print('Adding channel names from rcjob')
            channel_names = tile._rcjob_channel_names(rcjob)            
        channel_names = [
            f"{nn} ({rcpnl.name.split('@')[0]})"
            for nn in channel_names
        ]
        ngff_metadata.add_channel_metadata(root.store.dir_path(), channel_names=channel_names)

        end_time = time.perf_counter()
        img_shape = root['0'].shape
        time_diff = int(end_time - start_time)
        _to_log(output_dir / '000-process.log', rcpnl, img_shape, time_diff)

        print('elapsed', datetime.timedelta(seconds=time_diff))
        print()

        if inspect_focus:
            img_path = pathlib.Path(root.store.dir_path())
            focus_path = img_path.parent / img_path.name.replace('.ome.zarr', '-focus.ome.zarr')
            focus_img = _inspect_focus(root)
            focus_img = focus_img.reshape(1, *focus_img.shape).astype('float32')
            root_focus = tile._make_ngff(
                focus_path,
                focus_img.shape,
                tile_shape=focus_img.shape[1:],
                dtype=focus_img.dtype,
                pixel_size=1.3*32
            )
            for i, (_, aa) in enumerate(root_focus.arrays()):
                aa[:] = focus_img[:, ::8**i, ::8**i]


def watch_directory(
    input_dir,
    output_dir,
    process_kwargs=None
):
    from . import watcher
    from watchdog.events import FileSystemEventHandler
    
    class QuickLookHandler(FileSystemEventHandler):
        def __init__(self):
            self.timer = None

        def on_created(self, event):
            print(event)
            if not self.timer:
                self.timer = time.time()

        def process_events(self):
            if self.timer and (time.time() - self.timer >= 1):
                self.timer = None
                self.run_task()

        def run_task(self):
            process_directory(input_dir, output_dir, process_kwargs)
            time.sleep(5)
            print(f"\nWatcher Running in {input_dir}\n")
            
    input_dir = util.get_path(input_dir)
    w = watcher.Watcher(input_dir, QuickLookHandler())
    w.run()


def watch_directory_production(
    input_dir=path_configs.config['production']['input_dir'],
    output_dir=path_configs.config['production']['output_dir'],
    process_kwargs=None
):
    return watch_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        process_kwargs=process_kwargs
    )


def process_directory_production(
    input_dir=path_configs.config['production']['input_dir'],
    output_dir=path_configs.config['production']['output_dir'],
    process_kwargs=None
):
    return process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        process_kwargs=process_kwargs
    )


def _inspect_focus(root):
    from . import inspect
    img = root[0][0]
    lvar_img = inspect.var_of_laplacian(img, 32, sigma=0)
    mean_img = inspect.mean_block(img, 32)
    return lvar_img / mean_img


def main():
    import fire
    fire.Fire({
        'process': process_directory_production,
        'monitor': watch_directory_production,
        'process-dev': process_directory,
        'monitor-dev': watch_directory
    })


if __name__ == '__main__':
    sys.exit(main())

# ---------------------------------------------------------------------------- #
#                                  Next steps                                  #
# ---------------------------------------------------------------------------- #
'''
[ ] Import into omero or upload to AWS S3 bucket
[ ] Write better min/max
[ ] Detect blurry region
'''
