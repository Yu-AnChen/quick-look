# ---------------------------------------------------------------------------- #
#                         Process folder with shortcuts                        #
# ---------------------------------------------------------------------------- #
import datetime
import pathlib
import sys
import time

from . import ngff_metadata, preview_slide, util

INPUT_DIR = r"/Users/yuanchen/Dropbox (HMS)/ashlar-dev-data/ashlar-rotation-data/3"
OUTPUT_DIR = r"/Users/yuanchen/projects/napari-wsi-reader/src/.dev"


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


def process_dir(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    process_kwargs=None
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

        root = preview_slide.rcpnl_to_mosaic_ngff(
            img_path, out_path, **process_kwargs
        )
        if root is None:
            continue

        channel_names = [f"Channel {i}" for i in range(len(root['0']))]
        if rcjob is not None:
            print('Adding channel names from rcjob')
            channel_names = preview_slide._rcjob_channel_names(rcjob)            
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


def watch_directory(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
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
            process_dir(input_dir, output_dir, process_kwargs)
            time.sleep(5)
            print(f"\nWatcher Running in {input_dir}/\n")
            
    input_dir = util.get_path(input_dir)
    w = watcher.Watcher(input_dir, QuickLookHandler())
    w.run()


def watch_directory_production(
    input_dir=r"C:\Users\rarecyte\Desktop\INPUT-ashlar-lt",
    output_dir=r"C:\Users\rarecyte\Desktop\OUTPUT-ashlar-lt.lnk",
    process_kwargs=None
):
    return watch_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        process_kwargs=process_kwargs
    )


def main():
    import fire
    fire.Fire({
        'run': process_dir,
        'watch': watch_directory_production,
        'watch-dev': watch_directory
    })


if __name__ == '__main__':
    sys.exit(main())

# ---------------------------------------------------------------------------- #
#                                  Next steps                                  #
# ---------------------------------------------------------------------------- #
'''
[ ] Import into omero or upload to AWS S3 bucket
'''
