<h1>
    <img alt="quick-look" height="60" src="quick-look-logo.svg">
</h1>

Quickly tile whole-slide scans into pyramidal images for holistic image review
and data-driven decision-making in highly-multiplexed tissue imaging.

## What it does

`quick-look` is a command-line tool that processes whole-slide scans from
RareCyte imagers (`.rcpnl` files) and converts them into pyramidal
[OME-Zarr](https://ngff.openmicroscopy.org/0.4/) images. This format is ideal
for fast, interactive viewing of very large images, as it allows visualization
clients to load only the visible parts of the image at the appropriate
resolution.

*NOTE: Whole-slide scans from RareCyte imagers are supported. Processing file
formats from other vendors is feasible but is not supported at the moment.*

## Key features

- Converts `.rcpnl` files to multi-scale OME-Zarr.
- Processes a directory of scans or monitors a directory for new scans to
  process automatically.
- Extracts channel names from `.rcjob` files if present.

## Installation

We recommend using [pixi](https://pixi.sh/latest/installation/) for environment
management.

1. **Create a pixi workspace & install `quick-look`:**

    ```cmd
    mkdir quicklook-env
    cd quicklook-env
    
    # Download pixi configuration
    curl -OL https://raw.githubusercontent.com/Yu-AnChen/quick-look/main/pixi/pixi.toml
    curl -OL https://raw.githubusercontent.com/Yu-AnChen/quick-look/main/pixi/pixi.lock

    # Install dependencies
    pixi install
    ```

2. **Activate the environment:**

    When using pixi, prepend `pixi run` to all `quicklook` commands. For
    example: `pixi run quicklook process --help`.

## Usage

`quick-look` provides two main commands: `process` to convert a directory of
scans, and `monitor` to watch a directory for new scans.

### Process a directory of scans

The `process` command will process all compatible files/folders in the specified
input directory.

```bash
quicklook process -i <path/to/input/directory> -o <path/to/output/directory>
```

The input directory can contain:

- `.rcpnl` files.
- Folders, each containing one `.rcpnl` file and optionally one `.rcjob` file
  (for channel names).
- On Windows, shortcuts to the above files and folders are also supported.

**Example:**

Given this input directory, containing 1 rcpnl file and a folder with an rcpnl
file in it:

```cmd
C:\quick-look\input
│   LSP001@20230922_142033_633584.rcpnl
│
└───LSP002@20230922_200722_567967
        LSP002@20230922_200722_567967.rcjob
        LSP002@20230922_200722_567967.rcpnl
```

Running this command:

```bash
quicklook process -i C:\quick-look\input -o C:\quick-look\output
```

Will produce two OME-Zarr folders in the output directory:

```cmd
C:\quick-look\output
├───LSP001@20230922_142033_633584.ome.zarr
└───LSP002@20230922_200722_567967.ome.zarr
```

### Monitor a directory for new scans

The `monitor` command watches a directory and automatically processes new scans
as they are added. This is useful for automated processing pipelines.

```bash
quicklook monitor -i <path/to/input/directory> -o <path/to/output/directory>
```

The tool will watch the input directory and, when a new file, folder, or Windows
shortcut is added, it will launch the tiling process and write the result to the
output directory.

## Viewing the output

The generated `.ome.zarr` files can be viewed with various tools that support
the NGFF format. We recommend using [QuPath v0.6.x](https://qupath.github.io/).

Other tools that can open NGFF images are listed at [NGFF
documentation](https://ngff.openmicroscopy.org/resources/tools/index.html#zarr-viewers).
