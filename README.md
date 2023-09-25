<h1>
    <img alt="quick-look" height="60" src="quick-look-logo.svg">
</h1>

Quickly tile whole-slide scans into a pyramidal images for holistic image review
and data-driven decision-making in highly-multiplexed tissue imaging.

*NOTE: Whole-slide scans from RareCyte imagers are supported. Processing file
formats from other vendors are feasible but is not supported at the moment.*

## Installation

Installing palom in a fresh conda environment is recommended. [Instruction for
installing miniconda](https://docs.conda.io/en/latest/miniconda.html).

### Install `quick-look` on the computer that does the tiling

*In most cases, this is also the computer that acquires the images.*

```bash
conda create -n quick-look -c conda-forge python=3.10 numpy scipy matplotlib networkx scikit-image=0.19 scikit-learn tifffile zarr pyjnius blessed tqdm fire watchdog joblib pywin32

conda activate quick-look

python -m pip install quick-look
```

### Install `napari` and `napari-ome-zarr` to view the pyramidal images

```bash
conda create -n napari -c conda-forge python=3.10 pyqt napari

conda activate napari

python -m pip install napari-ome-zarr
```

## Using `quick-look`

### Usecase 1: process scans "by hand"

```bash
quicklook process -i <path/to/input/directory> -o <path/to/output/directory>
```

`quicklook process` processes **folders** or **files** in the input directory
(`<path/to/input/directory>`). The **folders** must contain one `.rcpnl` file
and optionally one `.rcjob` file; thie **files** must be a `.rcpnl` files. On
Windows computers, the **folders** and **files** can be "shortcuts". The user
doesn't need to move the actual files to the input folder.

Example input, containing one `.rcpnl` file and one folder:

```cmd
C:\quick-look\input
│   LSP001@20230922_142033_633584.rcpnl
│
└───LSP002@20230922_200722_567967
        LSP002@20230922_200722_567967.rcjob
        LSP002@20230922_200722_567967.rcpnl
```

Command executed

```bash
quicklook process -i C:\quick-look\input -o C:\quick-look\output
```

Example output, two `.ome.zarr` are generated:

```cmd
C:\quick-look\output
├───LSP001@20230922_142033_633584.ome.zarr
└───LSP002@20230922_200722_567967.ome.zarr
```

### Usecase 2: monitor a directory and tile a scan right after its file is created

```bash
quicklook monitor -i <path/to/onput/directory> -o <path/to/output/directory>
```

Use `quicklook monitor` command to monitor a folder (`<path/to/onput/directory>`)
on a computer. Everytime an user paste shortcuts (of folders or files as
described in usecase 1) into the folder, tiling will be launched, the results
will be written to `<path/to/output/directory>`.

## Review the tiled whole-slide scans

`quick-look` writes out [NGFF
v0.4](https://ngff.openmicroscopy.org/0.4/index.html) files. Using Napari with
napari-ome-zarr is currently recommended for opening `quick-look` outputs. Other
tools that might be able to open the images are listed
[here](https://ome.github.io/ome-ngff-tools/).
