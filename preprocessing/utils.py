import rasterio
from pathlib import Path

# reading in geotiff file as numpy array
def read_tif(file: Path):

    if not file.exists():
        raise FileNotFoundError(f'File {file} not found')

    with rasterio.open(file) as dataset:
        arr = dataset.read()
        transform = dataset.transform
        crs = dataset.crs

    return arr, transform, crs


# writing an array to a geo tiff file
def write_tif(file: Path, arr, transform, crs):

    if not file.parent.exists():
        file.parent.mkdir()

    bands, height, width = arr.shape
    with rasterio.open(
            file,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=bands,
            dtype=arr.dtype,
            crs=crs,
            transform=transform,
    ) as dst:
        for i in range(nbands):
            dst.write(arr[i, ], i + 1)