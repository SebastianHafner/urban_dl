import ee


def sentinel_to_drive(img: ee.Image, bbox: ee.Geometry, file_name, drive_folder, scale: int = 10,
                      crs: str = 'EPSG:4326', patch_size: int = 256):
    task = ee.batch.Export.image.toDrive(
        image=img,
        region=bbox.getInfo()['coordinates'],
        description='PythonExport',
        folder=drive_folder,
        fileNamePrefix=file_name,
        scale=scale,
        crs=crs,
        maxPixels=1e10,
        # fileDimensions=patch_size,
        fileFormat='GeoTIFF'
    )
    task.start()
    task.status()


def sentinel_to_cloud():
    pass


def label_to_drive():
    pass


def label_to_cloud():
    pass

def download_metadata_to_drive():
    pass

def split_metadata_to_drive():
    pass