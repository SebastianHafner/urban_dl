from pathlib import Path
from utils.geotiff import *
import json
import ee

SPACENET7_PATH = Path('C:/Users/shafner/urban_extraction/data/spacenet7/train')


def extract_bbox(aoi_name: str):

    root_path = SPACENET7_PATH / aoi_name
    img_folder = root_path / 'images'
    all_img_files = list(img_folder.glob('**/*.tif'))
    img_file = all_img_files[0]
    arr, transform, crs = read_tif(img_file)
    y_pixels, x_pixels, _ = arr.shape
    print(y_pixels, x_pixels)
    print(crs)
    print(transform)

    x_pixel_spacing = transform[0]
    x_min = transform[2]
    x_max = x_min + x_pixels * x_pixel_spacing

    y_pixel_spacing = transform[4]
    y_max = transform[5]
    y_min = y_max + y_pixels * y_pixel_spacing

    print(f'x min: {x_min:.3f} - x max: {x_max:.3f}')
    print(f'y min: {y_min:.3f} - y max: {y_max:.3f}')

    ul = ee.Geometry.Point([50, 50])
    print(ul.getInfo())
    bbox = ee.Geometry.Rectangle([x_min, y_min, x_max, y_max], proj=str(crs))
    print(bbox.getInfo())
    data = ee.FeatureCollection([bbox]).getInfo()

    bbox_file = SPACENET7_PATH.parent / 'bboxes' / f'bbox_{aoi_name}.geojson'
    with open(str(bbox_file), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':

    ee.Initialize()
    aoi_name = 'L15-0331E-1257N_1327_3160_13'

    extract_bbox(aoi_name)
