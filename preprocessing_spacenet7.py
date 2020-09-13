from pathlib import Path
from utils.geotiff import *
import json
import ee
import utm
import pandas as pd

SPACENET7_PATH = Path('C:/Users/hafne/urban_extraction/data/spacenet7/train')


def extract_bbox(aoi_id: str):

    root_path = SPACENET7_PATH / aoi_id
    img_folder = root_path / 'images'
    all_img_files = list(img_folder.glob('**/*.tif'))
    img_file = all_img_files[0]
    arr, transform, crs = read_tif(img_file)
    y_pixels, x_pixels, _ = arr.shape

    x_pixel_spacing = transform[0]
    x_min = transform[2]
    x_max = x_min + x_pixels * x_pixel_spacing

    y_pixel_spacing = transform[4]
    y_max = transform[5]
    y_min = y_max + y_pixels * y_pixel_spacing

    bbox = ee.Geometry.Rectangle([x_min, y_min, x_max, y_max], proj=str(crs)).transform('EPSG:4326')
    return bbox


def epsg_utm(bbox):
    center_point = bbox.centroid()
    coords = center_point.getInfo()['coordinates']
    lon, lat = coords
    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    return f'EPSG:326{zone_number}' if lat > 0 else f'EPSG:327{zone_number}'


def building_footprint_features(aoi_id, year, month):
    root_path = SPACENET7_PATH / aoi_id
    label_folder = root_path / 'labels_match'
    label_file = label_folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.geojson'

    with open(str(label_file)) as f:
        label_data = json.load(f)

    features = label_data['features']
    new_features = []
    for feature in features:
        coords = feature['geometry']['coordinates']
        geom = ee.Geometry.Polygon(coords, proj='EPSG:3857').transform('EPSG:4326')
        new_feature = ee.Feature(geom)
        new_features.append(new_feature)
    return new_features


def construct_buildings_file(metadata_file: Path):

    metadata = pd.read_csv(metadata_file)

    merged_buildings = None
    for index, row in metadata.iterrows():
        aoi_id, year, month = row['aoi_id'], row['year'], row['month']
        file_name = f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.geojson'
        file = SPACENET7_PATH / aoi_id / 'labels_match' / file_name
        with open(str(file)) as f:
            buildings = json.load(f)

        if merged_buildings is None:
            merged_buildings = buildings
        else:
            merged_features = merged_buildings['features']
            merged_features.extend(buildings['features'])
            merged_buildings['features'] = merged_features

    buildings_file = SPACENET7_PATH.parent / f'sn7_buildings.geojson'
    with open(str(buildings_file), 'w', encoding='utf-8') as f:
        json.dump(merged_buildings, f, ensure_ascii=False, indent=4)


def construct_samples_file(metadata_file: Path):
    metadata = pd.read_csv(metadata_file)
    samples = []
    for index, row in metadata.iterrows():
        # TODO: add country information
        sample = {
            'aoi_id': str(row['aoi_id']),
            'group': int(row['group'])
        }
        samples.append(sample)

    # writing data to json file
    data = {
        'label': 'buildings',
        'sentinel1_features': ['VV', 'VH'],
        'sentinel2_features': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
        'samples': samples
    }
    dataset_file = SPACENET7_PATH.parent / f'samples.json'
    with open(str(dataset_file), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':

    ee.Initialize()

    metadata_file = SPACENET7_PATH.parent / 'sn7_metadata.csv'
    mdf = pd.read_csv(metadata_file)
    # construct_buildings_file(metadata_file)
    # construct_samples_file(metadata_file)

    aoi_names = [f.name for f in SPACENET7_PATH.iterdir() if f.is_dir()]
    patch_features = []
    point_features = []
    for i, aoi_id in enumerate(aoi_names):
        row = mdf.loc[mdf['aoi_id'] == aoi_id]
        year = int(row['year'])
        month = int(row['month'])
        group = int(row['group'])
        bbox = extract_bbox(aoi_id)
        epsg = epsg_utm(bbox)
        properties = {
            'aoi': aoi_id,
            'epsg': epsg,
            'year': year,
            'month': month,
            'group': group
        }
        patch_features.append(ee.Feature(bbox, properties))
        point_features.append(ee.Feature(bbox.centroid(), properties))

    patch_data = ee.FeatureCollection(patch_features).getInfo()
    patch_file = SPACENET7_PATH.parent / f'sn7_patches.geojson'
    with open(str(patch_file), 'w', encoding='utf-8') as f:
        json.dump(patch_data, f, ensure_ascii=False, indent=4)

    points_data = ee.FeatureCollection(point_features).getInfo()
    points_file = SPACENET7_PATH.parent / f'sn7_points.geojson'
    with open(str(points_file), 'w', encoding='utf-8') as f:
        json.dump(points_data, f, ensure_ascii=False, indent=4)