from pathlib import Path
from utils.geotiff import *
import json
import ee
import utm
import pandas as pd

SPACENET7_PATH = Path('C:/Users/shafner/urban_extraction/data/spacenet7/train')


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
        sample = {
            'aoi_id': str(row['aoi_id']),
            'group': int(row['group']),
            'country': str(row['country']),
            'month': int(row['month']),
            'year': int(row['year']),
        }
        samples.append(sample)

    # writing data to json file
    data = {
        'label': 'buildings',
        'sentinel1_features': ['VV_mean', 'VV_stdDev', 'VH_mean', 'VH_stdDev'],
        'sentinel2_features': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
        'group_names': {'1': 'NA_AU', '2': 'SA', '3': 'EU', '4': 'SSA', '5': 'NAF_ME', '6': 'AS'},
        'samples': samples
    }
    dataset_file = SPACENET7_PATH.parent / f'samples.json'
    with open(str(dataset_file), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def dt_months(year: int, month: int, dt_months: int):
    if month > dt_months:
        return year, month - dt_months
    else:
        return year - 1, 12 - abs(month - dt_months)


def find_offset_file(aoi_id: str, year: int, month: int):

    ref_year, ref_month = dt_months(year, month, 6)

    while True:
        ref_file_name = f'global_monthly_{ref_year}_{ref_month:02d}_mosaic_{aoi_id}_Buildings.geojson'
        ref_file = SPACENET7_PATH / aoi_id / 'labels_match' / ref_file_name
        if ref_file.exists():
            return ref_file
        else:
            ref_year, ref_month = dt_months(ref_year, ref_month, 1)
        if year < 2018:
            raise Exception('No file found!')


def construct_stable_buildings_file(metadata_file: Path):

    metadata = pd.read_csv(metadata_file)

    merged_buildings = None
    for index, row in metadata.iterrows():
        aoi_id, year, month = row['aoi_id'], row['year'], row['month']
        file_name = f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.geojson'
        file = SPACENET7_PATH / aoi_id / 'labels_match' / file_name

        ref_file = find_offset_file(aoi_id, year, month)

        buildings = load_json(file)
        ref_buildings = load_json(ref_file)

        ref_ids = [feature['properties']['Id'] for feature in ref_buildings['features']]

        stable_features = []
        for feature in buildings['features']:
            feature_id = feature['properties']['Id']
            stable = 1 if feature_id in ref_ids else 0
            feature['properties']['stable'] = stable
            stable_features.append(feature)

        if merged_buildings is None:
            buildings['features'] = []
            merged_buildings = buildings

        merged_features = merged_buildings['features']
        merged_features.extend(stable_features)
        merged_buildings['features'] = merged_features

    # buildings_file = SPACENET7_PATH.parent / f'sn7_buildings.geojson'
    # with open(str(buildings_file), 'w', encoding='utf-8') as f:
    #     json.dump(merged_buildings, f, ensure_ascii=False, indent=4)


def construct_reference_buildings_file(metadata_file: Path):
    metadata = pd.read_csv(metadata_file)

    merged_buildings = None
    for index, row in metadata.iterrows():
        aoi_id, year, month = row['aoi_id'], row['year'], row['month']

        ref_file = find_offset_file(aoi_id, year, month)
        buildings = load_json(ref_file)

        if merged_buildings is None:
            merged_buildings = buildings
        else:
            merged_features = merged_buildings['features']
            merged_features.extend(buildings['features'])
            merged_buildings['features'] = merged_features

    buildings_file = SPACENET7_PATH.parent / f'sn7_reference_buildings.geojson'
    with open(str(buildings_file), 'w', encoding='utf-8') as f:
        json.dump(merged_buildings, f, ensure_ascii=False, indent=4)


def merge_time_series(aoi_id):

    all_mighty_container = None
    all_mighty_ids = None

    aoi_path = SPACENET7_PATH / aoi_id
    buildings_path = aoi_path / 'labels_match_pix'
    udm_path = aoi_path / 'UDM_masks'

    def file2number(file: Path) -> int:
        fname = file.name
        parts = fname.split('_')
        year = int(parts[2])
        month = int(parts[3])
        return year * 12 + month

    files = [(f, file2number(f)) for f in buildings_path.glob('**/*')]
    files = sorted(files, key=lambda f: f[1])
    building_files = [item[0] for item in files]

    for building_file in building_files:
        name = building_file.name
        name_parts = name.split('_')

        year = int(name_parts[2])
        month = int(name_parts[3])

        fc = load_json(building_file)

        # TODO: check for UDM mask

        if all_mighty_container is None:
            all_mighty_container = fc
            all_mighty_ids = [b['properties']['Id'] for b in fc['features']]
        else:
            buildings = fc['features']
            for building in buildings:
                building_id = building['properties']['Id']
                if building_id in all_mighty_ids:
                    continue
                else:
                    all_mighty_ids.append(building_id)
                    



if __name__ == '__main__':

    ee.Initialize()

    metadata_file = SPACENET7_PATH.parent / 'sn7_metadata_v3.csv'
    construct_reference_buildings_file(metadata_file)

    # construct_buildings_file(metadata_file)
    # construct_samples_file(metadata_file)