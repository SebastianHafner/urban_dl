import json
import ee
from pathlib import Path
from gee import assets, sentinel1, sentinel2, export


def write_download_metadata(save_dir: Path, cities, year, from_date, to_date, sentinel1_params, sentinel2_params):

    download_metadata = {
        'cities': cities,
        'year': year,
        'from_date': from_date,
        'to_date': to_date,
        'sentinel1': {
            'polarizations': sentinel1_params['polarizations'],
            'orbits': ['asc', 'desc'],
            'metrics': sentinel1_params['metrics']
        },
        'sentinel2': {
            'bands': sentinel2_params['bands'],
            'indices': sentinel2_params['indices'],
            'metrics': sentinel2_params['metrics']
        }
    }

    print(download_metadata)
    with open(str(save_dir / 'download_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(download_metadata, f, ensure_ascii=False, indent=4)
    pass

if __name__ == '__main__':

    save_dir = Path('C:/Users/hafne/Desktop/projects/data/gee/')

    ee.Initialize()

    # cities to export
    cities = ['StockholmTest']

    # time series range
    year = 2017
    from_date = f'{year}-06-01'
    to_date = f'{year}-09-30'

    # sentinel 1 params
    sentinel1_params = {
        'polarizations': ['VV', 'VH'],
        'metrics': ['mean', 'stdDev', 'min', 'max'],
        'include_count': True
    }

    # sentinel 2 params
    sentinel2_params = {
        'bands': ['Blue', 'Green', 'Red', 'NIR'],
        'indices': [],
        'metrics': ['median'],
        'include_count': True
    }


    for city in cities:

        bbox = assets.get_bbox(city)

        # sentinel1_params['orbit_numbers'] = assets.get_orbit_numbers(city)
        # s1_features = sentinel1.get_time_series_features(bbox, from_date, to_date, **sentinel1_params)
        # export.sentinel_to_drive(s1_features, bbox, 'Sentinel1_StockholmTest1', 'gee_test_exports')

        # s2_features = sentinel2.get_time_series_features(bbox, from_date, to_date, **sentinel2_params)
        # export.sentinel_to_drive(s2_features, bbox, 'Sentinel2_StockholmTest4', 'gee_test_exports')


    write_download_metadata(save_dir, cities, year, from_date, to_date, sentinel1_params, sentinel2_params)