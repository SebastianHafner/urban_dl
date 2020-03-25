import json
import ee
from gee import assets, sentinel1, sentinel2, export

if __name__ == '__main__':

    ee.Initialize()

    # cities to export
    cities = ['StockholmTest']

    # time series range
    from_date = '2017-06-01'
    to_date = '2017-09-30'

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

        sentinel1_params['orbit_numbers'] = assets.get_orbit_numbers(city)
        # s1_features = sentinel1.get_time_series_features(bbox, from_date, to_date, **sentinel1_params)
        # export.sentinel_to_drive(s1_features, bbox, 'Sentinel1_StockholmTest1', 'gee_test_exports')

        s2_features = sentinel2.get_time_series_features(bbox, from_date, to_date, **sentinel2_params)
        # export.sentinel_to_drive(s2_features, bbox, 'Sentinel2_StockholmTest4', 'gee_test_exports')

