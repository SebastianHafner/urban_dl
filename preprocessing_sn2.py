from pathlib import Path
from utils.geotiff import *

SN2_PATH = Path('C:/Users/shafner/urban_extraction/data/spacenet2')
SN2_SITES = ['AOI_2_Vegas', 'AOI_3_Paris', 'AOI_4_Shanghai', 'AOI_5_Khartoum']


def combine_geojsons(aoi: str):

    path = SN2_PATH / aoi
    files = [f for f in path.glob('**/*')]

    all_data = None
    for file in files:
        data = load_json(file)
        if all_data is None:
            all_data = data
        else:
            all_data['features'].extend(data['features'])

    output_file = SN2_PATH / f'{aoi}_Buildings.geojson'
    write_json(output_file, all_data)


if __name__ == '__main__':

    for aoi in SN2_SITES:
        combine_geojsons(aoi)
