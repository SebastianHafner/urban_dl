import json
from pathlib import Path


def combine_geojson_files(folder: Path, output_file: Path):

    all_files = list(folder.glob('*.geojson'))

    first_file = all_files[0]
    with open(str(first_file)) as f:
        output_data = json.load(f)
    output_data['features'] = []

    for file in all_files:
        with open(str(file)) as f:
            data = json.load(f)
            features = data['features']
        output_data['features'].extend(features)

    with open(str(output_file), 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':

    SPACENET_PATH = Path('C:/Users/shafner/urban_extraction/data/spacenet_buildings')
    TEST_FOLDER = SPACENET_PATH / 'test'
    test_file = SPACENET_PATH / 'test.geojson'


    combine_geojson_files(TEST_FOLDER, test_file)
