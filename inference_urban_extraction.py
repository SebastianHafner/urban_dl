from pathlib import Path

import gdal
import osr
import tifffile
import rasterio
from rasterio.merge import merge

from unet import UNet
from unet import dataloader
from experiment_manager.config import new_config
from preprocessing.utils import *
from unet.augmentations import *
from torchvision import transforms

import matplotlib as mpl


# loading cfg for inference
def load_cfg(configs_dir: Path, experiment: str):
    cfg = new_config()
    cfg.merge_from_file(str(configs_dir / f'{experiment}.yaml'))
    return cfg


# loading network for inference
def load_net(cfg, models_dir: Path, experiment: str):
    net = UNet(cfg)
    state_dict = torch.load(models_dir / f'{experiment}.pkl', map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict)

    mode = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(mode)

    net.to(device)
    net.eval()

    return net


# loading dataset for inference
def load_dataset(cfg, data_dir: Path):
    trfm = transforms.Compose([Npy2Torch()])
    dataset = dataloader.UrbanExtractionDataset(cfg, data_dir, transform=trfm, include_projection=True)
    return dataset


# uses trained model to make a prediction for each tiles
def inference_tiles(data_dir: Path, experiment: str, city: str, configs_dir: Path, models_dir: Path,
                    metadata_exists: bool = True):

    mode = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(mode)

    # loading cfg and network
    cfg = load_cfg(configs_dir, experiment)
    net = load_net(cfg, models_dir, experiment)

    # setting up save directory
    save_dir = data_dir / f'pred_{experiment}'
    if not save_dir.exists():
        save_dir.mkdir()

    # loading dataset from config (requires metadata)
    if not metadata_exists:
        # TODO: generate metadata file
        pass
    # TODO: create dataloader for no labels
    dataset = load_dataset(cfg, data_dir)
    year = dataset.metadata['year']

    # classifying all tiles in dataset
    for i in range(len(dataset)):
        item = dataset.__getitem__(i)
        img = item['x'].to(device)

        metadata = dataset.metadata['samples'][i]
        patch_city = metadata['city']
        patch_id = metadata['patch_id']

        if patch_city == city:
            print(patch_city, patch_id)
            geotransform = item['geotransform']
            epsg = item['epsg']

            y_pred = net(img.unsqueeze(0))
            y_pred = torch.sigmoid(y_pred)

            y_pred = y_pred.cpu().detach().numpy()
            y_pred = y_pred[0, 0, ] > cfg.THRESH
            y_pred = y_pred.astype('uint8')

            fname = f'pred_{patch_city}_{year}_{patch_id}'
            write_tif(y_pred, geotransform, epsg, save_dir, fname, dtype=gdal.GDT_Byte)




def combine_tiles(data_dir: Path, city: str, year: int, tile_size=256, top_left=(0, 0)):

    bottom_right = top_left

    # getting total height
    height = 0
    while True:
        file = data_dir / f'pred_{city}_{year}_{bottom_right[0]:010d}-{top_left[1]:010d}.tif'
        if file.exists():
            bottom_right = (bottom_right[0] + tile_size, bottom_right[1])

            ds = gdal.Open(str(file))
            geotransform = ds.GetGeoTransform()
            print(geotransform)

            # proj = osr.SpatialReference(wkt=ds.GetProjection())
            # epsg = int(proj.GetAttrValue('AUTHORITY', 1))

        else:
            break
    while True:
        file = data_dir / f'pred_{city}_{year}_{top_left[0]:010d}-{bottom_right[1]:010d}.tif'
        if file.exists():
            bottom_right = (bottom_right[0], bottom_right[1] + tile_size)
        else:
            break

    shape = (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
    arr = np.empty(shape, dtype=np.float16)
    # populating arr with classification results
    for i, m in enumerate(range(top_left[0], bottom_right[0], tile_size)):
        for j, n in enumerate(range(top_left[1], bottom_right[1], tile_size)):
            patch_file = data_dir / f'pred_{city}_{year}_{m:010d}-{n:010d}.tif'
            patch_pred = tifffile.imread(str(patch_file))
            arr[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = patch_pred

    classified_map = save_dir / f'pred_{city}_{year}.png'
    mpl.image.imsave(classified_map, arr, vmin=-20, vmax=0, cmap='viridis')


def merge_tiles(root_dir: Path, product: str, save_dir: Path = None):

    # getting all files of product in train and test directory
    train_files_dir = root_dir / 'train' / product
    test_files_dir = root_dir / 'test' / product
    files_in_train = [file for file in train_files_dir.glob('**/*')]
    files_in_test = [file for file in test_files_dir.glob('**/*')]
    files = files_in_train + files_in_test

    # sub setting files to city
    files_to_mosaic = []
    for file in files:
        city_file = file.stem.split('_')[1]
        if city_file == city:
            src = rasterio.open(str(file))
            files_to_mosaic.append(src)

    # merging all files to a mosaic
    mosaic, out_trans = merge(files_to_mosaic)

    # getting metadata
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans})

    # dropping patch id from file name
    fname_parts = file.stem.split('_')[:-1]
    prefix = '_'.join(fname_parts)

    if save_dir is None:
        save_dir = root_dir / 'maps'
    if not save_dir.exists():
        save_dir.mkdir()

    out_file = save_dir / f'{prefix}_{experiment}.tif'
    with rasterio.open(out_file, "w", **out_meta) as dest:
        dest.write(mosaic)



if __name__ == '__main__':

    CONFIGS_DIR = Path('configs/urban_extraction')

    # models_dir = Path('C:/Users/shafner/models')
    models_dir = Path('/home/yonk/saved_models')

    # preprocessed_dir = Path('C:/Users/shafner/projects/urban_extraction/data/preprocessed')
    preprocessed_dir = Path('/storage/yonk/')

    # set dataset and experiment
    dataset = 'twocities'
    experiment = 's1s2_allbands'

    for train_test in ['train', 'test']:
        for city in ['Beijing', 'Stockholm']:
            data_dir = preprocessed_dir / f'urban_extraction_{dataset}' / train_test
            inference_tiles(
                data_dir=data_dir,
                experiment=f'{experiment}_{dataset}',
                city=city,
                configs_dir=CONFIGS_DIR,
                models_dir=models_dir,
                metadata_exists=True
            )

    product = 'pred_s1s2_allbands_twocities'
    root_dir = preprocessed_dir / f'urban_extraction_{dataset}'
    # merge_tiles(root_dir, product)
