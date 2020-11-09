from utils.visualization import *
from pathlib import Path
import numpy as np
import json
import torch
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from networks.network_loader import load_network
from utils.dataloader import SpaceNet7Dataset
from experiment_manager.config import config
from utils.metrics import *
from utils.geotiff import *

# TODO: add coordinates to title for area of interests

URBAN_EXTRACTION_PATH = Path('/storage/shafner/urban_extraction')
DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_extraction_dataset_version4')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')
NETWORK_PATH = Path('/storage/shafner/urban_extraction/networks/')\

