import os
import kaggle
from zipfile import ZipFile

kaggle.api.authenticate()

if not os.path.exists('./data/raw'):
    os.makedirs('./data/raw')

kaggle.api.competition_download_files('ieee-fraud-detection', path='./data/raw')

with ZipFile('./data/raw/ieee-fraud-detection.zip', 'r') as zip_ref:
    zip_ref.extractall('./data/raw/ieee-fraud-detection')