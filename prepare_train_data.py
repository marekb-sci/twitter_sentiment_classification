import pandas as pd
from pathlib import Path
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script splits input data into training and validation files.')

    parser.add_argument('--input-path', type=str, help='path to .csv data file with tweet data')
    parser.add_argument('--output-dir', type=str, help='path to directory where splitted data will be saved')
    parser.add_argument('--val-size', type=float, default=0.2, help='fraction of data used for validation data')

    args = parser.parse_args()

    raw_data = pd.read_csv(args.input_path, names= ['label', 'id', 'timestamp', 'query', 'user', 'text'], encoding='latin-1')
    raw_data['label'] = LabelEncoder().fit_transform(raw_data['label'])
    train_indices, val_indices = train_test_split(raw_data.index, test_size=args.val_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    filename_stem = Path(args.input_path).stem

    raw_data.loc[train_indices].to_csv(output_dir / f'{filename_stem}_train.csv', index=False)
    raw_data.loc[val_indices].to_csv(output_dir / f'{filename_stem}_val.csv', index=False)
