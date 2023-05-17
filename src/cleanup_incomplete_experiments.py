import clize
import glob
from tqdm import tqdm
import re
import json
import pandas as pd
import os.path
import shutil

def main(*, dry_run: bool = False):
    variant_jsons = glob.glob('data/experiments/**/**/variant.json')
    for var_json in tqdm(variant_jsons):
        with open(var_json) as f:
            variant = json.load(f)
        try:
            total_steps_expected = variant["total_steps"]
        except KeyError:
            print(f"total_steps not found in {var_json}")
        print(total_steps_expected)
        csv_path = var_json.replace('variant.json', 'progress.csv')
        should_move = True
        try:
            progress = pd.read_csv(csv_path)
            if progress["time/total_timesteps"].max() >= total_steps_expected:
                should_move = False
        except (TypeError, pd.errors.ParserError):
            print(f"Error parsing {csv_path}")
        exp_dir = os.path.split(var_json)[0]
        if not dry_run:
            print(f"Moving {exp_dir}")
            shutil.move(exp_dir, 'data_incomplete/' + exp_dir)
        else:
            print(f"Would move {exp_dir}")


if __name__ == '__main__':
    clize.run(main)
