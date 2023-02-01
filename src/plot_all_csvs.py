import glob
import clize
import os
import polars as pl
import plotly.graph_objects as go
import re
import numpy as np

SEED_RE = re.compile("^(.*)_seed=([0-9]+)_(.*)$")


def main(data_dir: str = "data", out_file_base: str = "data/plots"):
    exp_types = {}
    for filepath in glob.glob(f"{data_dir}/**/*.csv", recursive=True):
        match = SEED_RE.match(filepath)
        if match is not None:
            groups = match.groups()
            exp_types.setdefault(f"{groups[0]}_{groups[2]}", {})[
                int(groups[1])
            ] = filepath
    for exp_type, experiments in exp_types.items():
        plot_csvs(exp_type, experiments, out_file_base)


def plot_csvs(exp_type: str, experiments: dict[int, str], out_file_base: str):
    data = {seed: pl.read_csv(filepath) for seed, filepath in experiments.items()}
    columns = next(iter(data.values())).columns
    x_axis = "train/total_timesteps"
    if x_axis not in columns:
        x_axis = "TotalEnvSteps"
    if x_axis not in columns:
        x_axis = None
    for column in columns:
        fig = go.Figure()
        for seed, d in data.items():
            if x_axis is not None:
                x = d[x_axis]
            else:
                x = np.arange(0, len(d[column]))
            fig.add_trace(go.Scatter(x=x, y=d[column], name=str(seed)))
        column_name = column.replace("/", ":")
        exp_name = "/".join(exp_type.split("/")[2:-1])
        os.makedirs(f"{out_file_base}/{exp_name}", exist_ok=True)
        fig.write_html(f"{out_file_base}/{exp_name}_{column_name}.html")


if __name__ == "__main__":
    clize.run(main)
