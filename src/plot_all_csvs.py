import glob
import clize
import os
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import re
import numpy as np
from typing import Union
import itertools

COLOR_PALETTE = px.colors.sequential.Viridis
SEED_RE = re.compile("^(.*)_seed=([0-9]+)_(.*)$")
KV_RE = re.compile("[/_]([^=/]*)=([^/_]+)")
TOP_KEYS = ["env", "algo", "note"]
TOP_KEY_PRODUCTS = [f"{k1}_{k2}" for k1 in TOP_KEYS for k2 in TOP_KEYS if k1 != k2]

data_dir = "data"
out_file_base = "data/plots"


def kv_from_csv_path(csv_path: str) -> list[tuple[str, str]]:
    as_list = KV_RE.findall(csv_path) + [("algo", csv_path.split("/")[2])]
    as_dict = dict(as_list)
    for i, k1 in enumerate(TOP_KEYS):
        for j, k2 in enumerate(TOP_KEYS):
            if j <= i:
                continue
            as_list += [(f"{k1}_{k2}", f"{as_dict.get(k1, '')}_{as_dict.get(k2, '')}")]
    return as_list


def main(data_dir: str = "data", out_file_base: str = "data/plots"):
    csv_paths = glob.glob(f"{data_dir}/**/*.csv", recursive=True)
    csv_paths = [csv_path for csv_path in csv_paths if "/tmp/" not in csv_path]
    data = {}
    for csv_path in csv_paths:
        try:
            data[csv_path] = pl.read_csv(csv_path, ignore_errors=True)
        except pl.NoDataError:
            pass

    kv_pair_groups = {}
    for csv_path in csv_paths:
        for kv_pair in kv_from_csv_path(csv_path):
            kv_pair_groups.setdefault(kv_pair, []).append(csv_path)

    for kv_pair, kv_csv_path in kv_pair_groups.items():
        if kv_pair[0] in TOP_KEYS or kv_pair[0] in TOP_KEY_PRODUCTS:
            plot_kv_pair_group(kv_pair, kv_csv_path, data, out_file_base)
            plot_kv_pair_group(
                kv_pair,
                kv_csv_path,
                data,
                out_file_base,
                y_axes=["rollout/ep_rew_mean"],
            )

    exp_types = {}
    for csv_path in csv_paths:
        match = SEED_RE.match(csv_path)
        if match is not None:
            groups = match.groups()
            seed = int(groups[1])
            exp_types.setdefault(f"{groups[0]}_{groups[2]}", []).append(
                (seed, csv_path)
            )
    for exp_type, experiments in exp_types.items():
        plot_seed_groups(exp_type, experiments, data, out_file_base)


def get_axis(candidates: list[str], columns: list[str]) -> Union[str, None]:
    i = 0
    while i < len(candidates) and candidates[i] not in columns:
        i += 1
    if i == len(candidates):
        return None
    else:
        return candidates[i]


X_AXES = ["rollout/AverageReturn", "Evaluation/AverageReturn"]
Y_AXES = ["time/total_timesteps", "TotalEnvSteps"]


def interp_experiments(
    experiments: list[tuple[str, pl.DataFrame]],
    x_axes: list[str] = X_AXES,
    y_axes: list[str] = Y_AXES,
    x_resolution: int = 5000,
) -> Union[None, tuple[np.ndarray, list[np.ndarray]]]:
    xys = []
    for csv_path, exp in experiments:
        x_axis = get_axis(x_axes, exp.columns)
        y_axis = get_axis(y_axes, exp.columns)
        if x_axis is None or y_axis is None:
            # print(f"Skipping {csv_path}")
            continue
        xys.append(pl.DataFrame({"x": exp[x_axis], "y": exp[y_axis]}))
    if not xys:
        return None
    last_x = max(xy["x"].max() for xy in xys)
    interp_x = np.linspace(0, last_x, x_resolution)
    interp_ys = []
    for xy in xys:
        xy_no_nulls = xy[1:].drop_nulls()
        x = xy_no_nulls["x"]
        y = xy_no_nulls["y"]
        interp_y = np.interp(interp_x, x, y)
        interp_ys.append(interp_y)
    return interp_x, interp_ys


def plot_kv_pair_group(
    kv_pair: tuple[str, str],  # e.g. ("env", "HalfCheetah-v2")
    kv_csv_path: list[str],
    data: dict[str, pl.DataFrame],
    out_file_base: str,
    y_axes: list[str] = X_AXES,
    x_axes: list[str] = Y_AXES,
):
    reduce_by_keys: set[str] = set()
    for csv_path in kv_csv_path:
        kvs = kv_from_csv_path(csv_path)
        for k, _ in kvs:
            reduce_by_keys.add(k)
    for reduce_key in reduce_by_keys:
        by_reduce_key: dict[Union[str, None], list[tuple[str, pl.DataFrame]]] = {}
        # reduce_key e.g. "kl-target"
        for csv_path in kv_csv_path:
            kv = dict(kv_from_csv_path(csv_path))
            by_reduce_key.setdefault(kv.get(reduce_key, None), []).append(
                (csv_path, data[csv_path])
            )
        fig = go.Figure()
        color_palette = itertools.cycle(COLOR_PALETTE)
        dash_palette = itertools.cycle(
            ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]
        )
        for reduce_value, experiments in by_reduce_key.items():
            # reduce_value e.g. "0.3"
            color = next(color_palette)
            color = next(color_palette)
            dash = next(dash_palette)
            interp = interp_experiments(experiments, x_axes=x_axes, y_axes=y_axes)
            if interp is None:
                continue
            interp_x, interp_ys = interp
            max_line = np.stack(interp_ys).max(axis=0)
            min_line = np.stack(interp_ys).min(axis=0)
            mean_line = np.stack(interp_ys).mean(axis=0)
            boundary = go.Scatter(
                x=np.concatenate([interp_x, interp_x[::-1]]),
                y=np.concatenate([max_line, min_line[::-1]]),
                fill="toself",
                name=f"{reduce_key}: {reduce_value}",
                hovertext=reduce_value,
                line=dict(color=color, dash=dash),
            )
            fig.add_trace(boundary)
            fig.add_trace(
                go.Scatter(
                    x=interp_x,
                    y=mean_line,
                    name=f"{reduce_key}: {reduce_value} mean",
                    hovertext=reduce_value,
                    line=dict(color=color, dash=dash),
                )
            )
        column_name = y_axes[0].replace("/", ":")
        key, value = kv_pair
        key = key.replace("/", ":")
        os.makedirs(f"{out_file_base}/{key}={value}", exist_ok=True)
        fig.write_html(
            f"{out_file_base}/{key}={value}/{column_name}_by_{reduce_key}.html"
        )


def plot_seed_groups(
    exp_type: str,
    experiments: list[tuple[int, str]],
    data: dict[str, pl.DataFrame],
    out_file_base: str,
):
    seeds = [seed for seed, _ in experiments]
    csv_paths = [csv_path for _, csv_path in experiments]
    data_for_exp = [(csv_path, data[csv_path]) for csv_path in csv_paths]
    columns = set()
    for table in data.values():
        columns = columns.union(set(table.columns))
    for column in columns:
        try:
            interp = interp_experiments(data_for_exp, y_axes=[column])
        except NotImplementedError:
            # This usually indicates invalid data in the CSV, that was then parsed as a string
            continue
        if interp is None:
            continue
        interp_x, interp_ys = interp
        fig = go.Figure()
        for interp_y, seed in zip(interp_ys, seeds):
            fig.add_trace(go.Scatter(x=interp_x, y=interp_y, name=str(seed)))
        column_name = column.replace("/", ":")
        exp_name = "/".join(exp_type.split("/")[2:-1])
        os.makedirs(f"{out_file_base}/{exp_name}", exist_ok=True)
        fig.write_html(f"{out_file_base}/{exp_name}_{column_name}.html")


if __name__ == "__main__":
    clize.run(main)
