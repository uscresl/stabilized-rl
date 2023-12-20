from dataclasses import dataclass, field, fields
from typing import Any, List, Type

import optuna
import simple_parsing

class Config(simple_parsing.Serializable):

    def state_dict(self):
        return self.to_dict()

    @classmethod
    def sample(cls, trial: optuna.Trial, **kwargs):
        return suggest_config(trial, cls, **kwargs)


def get_config(config_type):
    return _get_config_inner(config_type)


def _get_config_inner(config_type, default=None):
    parser = simple_parsing.ArgumentParser(
        nested_mode=simple_parsing.NestedMode.WITHOUT_ROOT)
    parser.add_arguments(config_type, dest='config_arguments', default=default)
    parser.add_argument('--cfg-path', default=None, type=str)
    # print(parser.equivalent_argparse_code())
    args = parser.parse_args()
    if default is None and args.cfg_path is not None:
        # We haven't loaded defaults from a config, and we're being asked to.
        cfg_loaded = simple_parsing.helpers.serialization.load(config_type, args.cfg_path)
        return _get_config_inner(config_type, default=cfg_loaded)
    else:
        return args.config_arguments


def to_yaml(obj) -> str:
    return simple_parsing.helpers.serialization.dumps_yaml(obj)


def save_yaml(obj, path):
    simple_parsing.helpers.serialization.save_yaml(obj, path)


class CustomDistribution:
    def sample(self, name: str, trial: optuna.Trial) -> Any:
        del name, trial
        raise NotImplementedError()


@dataclass
class IntListDistribution(CustomDistribution):

    low: List[int]
    high: List[int]

    def sample(self, name, trial) -> List[int]:
        list_len = trial.suggest_int(f"{name}_len", low=len(self.low),
                                     high=len(self.high))
        values = []
        for i in range(list_len):
            low_i = min(i, len(self.low) - 1)
            values.append(trial.suggest_int(f"{name}_{i}", low=self.low[low_i],
                                            high=self.high[i]))
        return values



OPTUNA_DISTRIBUTION = 'OPTUNA_DISTRIBUTION'


def tunable(*args, distribution, metadata=None, **kwargs):
    if metadata is None:
        metadata = {}
    metadata['OPTUNA_DISTRIBUTION'] = distribution
    return field(*args, **kwargs, metadata=metadata)


def suggest_config(trial: optuna.Trial, config: Type, **kwargs):
    sampled = {}
    for f in fields(config):
        if f.name in kwargs:
            continue
        if OPTUNA_DISTRIBUTION in f.metadata:
            dist = f.metadata[OPTUNA_DISTRIBUTION]
            if isinstance(dist, CustomDistribution):
                sampled[f.name] = dist.sample(f.name, trial)
            else:
                sampled[f.name] = trial._suggest(f.name, dist)
    return config(**kwargs, **sampled)
