from dataclasses import dataclass, field, fields
from typing import Any, List, Type, Optional
import os

import stick
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
    parser.add_argument('--config', default=None, type=str)
    # print(parser.equivalent_argparse_code())
    args = parser.parse_args()
    if default is None and args.config is not None:
        # We haven't loaded defaults from a config, and we're being asked to.
        cfg_loaded = simple_parsing.helpers.serialization.load(config_type,
                                                               args.config)
        return _get_config_inner(config_type, default=cfg_loaded)
    else:
        return args.config_arguments


def to_yaml(obj) -> str:
    return simple_parsing.helpers.serialization.dumps_yaml(obj)


def save_yaml(obj, path):
    simple_parsing.helpers.serialization.save_yaml(obj, path)


def load_yaml(obj_type, path):
    return simple_parsing.helpers.serialization.load(obj_type, path)


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


@dataclass
class AlgoConfig(Config):
    seed: int = 0
    exp_name: Optional[str] = None
    log_dir: Optional[str] = None

    '''Used for selecting the best checkpoint and for hparam optimization'''
    minimization_objective: Optional[str] = None


def prepare_training_directory(cfg: AlgoConfig):
    os.makedirs(cfg.log_dir, exist_ok=True)
    save_yaml(cfg, os.path.join(cfg.log_dir, 'config.yaml'))
    # stick will handle seeding for us
    stick.init_extra(log_dir=cfg.log_dir, run_name=cfg.exp_name,
                     config=cfg.to_dict())


class ExperimentInvocation:

    def __init__(self, train_fn, config_type):
        self.parser = simple_parsing.ArgumentParser(
            nested_mode=simple_parsing.NestedMode.WITHOUT_ROOT)
        self.parser.add_argument('--config', default=None, type=str)
        self.parser.add_arguments(config_type, dest='cfg')

        def _train():
            cfg = get_config(config_type)
            prepare_training_directory(cfg)
            train_fn(cfg)

        def _sample_config():
            study = optuna.load_study(storage=self.args.study_storage,
                                      study_name=self.args.study_name)
            trial = study.ask()
            cfg = suggest_config(trial, config_type)
            save_yaml(cfg, self.args.out_path)
            base_path = os.path.splitext(self.args.out_path)[0]
            save_yaml({
                'trial_number': trial.number,
                'study_storage': self.args.study_storage,
                'study_name': self.args.study_name,
                'config': cfg,
            }, f"{base_path}-optuna.yaml")

        def _create_study():
            optuna.create_study(storage=self.args.study_storage,
                                study_name=self.args.study_name)

        def _report_trial():
            trial_data = load_yaml(dict, self.args.trial_file)
            study = optuna.load_study(storage=trial_data['study_storage'],
                                      study_name=trial_data['study_name'])
            result_key = self.args.config.minimization_objective
            # TODO(kr): This function isn't implemented yet (and should it be?)
            results = stick.load_log_file(self.args.log_file,
                                          keys=[result_key])
            study.tell(trial_data["trial_number"], min(results[result_key]))

        self.parser.add_argument('--done-token', type=str, default=None)
        subp = self.parser.add_subparsers()
        train_parser = subp.add_parser("train")
        train_parser.set_defaults(func=_train)
        create_parser = subp.add_parser("create-study")
        create_parser.set_defaults(func=_create_study)
        create_parser.add_argument('--study-storage', type=str)
        create_parser.add_argument('--study-name', type=str)
        report_trial = subp.add_parser("report-trial")
        report_trial.set_defaults(func=_report_trial)
        report_trial.add_argument('--trial-file', type=str)
        report_trial.add_argument('--log-file', type=str)
        sample_parser = subp.add_parser("sample-config")
        sample_parser.set_defaults(func=_sample_config)
        sample_parser.add_argument('--study-storage', type=str)
        sample_parser.add_argument('--study-name', type=str)
        sample_parser.add_argument('--out-path', type=str)
        self.args = self.parser.parse_args()
        if self.args.config:
            loaded_config = load_yaml(config_type, self.args.config)
            self.parser.set_defaults(cfg=loaded_config)
            self.args = self.parser.parse_args()

    def run(self):
        self.args = self.parser.parse_args()
        self.args.func()
        if self.args.done_token:
            with open(self.args.done_token, 'w') as f:
                f.write('done\n')
