import copy
import warnings
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import gym
import numpy as np
import torch as th
from gym import spaces
from v_trace_buffer import VTraceRolloutBuffer
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    Schedule,
    RolloutBufferSamples,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
    obs_as_tensor,
)
from stable_baselines3.common.vec_env import VecEnv
from torch.distributions import kl_divergence
from torch.distributions.independent import Independent
from torch.nn import functional as F

MIN_KL_LOSS_COEFF = 1e-2
SECOND_PENALTY_LOOP_MAX = 100
MAX_KL_LOSS_COEFF = 100
# MAX_LOG_KL_LOSS_COEFF = 10

# MAX_KL_LOSS_COEFF = 2**20
# MAX_LOG_KL_LOSS_COEFF = 20


class KLPOStbl(OnPolicyAlgorithm):
    """
    KLPO Algo

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 8192,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = False,
        normalize_batch_advantage: bool = False,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        clip_grad_norm=True,
        _init_setup_model: bool = True,
        max_path_length: int = None,
        *,
        kl_loss_coeff_lr: float,
        kl_loss_coeff_momentum: float,
        maximum_kl_loss_coeff: float = 2**20,
        min_kl_loss_coeff: float = 0.01,
        bang_bang_kl_loss_opt: bool = False,
        bang_bang_reset_kl_loss_coeff: bool = False,
        kl_target_stat: str,
        optimize_log_loss_coeff: bool = False,
        reset_optimizers: bool = False,
        historic_buffer_size: int = 8192,
        second_penalty_loop: bool = True,
        minibatch_kl_penalty: bool = True,
        use_beta_adam: bool = True,
        sparse_second_loop: bool = True,
        second_loop_batch_size: int = 1024,
        second_loop_vf: bool = False,
        multi_step_trust_region: bool = False,
        eval_policy: bool = False,
        debug_plots: bool = False,
        debug_pkls: bool = False,
        early_stop_epoch: Optional[bool] = False,
        early_stop_across_epochs: Optional[bool] = False,
        v_trace: bool = False,
        reset_beta: bool = False,
    ):

        assert not multi_step_trust_region, "Are you sure you want to do this?"
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"
        assert self.n_steps == n_steps
        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self._historic_buffer_size = historic_buffer_size
        self._v_trace = v_trace

        if _init_setup_model:
            self._setup_model()

        self._old_policy = copy.deepcopy(self.policy)
        self._normalize_batch_advantage = normalize_batch_advantage
        self._clip_grad_norm = clip_grad_norm

        self.target_kl = target_kl
        if max_path_length is None:
            self.max_path_length = n_steps
        else:
            self.max_path_length = max_path_length

        self._optimize_log_loss_coeff = optimize_log_loss_coeff
        self._kl_loss_coeff_lr = kl_loss_coeff_lr
        self._kl_loss_coeff_momentum = kl_loss_coeff_momentum
        # assert kl_target_stat in ["mean", "max", "logmax"]
        self._kl_target_stat = kl_target_stat
        self._initial_policy_opt_state_dict = self.policy.optimizer.state_dict()
        self._reset_optimizers = reset_optimizers
        self._second_penalty_loop = second_penalty_loop
        self._minibatch_kl_penalty = minibatch_kl_penalty
        self._use_beta_adam = use_beta_adam
        self._sparse_second_loop = sparse_second_loop
        self._train_calls = 0
        self._start_using_sparse_second_loop_at = 0
        self._second_loop_batch_size = second_loop_batch_size
        self._kl_loss_coeff_param = th.nn.Parameter(th.tensor(1.0))
        self._second_loop_vf = second_loop_vf
        self._multi_step_trust_region = multi_step_trust_region
        if self._use_beta_adam:
            self._kl_loss_coeff_opt = th.optim.Adam(
                [self._kl_loss_coeff_param],
                lr=self._kl_loss_coeff_lr,
            )
        else:
            self._kl_loss_coeff_opt = th.optim.SGD(
                [self._kl_loss_coeff_param],
                lr=self._kl_loss_coeff_lr,
                momentum=self._kl_loss_coeff_momentum,
            )
        self._initial_kl_loss_coeff_state_dict = self._kl_loss_coeff_opt.state_dict()
        self._max_kl_loss_coeff = maximum_kl_loss_coeff
        self._bang_bang_kl_loss_opt = bang_bang_kl_loss_opt
        self._bang_bang_reset_kl_loss_coeff = bang_bang_reset_kl_loss_coeff
        self._eval_policy = eval_policy
        self._debug_plots = debug_plots
        self._debug_pkls = debug_pkls

        self._early_stop_epoch = early_stop_epoch
        self._early_stop_across_epochs = early_stop_across_epochs
        self._reset_beta = reset_beta
        self._min_kl_loss_coeff = min_kl_loss_coeff

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self._v_trace:
            buffer_cls = VTraceRolloutBuffer
        else:
            buffer_cls = (
                DictRolloutBuffer
                if isinstance(self.observation_space, spaces.Dict)
                else RolloutBuffer
            )

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        self.historic_buffer = buffer_cls(
            self._historic_buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.historic_buffer.reset()  # initialize the arrays

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # if self._train_calls % 10 == 0:
        #     eval_return_mean, eval_return_std = evaluate_policy(self.policy, self.env)
        #     self.logger.record("rollout/EvalReturnMean", eval_return_mean)
        #     self.logger.record("rollout/EvalReturnStd", eval_return_std)
        self._train_calls += 1
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function

        entropy_losses = []
        pg_losses, value_losses = [], []
        kl_losses = []
        kl_loss_coeffs = [self._kl_loss_coeff_param.item()]
        clip_fractions = []

        continue_training = True
        self._log_avg_episode_returns()

        self._copy_over_to_history_buffer()

        if self.historic_buffer.full:
            max_idx = self.historic_buffer.buffer_size
        else:
            max_idx = self.historic_buffer.pos

        historic_obs = self.historic_buffer.observations[:max_idx].copy()
        historic_obs = self.historic_buffer.swap_and_flatten(historic_obs)
        historic_obs = self.historic_buffer.to_torch(historic_obs)
        self.logger.record("buffer/historic_obs_count", max_idx)
        self.logger.record(
            "buffer/historic_buffer_size", self.historic_buffer.buffer_size
        )
        self.logger.record("buffer/rollout_buffer_len", self.rollout_buffer.pos)

        if self._reset_optimizers and self._reset_beta:
            with th.no_grad():
                self._kl_loss_coeff_param.copy_(1.0)
            self._kl_loss_coeff_opt.load_state_dict(
                self._initial_kl_loss_coeff_state_dict
            )
        kl_divs = []
        full_kl_divs = []
        full_max_kl_divs = []

        def minibatch_step(rollout_data, use_pg_loss):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()

            # Re-sample the noise matrix because the log_std has changed
            if self.use_sde:
                self.policy.reset_noise(self.batch_size)

            full_batch_new_dist = self.policy.get_distribution(
                historic_obs
            ).distribution

            values, log_prob, entropy = self.policy.evaluate_actions(
                rollout_data.observations, actions
            )
            values = values.flatten()
            # Normalize advantage
            advantages = rollout_data.advantages
            # Normalization does not make sense if mini batchsize == 1, see GH issue #325
            if self.normalize_advantage and len(advantages) > 1:
                if self._normalize_batch_advantage:
                    advantages = (advantages - batch_adv_mean) / (batch_adv_std + 1e-8)
                else:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

            # ratio between old and new policy, should be one at the first iteration
            ratio = th.exp(log_prob - rollout_data.old_log_prob)

            if (self._minibatch_kl_penalty and use_pg_loss) or self._sparse_second_loop:
                minibatch_new_dist = self.policy.get_distribution(
                    rollout_data.observations
                ).distribution
                minibatch_old_dist = self._old_policy.get_distribution(
                    rollout_data.observations
                ).distribution
                kl_div = kl_divergence(
                    Independent(minibatch_new_dist, 1),
                    Independent(minibatch_old_dist, 1),
                )
            elif self._minibatch_kl_penalty and not use_pg_loss:
                full_batch_kl_div = kl_divergence(
                    Independent(full_batch_new_dist, 1),
                    Independent(full_batch_old_dist, 1),
                )
                kl_div = full_batch_kl_div

            else:
                # Don't accidentally use this code path
                assert False, "Are you sure you meant to full batch kl div?"
            kl_divs.append(kl_div.mean().item())

            pg_loss = -(advantages * ratio).mean()

            # Logging
            pg_losses.append(pg_loss.item())
            clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
            clip_fractions.append(clip_fraction)

            values_pred = values
            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            value_losses.append(value_loss.item())

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            entropy_losses.append(entropy_loss.item())

            loss = pg_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            kl_losses.append(kl_div.mean().item())
            if self._optimize_log_loss_coeff:
                # Optimizing the log loss coeff, therefore need to take
                # exp to get loss coeff
                loss_coeff_param = self._kl_loss_coeff_param.exp2()
            else:
                loss_coeff_param = self._kl_loss_coeff_param

            if use_pg_loss:
                loss += loss_coeff_param.detach() * kl_div.mean()
            else:
                loss = th.tensor(0.0).to(self.device)
                if self._second_loop_vf:
                    for r_data in sample_partial_buffer(self.rollout_buffer):
                        acts = r_data.actions
                        if isinstance(self.action_space, spaces.Discrete):
                            # Convert discrete action from float to long
                            acts = r_data.actions.long().flatten()
                        values, _, _ = self.policy.evaluate_actions(
                            r_data.observations, acts
                        )
                        values_pred = values
                        # Value loss using the TD(gae_lambda) target
                        value_loss = F.mse_loss(r_data.returns, values_pred.squeeze())
                        value_losses.append(value_loss.item())
                        loss += self.vf_coef * value_loss
                if self._sparse_second_loop and self._kl_target_stat != "mean":
                    need_loss = kl_div > self.target_kl
                    if need_loss.any():
                        loss += loss_coeff_param.detach() * (
                            (kl_div * need_loss).sum() / need_loss.sum()
                        )
                    else:
                        return kl_div
                else:
                    loss = loss_coeff_param.detach() * kl_div.mean()

            kl_loss_coeffs.append(self._kl_loss_coeff_param.item())
            if self._debug_plots:
                full_batch_kl_div = kl_divergence(
                    Independent(full_batch_new_dist, 1),
                    Independent(full_batch_old_dist, 1),
                )
                full_kl_divs.append(full_batch_kl_div.mean().item())
                full_max_kl_divs.append(full_batch_kl_div.max().item())

            # If optimizing the log loss coeff, then
            # self._kl_loss_coeff_param is the "log loss coeff"

            if (
                not self._bang_bang_kl_loss_opt and not use_pg_loss
            ):  # Omitting L_beta from the loss
                target_kl = th.tensor(self.target_kl)
                # Needed to move the inverse soft-plus zero
                ln_2 = th.log(th.tensor(2))
                if self._kl_target_stat == "mean":
                    loss += self._kl_loss_coeff_param * (
                        self.target_kl - kl_div.mean().detach()
                    )
                elif self._kl_target_stat == "max":
                    loss += self._kl_loss_coeff_param * (
                        self.target_kl - kl_div.max().detach()
                    )
                elif self._kl_target_stat == "logmax":
                    loss += self._kl_loss_coeff_param * (
                        th.log(target_kl) - kl_div.max().detach().log()
                    )
                elif self._kl_target_stat == "ispmax":
                    # Inverse soft-plus with zero at target_kl
                    loss += self._kl_loss_coeff_param * (
                        th.log(th.expm1(ln_2 * kl_div.max()/target_kl))
                    )
                elif self._kl_target_stat == "cubeispmax":
                    # Cube of inverse soft-plus with zero at target_kl
                    loss += self._kl_loss_coeff_param * (
                        th.log(th.expm1(ln_2 * kl_div.max()/target_kl)) ** 3
                    )
                else:
                    raise ValueError("Invalid kl_target_stat")

            self._kl_loss_coeff_opt.zero_grad()

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip grad norm
            if self._clip_grad_norm:
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
            self.policy.optimizer.step()
            self._kl_loss_coeff_opt.step()
            if self._optimize_log_loss_coeff:
                if self._kl_loss_coeff_param < np.log2(self._min_kl_loss_coeff):
                    with th.no_grad():
                        self._kl_loss_coeff_param.copy_(np.log2(self._min_kl_loss_coeff))
                elif self._kl_loss_coeff_param > np.log2(self._max_kl_loss_coeff):
                    with th.no_grad():
                        self._kl_loss_coeff_param.copy_(
                            np.log2(self._max_kl_loss_coeff)
                        )
            else:
                if self._kl_loss_coeff_param < self._min_kl_loss_coeff:
                    with th.no_grad():
                        self._kl_loss_coeff_param.copy_(self._min_kl_loss_coeff)
                    assert self._kl_loss_coeff_param >= self._min_kl_loss_coeff
                elif self._kl_loss_coeff_param > self._max_kl_loss_coeff:
                    with th.no_grad():
                        self._kl_loss_coeff_param.copy_(self._max_kl_loss_coeff)
            # kl_loss_coeffs.append(self._kl_loss_coeff_param.item())
            return kl_div

        second_penalty_loops = []
        second_penalty_skip_ratio = []
        # Save the current policy state and train
        self._old_policy.load_state_dict(self.policy.state_dict())
        second_loop_backwards = []
        second_loop_minibatches = []
        second_loop_all_skips = []
        first_loop_minibatches = []
        # train for n_epochs epochs
        self._old_policy.load_state_dict(self.policy.state_dict())
        complete_rollout_data = self.rollout_buffer.get()
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            if self._v_trace:
                with th.no_grad():
                    _, log_prob, _ = self.policy.evaluate_actions(
                        self.rollout_buffer.to_torch(self.rollout_buffer.observations),
                        self.rollout_buffer.to_torch(
                            self.rollout_buffer.actions
                        ).squeeze(),
                    )
                    self.rollout_buffer.compute_returns_and_advantage(
                        learner_log_probs=log_prob
                    )

                    _, historic_log_prob, _ = self.policy.evaluate_actions(
                        self.historic_buffer.to_torch(
                            self.historic_buffer.observations
                        ),
                        self.historic_buffer.to_torch(
                            self.historic_buffer.actions
                        ).squeeze(),
                    )
                    self.historic_buffer.compute_returns_and_advantage(
                        learner_log_probs=historic_log_prob
                    )
            batch_adv_mean = self.rollout_buffer.advantages.mean()
            batch_adv_std = self.rollout_buffer.advantages.std()
            if self._multi_step_trust_region:
                self._old_policy.load_state_dict(self.policy.state_dict())

            with th.no_grad():
                full_batch_old_dist = self._old_policy.get_distribution(
                    historic_obs
                ).distribution

            if self._reset_optimizers:
                self.policy.optimizer.load_state_dict(
                    self._initial_policy_opt_state_dict
                )
            n_first_loop_minibatch = 0
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                kl_div = minibatch_step(rollout_data=rollout_data, use_pg_loss=True)

                if (
                    self._early_stop_epoch or self._early_stop_across_epochs
                ):  # Early stop the current epoch.
                    if self._kl_target_stat == "mean":
                        kl_stat = kl_div.mean().detach()
                    elif self._kl_target_stat == "max":
                        kl_stat = kl_div.max().detach()
                    else:
                        raise ValueError("Invalid kl_target_stat")
                    if kl_stat > self.target_kl:
                        n_first_loop_minibatch += 1
                        self.logger.debug("Early stopping the current epoch.")

                        if (
                            self._early_stop_across_epochs
                        ):  # Will exit training and gather a new training batch after second penalty loop
                            continue_training = False
                            self.logger.debug("Early stopping across epochs.")
                        break
                n_first_loop_minibatch += 1

            penalty_loops = 0
            n_second_loop_minibatch = 0
            n_second_loop_backward = 0
            second_loop_skips = []

            if (
                self._bang_bang_kl_loss_opt
            ):  # Will set the kl_loss_coeff_param to maximum value before second loop.
                with th.no_grad():
                    self._kl_loss_coeff_param.copy_(self._max_kl_loss_coeff)
            if self._sparse_second_loop:
                while self._second_penalty_loop:
                    skipped_minibatches = 0
                    total_minibatches = 0
                    for historic_data in sample_partial_buffer(
                        self.historic_buffer, self._second_loop_batch_size
                    ):
                        total_minibatches += 1
                        kl_div = minibatch_step(
                            rollout_data=historic_data, use_pg_loss=False
                        )
                        if (kl_div <= self.target_kl).all():
                            second_loop_skips.append(True)
                            skipped_minibatches += 1
                        elif (
                            self._kl_target_stat == "mean"
                            and kl_div.mean() <= self.target_kl
                        ):
                            second_loop_skips.append(True)
                            skipped_minibatches += 1
                        else:
                            second_loop_skips.append(False)
                            n_second_loop_backward += 1
                        n_second_loop_minibatch += 1
                    if total_minibatches == 0:
                        second_penalty_skip_ratio.append(1)
                    else:
                        second_penalty_skip_ratio.append(
                            skipped_minibatches / total_minibatches
                        )
                    penalty_loops += 1
                    if skipped_minibatches == total_minibatches:
                        if (
                            self._bang_bang_kl_loss_opt and penalty_loops == 1
                        ):  # Only reset when constraint is not broken during the whole batch.
                            with th.no_grad():
                                self._kl_loss_coeff_param.copy_(1.0)
                        break
                    if penalty_loops > SECOND_PENALTY_LOOP_MAX:
                        print("Too many loops")
                        self.policy.load_state_dict(self._old_policy.state_dict())
                        self.logger.record("train/broke_loop", 1)
                        break
            else:
                while self._second_penalty_loop:
                    if self._kl_target_stat == "mean":
                        if kl_div.mean() <= self.target_kl:
                            break
                    elif self._kl_target_stat == "max":
                        if kl_div.max() <= self.target_kl:
                            break
                    penalty_loops += 1
                    for historic_data in sample_partial_buffer(self.historic_buffer):
                        # This rollout_data is not used to compute KL div
                        kl_div = minibatch_step(
                            rollout_data=historic_data, use_pg_loss=False
                        )
                        second_penalty_skip_ratio.append(0)
                    if penalty_loops > SECOND_PENALTY_LOOP_MAX:
                        print("Too many loops")
                        self.policy.load_state_dict(self._old_policy.state_dict())
                        self.logger.record("train/broke_loop", 1)
                        break

            if (
                self._bang_bang_kl_loss_opt and self._bang_bang_reset_kl_loss_coeff
            ):  # Will set the kl_loss_coeff_param to init value after second loop.
                with th.no_grad():
                    self._kl_loss_coeff_param.copy_(1.0)

            second_penalty_loops.append(penalty_loops)
            first_loop_minibatches.append(n_first_loop_minibatch)
            second_loop_minibatches.append(n_second_loop_minibatch)
            second_loop_backwards.append(n_second_loop_backward)
            second_loop_all_skips.append(second_loop_skips)

            if not continue_training:
                break

        if self._debug_plots:
            with th.no_grad():
                full_batch_old_dist = self._old_policy.get_distribution(
                    historic_obs
                ).distribution
                full_batch_new_dist = self.policy.get_distribution(
                    historic_obs
                ).distribution
            full_batch_kl_div = kl_divergence(
                Independent(full_batch_new_dist, 1),
                Independent(full_batch_old_dist, 1),
            )
            kl_loss_coeffs.append(self._kl_loss_coeff_param.item())
            full_kl_divs.append(full_batch_kl_div.mean().item())
            full_max_kl_divs.append(full_batch_kl_div.max().item())

            debug_data = {
                "kl_divs": kl_divs,
                "full_kl_divs": full_kl_divs,
                "full_max_kl_divs": full_max_kl_divs,
                "first_loop_minibatches": first_loop_minibatches,
                "second_loop_minibatches": second_loop_minibatches,
                "second_loop_backwards": second_loop_backwards,
                "second_loop_all_skips": second_loop_all_skips,
                "kl_loss_coeffs": kl_loss_coeffs,
            }

            gradient_steps_plot(
                debug_data, f"{self.logger.dir}/grad_steps_{self._train_calls}.svg"
            )
            kl_div_plot(
                debug_data,
                f"{self.logger.dir}/batch_mean_kl_divs_{self._train_calls:05}.svg",
            )

            if self._debug_pkls:
                with open(f"{self.logger.dir}/step_{self._train_calls}.pkl", "wb") as f:
                    import pickle

                    pickle.dump(
                        debug_data,
                        f,
                    )

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/kl_loss", np.mean(kl_losses))
        self.logger.record("train/max_batch_kl_div", np.max(kl_divs))
        self.logger.record("train/mean_batch_kl_div", np.mean(kl_divs))
        self.logger.record("train/final_kl_div", kl_divs[-1])
        self.logger.record("train/final_max_kl_div", kl_div.max().item())
        self.logger.record("train/kl_div", np.mean(kl_divs))
        self.logger.record(
            "train/first_loop_minibatches", np.mean(first_loop_minibatches)
        )
        self.logger.record(
            "train/second_loop_minibatches", np.mean(second_loop_minibatches)
        )
        self.logger.record(
            "train/first_loop_minibatches_min", np.min(first_loop_minibatches)
        )
        self.logger.record(
            "train/second_loop_minibatches_min", np.min(second_loop_minibatches)
        )
        self.logger.record(
            "train/first_loop_minibatches_max", np.max(first_loop_minibatches)
        )
        self.logger.record(
            "train/second_loop_minibatches_max", np.max(second_loop_minibatches)
        )
        self.logger.record(
            "train/first_loop_minibatches_total", np.sum(first_loop_minibatches)
        )
        self.logger.record(
            "train/second_loop_minibatches_total", np.sum(second_loop_minibatches)
        )
        self.logger.record(
            "train/minibatches_total",
            np.sum(first_loop_minibatches) + np.sum(second_loop_minibatches),
        )

        if second_penalty_loops:
            self.logger.record(
                "train/final_second_penalty_loops", second_penalty_loops[-1]
            )
            self.logger.record(
                "train/mean_second_penalty_loops", np.mean(second_penalty_loops)
            )
            self.logger.record(
                "train/max_second_penalty_loops", np.max(second_penalty_loops)
            )
        if second_penalty_skip_ratio:
            self.logger.record(
                "train/mean_second_penalty_skip_ratio",
                np.mean(second_penalty_skip_ratio),
            )
            self.logger.record(
                "train/max_second_penalty_skip_ratio", np.max(second_penalty_skip_ratio)
            )
            self.logger.record(
                "train/min_second_penalty_skip_ratio", np.min(second_penalty_skip_ratio)
            )
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/advantages", batch_adv_mean)
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/kl_loss_coeff", self._kl_loss_coeff_param.item())
        self.logger.record("train/kl_loss_coeff_avg", np.mean(kl_loss_coeffs))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "KLPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _log_avg_episode_returns(self):
        eps_start_idx = np.where(self.rollout_buffer.episode_starts.flatten() == 1)[0]
        returns = []

        for i, start in enumerate(eps_start_idx[:-1]):  # only use full episodes
            end_idx = eps_start_idx[i + 1]
            returns.append(self.rollout_buffer.rewards[start:end_idx].sum())
        avg_return = np.mean(returns)
        self.logger.record("rollout/AverageReturn", avg_return)
        self.logger.record("rollout/FullEpisodeCount", len(returns))

    def _copy_over_to_history_buffer(self):
        vars_to_copy = [
            "observations",
            "actions",
            "rewards",
            "episode_starts",
            "values",
            "log_probs",
        ]
        buffer_len = self.rollout_buffer.pos
        remaining_space = self.historic_buffer.buffer_size - self.historic_buffer.pos

        assert buffer_len != 0, "Empty batch being copied over to the historic buffer"
        
        if remaining_space < buffer_len and isinstance(
            self.historic_buffer, VTraceRolloutBuffer
        ):
            # Free up space to maintain full episodes & unifrom reward dist. over episodes
            self.historic_buffer._on_out_of_space(
                space_needed=buffer_len - remaining_space
            )
            remaining_space = (
                self.historic_buffer.buffer_size - self.historic_buffer.pos
            )

        if remaining_space < buffer_len:
            for var in vars_to_copy:
                self.historic_buffer.__getattribute__(var)[
                    self.historic_buffer.pos : self.historic_buffer.pos
                    + remaining_space
                ] = self.rollout_buffer.__getattribute__(var)[:remaining_space].copy()

            # Overwrite the remaining from the start
            for var in vars_to_copy:
                self.historic_buffer.__getattribute__(var)[
                    0 : (buffer_len - remaining_space)
                ] = self.rollout_buffer.__getattribute__(var)[remaining_space:].copy()

            self.historic_buffer.pos = buffer_len - remaining_space
            self.historic_buffer.full = True
        else:
            for var in vars_to_copy:
                self.historic_buffer.__getattribute__(var)[
                    self.historic_buffer.pos : (self.historic_buffer.pos + buffer_len)
                ] = self.rollout_buffer.__getattribute__(var)[:].copy()
            self.historic_buffer.pos = (
                self.historic_buffer.pos + buffer_len
            ) % self.historic_buffer.buffer_size
            if self.historic_buffer.pos == 0:
                self.historic_buffer.full = True

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        episode_steps = 0
        n_successes = 0
        n_episodes = 0
        success_flag = False
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            if infos[0].get("success", False):
                success_flag = True

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1
            episode_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
            if isinstance(rollout_buffer, VTraceRolloutBuffer):
                bootstrap_value = None
                if episode_steps >= self.max_path_length or n_steps == n_rollout_steps:
                    with th.no_grad():
                        next_obs_tensor = obs_as_tensor(new_obs, self.device)
                        _, bootstrap_value, _ = self.policy(next_obs_tensor)

                rollout_buffer.add(
                    self._last_obs,
                    actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                    bootstrap_value=bootstrap_value,
                )
            else:
                rollout_buffer.add(
                    self._last_obs,
                    actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                )
            if episode_steps >= self.max_path_length:
                self._last_obs = self.env.reset()
                self._last_episode_starts = True
                episode_steps = 0
                n_episodes += 1
                if success_flag:
                    n_successes += 1
                success_flag = False
            else:
                self._last_obs = new_obs
                self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        # Reset env at end of collection
        self._last_obs = self.env.reset()
        self._last_episode_starts = True
        self.logger.record("rollout/SuccessRate", n_successes / n_episodes)
        return True


def sample_partial_buffer(buffer, batch_size: Optional[int] = None):
    if buffer.full:
        indices = np.random.permutation(buffer.buffer_size * buffer.n_envs)
    else:
        indices = np.random.permutation(buffer.pos * buffer.n_envs)

    # Return everything, don't create minibatches
    if batch_size is None:
        batch_size = buffer.buffer_size * buffer.n_envs
    assert batch_size is not None

    start_idx = 0
    while start_idx < len(indices):
        batch_inds = indices[start_idx : start_idx + batch_size]
        data = (
            buffer.observations[batch_inds],
            buffer.actions[batch_inds],
            buffer.values[batch_inds],
            buffer.log_probs[batch_inds],
            buffer.advantages[batch_inds],
            buffer.returns[batch_inds],
        )
        yield RolloutBufferSamples(
            *tuple([buffer.to_torch(d).squeeze(1) for d in data])
        )
        start_idx += batch_size


MAX_GRAD_STEP = 0


def gradient_steps_plot(debug_data, filename):
    global MAX_GRAD_STEP
    import matplotlib

    matplotlib.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 14,
        }
    )
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    import matplotlib.pyplot as plt

    plt.clf()
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("Gradient Steps")
    ax1.set_ylabel("Max KL Divergence", color=color)
    ax1.plot(debug_data["full_max_kl_divs"], color=color)
    # ax1.plot(debug_data['kl_divs'], color='orange')
    ax1.tick_params(axis="y", labelcolor=color)
    # ax1.set_aspect(2)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel(r"$\beta$", color=color)  # we already handled the x-label with ax1
    ax2.plot(debug_data["kl_loss_coeffs"][1:], color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    grad_step = 0
    for epoch in range(len(debug_data["first_loop_minibatches"])):
        grad_step += debug_data["first_loop_minibatches"][epoch]
        ax1.axvspan(
            grad_step,
            grad_step + debug_data["second_loop_backwards"][epoch],
            alpha=0.25,
            color="green",
        )
        grad_step += debug_data["second_loop_backwards"][epoch]
    MAX_GRAD_STEP = max(grad_step, MAX_GRAD_STEP)
    ax1.hlines(0.2, xmin=0, xmax=MAX_GRAD_STEP, color="gray", linestyles="dashed")
    # ax1.vlines(len(debug_data['full_max_kl_divs']) - debug_data['second_loop_minibatches'][0], ymin=0, ymax=0.2, color="green")
    # ax1.set_aspect(1.5)
    # ax2.set_aspect(1.5)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(filename)


def kl_div_plot(debug_data, filename):
    import matplotlib

    matplotlib.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 14,
        }
    )
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.clf()
    sns.kdeplot(data=debug_data["kl_divs"])
    plt.savefig(filename)
