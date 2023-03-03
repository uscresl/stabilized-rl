import copy
import warnings
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import gym
import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
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
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        normalize_batch_advantage: bool = True,
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
        kl_loss_exp: float = 1.0,
        kl_loss_coeff_lr: float,
        kl_loss_coeff_momentum: float,
        kl_target_stat: str,
        optimize_log_loss_coeff: bool,
    ):

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
        self._kl_loss_coeff_param = th.nn.Parameter(th.tensor(1.0))
        self._kl_loss_coeff_lr = kl_loss_coeff_lr
        self._kl_loss_coeff_momentum = kl_loss_coeff_momentum
        assert kl_target_stat in ["mean", "max"]
        self._kl_target_stat = kl_target_stat
        self._kl_loss_exp = kl_loss_exp

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        self.historic_buffer = RolloutBuffer(
            min(64_000, self.n_steps * 10),
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
        kl_loss_coeffs = []
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

        self._kl_loss_coeff_param = th.nn.Parameter(th.tensor(1.0))
        kl_loss_coeff_opt = th.optim.SGD(
            [self._kl_loss_coeff_param],
            lr=self._kl_loss_coeff_lr,
            momentum=self._kl_loss_coeff_momentum,
        )

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            kl_divs = []
            # Do a complete pass on the rollout buffer
            if self.normalize_advantage and self._normalize_batch_advantage:
                batch_adv_mean = self.rollout_buffer.advantages.mean()
                batch_adv_std = self.rollout_buffer.advantages.std()
            # Save the current policy state and train
            self._old_policy.load_state_dict(self.policy.state_dict())
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                with th.no_grad():
                    old_dist = self._old_policy.get_distribution(
                        historic_obs
                    ).distribution

                new_dist = self.policy.get_distribution(historic_obs).distribution

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    if self._normalize_batch_advantage:
                        advantages = (advantages - batch_adv_mean) / (
                            batch_adv_std + 1e-8
                        )
                    else:
                        advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
                        )

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                kl_div = kl_divergence(
                    Independent(new_dist, 1),
                    Independent(old_dist, 1),
                )
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

                loss = (
                    pg_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                )
                if self.target_kl is not None:
                    kl_loss = (kl_div / self.target_kl) ** self._kl_loss_exp
                    kl_losses.append(kl_loss.mean().item())
                    if self._optimize_log_loss_coeff:
                        # Optimizing the log loss coeff, therefore need to take
                        # exp to get loss coeff
                        loss_coeff_param = self._kl_loss_coeff_param.exp2()
                    else:
                        loss_coeff_param = self._kl_loss_coeff_param
                    loss += loss_coeff_param.detach() * kl_loss.mean()
                    # If optimizing the log loss coeff, then
                    # self._kl_loss_coeff_param is the "log loss coeff"
                    if self._kl_target_stat == "mean":
                        loss += self._kl_loss_coeff_param * (
                            self.target_kl - kl_div.mean().detach()
                        )
                    elif self._kl_target_stat == "max":
                        loss += self._kl_loss_coeff_param * (
                            self.target_kl - kl_div.max().detach()
                        )
                    else:
                        raise ValueError("Invalid kl_target_stat")
                    kl_loss_coeff_opt.zero_grad()

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                if self._clip_grad_norm:
                    th.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.max_grad_norm
                    )
                self.policy.optimizer.step()
                if self.target_kl is not None:
                    kl_loss_coeff_opt.step()
                    if self._optimize_log_loss_coeff:
                        if self._kl_loss_coeff_param < 1 + MIN_KL_LOSS_COEFF:
                            with th.no_grad():
                                self._kl_loss_coeff_param.copy_(1 + MIN_KL_LOSS_COEFF)
                            assert self._kl_loss_coeff_param >= 1 + MIN_KL_LOSS_COEFF
                    else:
                        if self._kl_loss_coeff_param < MIN_KL_LOSS_COEFF:
                            with th.no_grad():
                                self._kl_loss_coeff_param.copy_(MIN_KL_LOSS_COEFF)
                            assert self._kl_loss_coeff_param >= MIN_KL_LOSS_COEFF
                kl_loss_coeffs.append(self._kl_loss_coeff_param.item())

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/kl_loss", np.mean(kl_losses))
        self.logger.record("train/final_kl_div", kl_divs[-1])
        self.logger.record("train/final_max_kl_div", kl_div.max().item())
        self.logger.record("train/kl_div", np.mean(kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
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

        if remaining_space <= buffer_len:
            for var in vars_to_copy:
                self.historic_buffer.__getattribute__(var)[
                    self.historic_buffer.pos : self.historic_buffer.pos
                    + remaining_space
                ] = self.rollout_buffer.__getattribute__(var)[:remaining_space]

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
