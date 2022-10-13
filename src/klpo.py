"""KL-Regularized Policy Optimization (KLPO)."""
import torch

from garage.torch.algos import VPG
from garage.torch.optimizers import (MinibatchOptimizer, EpisodeBatchOptimizer)
from garage.torch import is_policy_recurrent


class KLPO(VPG):
    """KL-Regularized Policy Optimization (KLPO).

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.Policy): Policy.
        value_function (garage.torch.value_functions.ValueFunction): The value
            function.
        sampler (garage.sampler.Sampler): Sampler.
        policy_optimizer (garage.torch.optimizer.MinibatchOptimizer): Optimizer
            for policy.
        vf_optimizer (garage.torch.optimizer.MinibatchOptimizer): Optimizer for
            value function.
        lr_clip_range (float): The limit on the likelihood ratio between
            policies.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 value_function,
                 sampler,
                 batch_size,
                 policy_optimizer=None,
                 vf_optimizer=None,
                 lr_clip_range=None,
                 lr_loss_coeff=0.01,
                 lr_sq_loss_coeff=None,
                 discount=0.99,
                 gae_lambda=0.97,
                 center_adv=True,
                 positive_adv=False,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy',
                 recurrent=None):
        if recurrent is None:
            recurrent = is_policy_recurrent(policy, env_spec)

        if policy_optimizer is None:
            if recurrent:
                policy_optimizer = EpisodeBatchOptimizer(
                    (torch.optim.Adam, dict(lr=2.5e-4)),
                    policy,
                    max_optimization_epochs=10,
                    minibatch_size=64)
            else:
                policy_optimizer = MinibatchOptimizer(
                    (torch.optim.Adam, dict(lr=2.5e-4)),
                    policy,
                    max_optimization_epochs=10,
                    minibatch_size=64)

        if vf_optimizer is None:
            if recurrent:
                vf_optimizer = EpisodeBatchOptimizer(
                    (torch.optim.Adam, dict(lr=2.5e-4)),
                    value_function,
                    max_optimization_epochs=10,
                    minibatch_size=64)
            else:
                vf_optimizer = MinibatchOptimizer(
                    (torch.optim.Adam, dict(lr=2.5e-4)),
                    value_function,
                    max_optimization_epochs=10,
                    minibatch_size=64)

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         value_function=value_function,
                         sampler=sampler,
                         batch_size=batch_size,
                         policy_optimizer=policy_optimizer,
                         vf_optimizer=vf_optimizer,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method,
                         recurrent=recurrent)

        self._lr_clip_range = lr_clip_range
        self._lr_loss_coeff = lr_loss_coeff
        self._lr_sq_loss_coeff = lr_sq_loss_coeff

    def hparam_ranges(self):
        hparam_ranges = {}
        if self._lr_clip_range is not None:
            hparam_ranges['lr_clip_range'] = (0., 1.)
        if self._lr_loss_coeff is not None:
            hparam_ranges['lr_loss_coeff'] = (0., 1.)
        if self._lr_sq_loss_coeff is not None:
            hparam_ranges['lr_sq_loss_coeff'] = (0., 1.)
        return hparam_ranges

    def get_hparams(self):
        hparams = {}
        if self._lr_clip_range is not None:
            hparams['lr_clip_range'] = self._lr_clip_range
        if self._lr_loss_coeff is not None:
            hparams['lr_loss_coeff'] = self._lr_loss_coeff
        if self._lr_sq_loss_coeff is not None:
            hparams['lr_sq_loss_coeff'] = self._lr_sq_loss_coeff
        return hparams

    def set_hparams(self, hparams):
        if 'lr_clip_range' in hparams:
            self._lr_clip_range = hparams['lr_clip_range']
        if 'lr_loss_coeff' in hparams:
            self._lr_loss_coeff = hparams['lr_loss_coeff']
        if 'lr_sq_loss_coeff' in hparams:
            self._lr_sq_loss_coeff = hparams['lr_sq_loss_coeff']

    def _compute_objective(self, advantages, obs, actions, rewards):
        r"""Compute objective value.

        Args:
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N \dot [T], )`.
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N \dot [T], )`.

        Returns:
            torch.Tensor: Calculated objective values
                with shape :math:`(N \dot [T], )`.

        """
        # Compute constraint
        with torch.no_grad():
            old_ll = self._old_policy(obs)[0].log_prob(actions)
        new_ll = self.policy(obs)[0].log_prob(actions)

        likelihood_ratio = (new_ll - old_ll).exp()

        # Calculate surrogate
        surrogate = likelihood_ratio * advantages

        pg_loss = surrogate

        if self._lr_clip_range is not None:
            # Clipping the constraint
            likelihood_ratio_clip = torch.clamp(likelihood_ratio,
                                                min=1 - self._lr_clip_range,
                                                max=1 + self._lr_clip_range)

            # Calculate surrotate clip
            surrogate_clip = likelihood_ratio_clip * advantages

            pg_loss = torch.min(surrogate, surrogate_clip)

        loss = pg_loss

        if self._lr_loss_coeff is not None:
            loss += self._lr_loss_coeff * -likelihood_ratio
        if self._lr_sq_loss_coeff is not None:
            loss += self._lr_sq_loss_coeff * -(likelihood_ratio ** 2)

        return loss
