from typing import Any, Dict, List, Type

import numpy as np
import torch
from torch import nn
from torch.distributions import kl_divergence

from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import A2CPolicy
from tianshou.utils.net.common import ActorCritic


class FixPOPolicy(A2CPolicy):
    r"""Implementation of Proximal Policy Optimization. arXiv:1707.06347.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param int fixup_batchsize: Number of minibatches per batch to
        perform in second loop. Default to 1024.
    :param float beta_lr: :math:`\gamma_{\beta}` in xPPO paper.
        Default to 0.01.
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param float eps_kl: :math:`\epsilon_{KL}` in :math:`L_{\beta}` in xPPO paper.
        Default to 0.2.
    :param float target_coeff: :math:`c_{KL}` in :math:`L_{\beta}` in xPPO paper.
        Default to 2.
    :param float eps_clip: :math:`\epsilon` in :math:`L_{CLIP}` in the original
        paper. Only used when value_clip is enabled. Default to 0.2.
    :param bool value_clip: a parameter mentioned in arXiv:1811.02553v3 Sec. 4.1.
        Default to True.
    :param bool fixup_loop: whether to run fixup loop (at all).
        Default to True.
    :param bool fixup_every_repeat: whether to run fixup loop every epoch.
        Default to True.
    :param bool advantage_normalization: whether to do per mini-batch advantage
        normalization. Default to True.
    :param bool recompute_advantage: whether to recompute advantage every update
        repeat according to https://arxiv.org/pdf/2006.05990.pdf Sec. 3.5.
        Default to False.
    :param float vf_coef: weight for value loss. Default to 0.5.
    :param float ent_coef: weight for entropy loss. Default to 0.01.
    :param float max_grad_norm: clipping gradients in back propagation. Default to
        None.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close
        to 1, also normalize the advantage to Normal(0, 1). Default to False.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the model;
        should be as large as possible within the memory constraint. Default to 256.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        *,
        fixup_batchsize: int = 1024,
        beta_lr: float = 0.01,
        eps_kl: float = 0.2,
        eps_clip: float = 0.2,
        value_clip: bool = False,
        fixup_loop: bool = True,
        fixup_every_repeat: bool = True,
        target_coeff: float = 2.,
        init_beta: float = 1.,
        kl_target_stat: str = "max",
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, optim, dist_fn, **kwargs)
        self._eps_clip = eps_clip  # Only used for vf clipping
        self._eps_kl = eps_kl
        self._value_clip = value_clip
        self._norm_adv = advantage_normalization
        self._recompute_adv = recompute_advantage
        self._actor_critic: ActorCritic
        self._beta = torch.nn.Parameter(torch.tensor(init_beta))
        self._beta_optim = torch.optim.Adam(params=[self._beta], lr=beta_lr)
        self._fixup_batchsize = fixup_batchsize
        self._fixup_every_repeat = fixup_every_repeat
        self._kl_target_stat = kl_target_stat
        self._fixup_loop = fixup_loop
        self._target_coeff = target_coeff

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        if self._recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        with torch.no_grad():
            result = self(batch)
            # Move batch dimension to start
            batch.logits = result.logits.transpose(0, 1)
            batch.logp_old = result.dist.log_prob(batch.act)
        return batch

    def _optimize_beta(self, kl_div: torch.Tensor):
        self._beta_optim.zero_grad()
        if self._kl_target_stat == "max":
            beta_loss = self._beta * (self._eps_kl - self._target_coeff * kl_div.detach().max())
        elif self._kl_target_stat == "mean":
            beta_loss = self._beta * (self._eps_kl - self._target_coeff * kl_div.detach().mean())
        else:
            raise ValueError("Unknown kl_target_stat", self._kl_target_stat)
        # This backward pass only affects self._beta
        beta_loss.backward()
        self._beta_optim.step()
        if self._beta < 0:
            with torch.no_grad():
                self._beta.copy_(0.)
        return beta_loss.item()

    def _violates_constraint(self, kl_div: torch.Tensor):
        if self._kl_target_stat == "max":
            return (kl_div > self._eps_kl).any()
        elif self._kl_target_stat == "mean":
            return kl_div.mean() > self._eps_kl
        else:
            raise ValueError("Unknown kl_target_stat", self._kl_target_stat)

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, pg_losses, vf_losses, ent_losses, beta_losses, kl_losses = [], [], [], [], [], []
        fixup_grad_steps = 0
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(batch_size, merge_last=True):
                # calculate loss for actor
                dist = self(minibatch).dist
                if self._norm_adv:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv -
                                     mean) / (std + self._eps)  # per-batch norm
                ratio = (dist.log_prob(minibatch.act) -
                         minibatch.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                pg_loss = -(ratio * minibatch.adv).mean()
                old_dist = self.dist_fn(*minibatch.logits.transpose(0, 1))
                kl_div = kl_divergence(old_dist, dist)
                kl_loss = self._beta.detach() * kl_div.mean()
                # calculate loss for critic
                value = self.critic(minibatch.obs).flatten()
                if self._value_clip:
                    v_clip = minibatch.v_s + \
                        (value - minibatch.v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = pg_loss + self._weight_vf * vf_loss \
                    - self._weight_ent * ent_loss + kl_loss
                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), max_norm=self._grad_norm
                    )
                self.optim.step()
                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                kl_losses.append(kl_loss.item())
                losses.append(loss.item())

                beta_losses.append(self._optimize_beta(kl_div))

            if self._fixup_loop and (self._fixup_every_repeat or step + 1 == repeat):
                while True:  # until constriant satisfied
                    constraint_satisfied = True
                    for minibatch in batch.split(self._fixup_batchsize, merge_last=True):
                        dist = self(minibatch).dist
                        old_dist = self.dist_fn(*minibatch.logits.transpose(0, 1))
                        kl_div = kl_divergence(old_dist, dist)
                        if self._violates_constraint(kl_div):
                            constraint_satisfied = False
                            fixup_grad_steps += 1

                            kl_loss = self._beta.detach() * kl_div.mean()
                            self.optim.zero_grad()
                            self._beta_optim.zero_grad()
                            kl_loss.backward()
                            if self._grad_norm:  # clip large gradient
                                nn.utils.clip_grad_norm_(
                                    self._actor_critic.parameters(), max_norm=self._grad_norm
                                )
                            self.optim.step()

                            beta_losses.append(self._optimize_beta(kl_div))
                    if constraint_satisfied:
                        break

        return {
            "loss": losses,
            "loss/pg": pg_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
            "loss/kl": kl_losses,
            "loss/beta": beta_losses,
            "fixup_grad_steps": fixup_grad_steps,
            "beta": self._beta.item(),
        }
