import clize

from garage import wrap_experiment
from torch import norm
from klpo_stable_baselines_algo import KLPOStbl
from stable_baselines3.common.logger import configure


@wrap_experiment(name_parameters="all", use_existing_dir=True)
def klpo_stbl(
    ctxt=None,
    env="HalfCheetah-v3",
    normalize_batch_advantage=True,
    n_steps=4096,
    gae_lambda=0.97,
    vf_arch=[64, 64],
    clip_grad_norm=False,
    total_steps=3_000_000,
    note="buffer_kl_loss",
    *,
    seed,
    batch_size,
    target_kl,
    ent_coef,
    kl_loss_coeff_lr,
    kl_loss_coeff_momentum,
    kl_target_stat,
    optimize_log_loss_coeff,
    reset_optimizers,
    minibatch_kl_penalty,
    use_beta_adam,
    sparse_second_loop,
    normalize_advantage,
    second_loop_batch_size,
    second_penalty_loop,
    historic_buffer_size,
    second_loop_vf,
):
    model = KLPOStbl(
        "MlpPolicy",
        env,
        seed=seed,
        normalize_batch_advantage=normalize_batch_advantage,
        n_steps=n_steps,
        policy_kwargs={"net_arch": [{"vf": vf_arch, "pi": [64, 64]}]},
        gae_lambda=gae_lambda,
        clip_grad_norm=clip_grad_norm,
        target_kl=target_kl,
        ent_coef=ent_coef,
        batch_size=batch_size,
        kl_loss_coeff_lr=kl_loss_coeff_lr,
        kl_target_stat=kl_target_stat,
        kl_loss_coeff_momentum=kl_loss_coeff_momentum,
        optimize_log_loss_coeff=optimize_log_loss_coeff,
        reset_optimizers=reset_optimizers,
        minibatch_kl_penalty=minibatch_kl_penalty,
        use_beta_adam=use_beta_adam,
        sparse_second_loop=sparse_second_loop,
        normalize_advantage=normalize_advantage,
        second_loop_batch_size=second_loop_batch_size,
        second_penalty_loop=second_penalty_loop,
        historic_buffer_size=historic_buffer_size,
        second_loop_vf=second_loop_vf,
    )

    new_logger = configure(ctxt.snapshot_dir, ["stdout", "log", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_steps)


if __name__ == "__main__":

    @clize.run
    def main(
        *,
        seed: int,
        env: str,
        log_dir: str,
        target_kl: float,
        note: str,
        ent_coef: float = 0.0,
        kl_loss_coeff_lr: float = 3.0,
        kl_loss_coeff_momentum: float = 0.9999, # Not used with adam
        kl_target_stat: str = "max",
        n_steps: int = 4096,
        batch_size: int = 512,
        optimize_log_loss_coeff: bool = False,
        reset_optimizers: bool = True,
        minibatch_kl_penalty: bool = True,
        use_beta_adam: bool = True,
        sparse_second_loop: bool = True,
        normalize_advantage: bool = False,
        second_penalty_loop: bool = True,
        total_steps: int=3_000_000,
        second_loop_batch_size: int=16000,
        historic_buffer_size: int = 32000,
        second_loop_vf: bool = False,
    ):
        klpo_stbl(
            dict(log_dir=log_dir),
            seed=seed,
            env=env,
            target_kl=target_kl,
            note=note,
            ent_coef=ent_coef,
            kl_loss_coeff_lr=kl_loss_coeff_lr,
            kl_loss_coeff_momentum=kl_loss_coeff_momentum,
            kl_target_stat=kl_target_stat,
            n_steps=n_steps,
            batch_size=batch_size,
            optimize_log_loss_coeff=optimize_log_loss_coeff,
            reset_optimizers=reset_optimizers,
            minibatch_kl_penalty=minibatch_kl_penalty,
            use_beta_adam=use_beta_adam,
            sparse_second_loop=sparse_second_loop,
            normalize_advantage=normalize_advantage,
            normalize_batch_advantage=normalize_advantage,
            total_steps=total_steps,
            second_loop_batch_size=second_loop_batch_size,
            second_penalty_loop=second_penalty_loop,
            historic_buffer_size=historic_buffer_size,
            second_loop_vf=second_loop_vf,
        )
