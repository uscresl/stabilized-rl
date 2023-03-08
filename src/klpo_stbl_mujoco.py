import clize

from garage import wrap_experiment
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
    target_kl,
    ent_coef,
    kl_loss_exp,
    kl_loss_coeff_lr,
    kl_loss_coeff_momentum: float,
    kl_target_stat,
    optimize_log_loss_coeff,
    reset_policy_optimizer,
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
        kl_loss_exp=kl_loss_exp,
        kl_loss_coeff_lr=kl_loss_coeff_lr,
        kl_target_stat=kl_target_stat,
        kl_loss_coeff_momentum=kl_loss_coeff_momentum,
        optimize_log_loss_coeff=optimize_log_loss_coeff,
        reset_policy_optimizer=reset_policy_optimizer,
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
        note: str = "",
        ent_coef: float = 0.0,
        kl_loss_coeff_lr: float = 0.1,
        kl_loss_coeff_momentum: float = 0.99,
        kl_target_stat: str = "max",
        kl_loss_exp: float = 1.0,
        n_steps: int = 4096,
        optimize_log_loss_coeff: bool = False,
        reset_policy_optimizer=False,
    ):
        klpo_stbl(
            dict(log_dir=log_dir),
            seed=seed,
            env=env,
            target_kl=target_kl,
            note=note,
            ent_coef=ent_coef,
            kl_loss_exp=kl_loss_exp,
            kl_loss_coeff_lr=kl_loss_coeff_lr,
            kl_loss_coeff_momentum=kl_loss_coeff_momentum,
            kl_target_stat=kl_target_stat,
            n_steps=n_steps,
            optimize_log_loss_coeff=optimize_log_loss_coeff,
            reset_policy_optimizer=reset_policy_optimizer,
        )
