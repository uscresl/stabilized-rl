import numpy as np
from dowel import logger, tabular, CsvOutput
from fake_algo import FakeAlgo
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
import clize
from matplotlib import pyplot as plt
from tqdm import tqdm


output = CsvOutput("gp_fit.csv")
logger.add_output(output)


def experiment(epochs=100, num_hparams_sampled=500, delta=0.5, seed=100):
    #kernel = RBF(length_scale=.25, length_scale_bounds="fixed")
    # kernel = RBF(length_scale=.5, length_scale_bounds="fixed")
    # kernel = (Matern(length_scale=0.1, nu=0.5, length_scale_bounds="fixed") +
              # (Matern(length_scale=0.1, nu=1.5, length_scale_bounds="fixed")
                # * WhiteKernel(1.0)))
    # kernel = (RBF(length_scale=0.1, length_scale_bounds="fixed") +
              # (RBF(length_scale=0.1, length_scale_bounds="fixed")
                # * WhiteKernel(1.0)))
    kernel = (Matern(length_scale=0.25, nu=1.5, length_scale_bounds="fixed") +
              (Matern(length_scale=0.25, nu=1.5, length_scale_bounds="fixed")
                * WhiteKernel()))
    np.random.seed = seed
    gp = GaussianProcessRegressor(kernel=kernel, random_state=seed)
    hparam_dims = 1
    algo = FakeAlgo(n_hparams=hparam_dims)
    init_hparam = np.random.sample(size=(hparam_dims,))
    init_perf = curr_perf = algo.step(init_hparam)

    gp.fit(init_hparam.reshape(1, -1), [init_perf])

    sampled_x = [init_hparam]
    sampled_y = [init_perf]

    for step in tqdm(range(1, epochs + 1)):
        sampled_hparams = np.random.sample(size=(num_hparams_sampled, hparam_dims))
        pred_mu, pred_std = gp.predict(sampled_hparams, return_std=True)
        # nu = 1
        # nu = 4
        nu = 8
        tau_t = 2 * np.log(
            step ** (hparam_dims / 2 + 2) * np.pi**2 / (3 * delta)
        )  # 2*log(t^(d/2+2)π^2/(3δ))
        k_t = np.sqrt(nu * tau_t)
        ucb = pred_mu + k_t * pred_std
        max_idx = np.argmax(ucb)

        next_hparam = sampled_hparams[max_idx]
        perf = algo.step(next_hparam)
        perf_improv, curr_perf = perf - curr_perf, perf

        tabular.record("step", step)
        tabular.record("perf_improv", perf_improv)
        tabular.record("curr_perf", curr_perf)
        tabular.record("perf_pred_diff", perf_improv - pred_mu[max_idx])
        logger.log(tabular)
        logger.dump_all(step=step)

        sampled_x.append(next_hparam)
        sampled_y.append(perf_improv)

        gp.fit(sampled_x, sampled_y)
        plot_gpr(gp, sampled_x, sampled_y, step, algo)


def plot_gpr(gp, X_train, Y_train, step, algo):
    #X_bel = np.linspace(np.min(X_train), np.max(X_train), num=1_000).reshape(-1, 1)
    X_bel = np.linspace(0, 1, num=1_000).reshape(-1, 1)
    mean_bel, std_bel = gp.predict(X_bel, return_std=True)

    plt.clf()
    plt.ylim((-5, 5))
    plt.xlim((0, 1))
    plt.scatter(X_train, Y_train, label="Observations")
    plt.plot(X_bel, mean_bel, label="Mean prediction")
    true_mean = np.array([algo.true_mean(x) for x in X_bel])
    plt.plot(X_bel, true_mean, label="True mean")
    true_noise = np.array([algo.noise_scale(x) for x in X_bel])
    plt.fill_between(
        X_bel.ravel(),
        true_mean - 1.96 * true_noise,
        true_mean + 1.96 * true_noise,
        alpha=0.5,
        label=r"True noise",
    )
    plt.fill_between(
        X_bel.ravel(),
        mean_bel - 1.96 * std_bel,
        mean_bel + 1.96 * std_bel,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$hparam$")
    plt.ylabel("Perf. Improvement")
    plt.savefig(f"gpr{step:03}.png")


clize.run(experiment)
