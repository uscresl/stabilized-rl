# stabilized-rl

## Setup Instructions
Install python 3.9
Install Mujoco 2.1

Enter a python 3.9 venv:
```
python3.9 -m venv venv
source venv/bin/activate
```

Install dependencies:
```
poetry install
pip install -e ./garage
pip install -e ./metaworld
pip install -e ./sample-factory
```

Add any experiments you'd like to run to `exps.py`

Run all unfinished experiments:
```
python src/runexp.py
```

## How to convert PPO to xPPO:

  - Remove PPO clip (just use likelyhood ratio times advantage)

$$ L_{pg} = -\mathbb{E}_t\left[\frac {\pi_{new}(a_t|s_t)} {\pi_{old}(a_t|s_t)} \hat{A_t}\right] $$
  - Use KL penalty with loss coefficient "beta" / $\beta$
    - Also, have a term $x = 1$
    - i.e.

$$ L_{pi} = xL_{pg} + \beta \mathbb{E}_t\left[KL(\pi_{new}(a_t|s_t), \pi_{old}(a_t|s_t))\right] $$
  - Make beta a parameter and add a optimizer and loss on it
    - If the optimizer makes beta become less than 0.01, reset it to 0.01
    - Loss on beta is

$$ L_{\beta} = \beta * \left(KL_{target} - \mathbb{SG}[max_t(KL(\pi_{new}(a_t|s_t), \pi_{old}(a_t|s_t)))]\right) $$

  - Reset value of beta and beta optimizer at start of "step" (group of epochs)
    - beta should be reset to one
    - Use the Adam optimizer, with lr=0.1, momentum=0.999
  - Reset policy optimizer at start of step
    - Use Adam optimizer, same hparams as PPO
  - Add a loop at the end of each epoch that keeps running the loss, but without the policy gradient loss (still step the policy gradient optimizer)
    - Should use full batch gradient descent (might need to use gradient accumulation)
    - I.e. set $x = 0$ in

$$ L_{pi} = xL_{pg} + \beta \mathbb{E}_t\left[KL(\pi_{new}(a_t|s_t), \pi_{old}(a_t|s_t))\right] $$
  - That loss should terminate as soon as the KL div statistic is less than the KL target (for the whole batch)
    - i.e. when

$$ max_t(KL(\pi_{new}(a_t|s_t), \pi_{old}(a_t|s_t))) < KL_{target} $$
 - (Optional?) use a larger offline buffer for the second loop
