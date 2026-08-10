# RelationalRailPricing-RL: Relational Multi-Agent Reinforcement Learning for Dynamic Pricing in High-Speed Railway Markets

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![ArXiv](https://img.shields.io/badge/arXiv-2501.08234-b31b1b.svg)](https://doi.org/10.48550/arXiv.2607.05179)

## 🛠️ Installation

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kinrre/RelationalRailPricing-RL.git
   cd RelationalRailPricing-RL

2. **Install poetry dependency manager:**
    ```bash
    pip3 install poetry

3. **Install project with poetry**:
   ```bash
   poetry install

## 💻 Usage

1. **Training the Agent:**

    To train a new relational agent RL agent from scratch, run the following command:
    ```bash
    python3 -m tests.test_rl_training --supply-config configs/rl/supply_data_large.yaml --demand-config configs/rl/demand_data_large.yaml --seed 0 --exp-name large --algorithm rache --total-timesteps 1000000 --batch-size 1024 --gamma 0.99 --policy-lr 0.001 --q-lr 0.001 --learning-starts 10000 --rgcn-num-layers 2 --normalize-obs-output --reward-scale-factor 1000 --detach-actor-from-preprocessor

2. **Evaluation:**

    ```bash
    python3 -m tests.test_rl_evaluator --seed 0 --input_dir $input_dir --algorithm rache --total-timesteps 100000

## 📜 Citation

If you use this repo in your work, please consider citing the corresponding paper:

```bibtex
@article{villarrubia2026relational,
  title={Relational Multi-Agent Reinforcement Learning for Dynamic Pricing in High-Speed Railway Markets},
  author={Villarrubia-Martin, Enrique Adrian and Mu{\~n}oz-Valero, David and Rodriguez-Benitez, Luis and Montana, Giovanni and Jimenez-Linares, Luis},
  journal={arXiv preprint arXiv:2607.05179},
  year={2026}
}
