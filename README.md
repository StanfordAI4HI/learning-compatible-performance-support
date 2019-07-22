# Paper

This code is for the following paper:
```
Jonathan Bragg and Emma Brunskill. 2019. Fake It Till You Make It: Learning-Compatible Performance Support. In UAI '19.
```

# Installation

```bash
conda create -n perflearn python=3.5
source activate perflearn
pip install tensorflow
conda env update -n perflearn --file environment.yml
```

If on gpu, replace `tensorflow` above with `tensorflow-gpu`.

If on mac, you may need to run the following first:
```bash
$ brew install mpich
$ brew install swig
```

Run `./link.sh` to sym link the `gym` and `baselines` modifications.

For useful ipynb tools, run `pip install jupyter nbdime`.

For code linting and autofixing, run `pip install flake8 autopep8`

# Usage

The main experiment file can be run as `python -m perflearn.test [OPTIONS]`.

UAI paper experiment options are:
- `--env_name [LunarLanderFixed-v2 | CliffWalking-treasure100-v0]`
- `--learner_policy q`

The stochastic q bumpers policy uses the options:
- `--learner_support bumpers`
- `--q_threshold [ALPHA]`
- `--q_bumper_boltzmann 1`
- `--q_bumper_logistic_upper_prob 0.999`
- `--q_bumper_alpha [ALPHA]`
- `--q_bumper_target_r [LOWER REWARD VALUE]`
- `--gamma 0.99`

The local q-thresholding policy uses the options:
- `--learner_support reddy_rss`
- `--q_threshold [ALPHA]`


