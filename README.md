# Acting on the Tangent Space of the Constraint Manifold
Implementation of "Robot Reinforcement Learning on the Constraint Manifold"

[[paper]](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/PuzeLiu/CORL_2021_Learning_on_the_Manifold.pdf)
[[website]](https://sites.google.com/view/robot-air-hockey/atacom)

## Install
```python
pip install -e .
```

## Run Examples
```python
cd examples
```
### CircularMotion Environment. 
Environment options [A, E, T], algorithms options [TRPO, PPO, SAC, DDPG, TD3]
```python
python circle_exp.py --render --env A --alg TRPO
```

### PlanarAirHockey Environment. 
Environment options [H, D, UH, UD], algorithms options [TRPO, PPO, SAC, DDPG, TD3]
```python
python planar_air_hockey_exp.py --debug-gui --env H --alg SAC
```

### IiwaAirHockey Environment. 
Environment options [7H, RMP], algorithms options [TRPO, PPO, SAC, DDPG, TD3]
```python
python iiwa_air_hockey_exp.py --debug-gui --env 7H --alg SAC
```

### CollisionAvoidance Environment. 
Environment options [C], algorithms options [TRPO, PPO, SAC, DDPG, TD3]
```python
python collision_avoidance_exp.py --render --env C --alg SAC
```


## Bibtex
```bibtex
@inproceedings{CORL_2021_Learning_on_the_Manifold,
  author =      "Liu, P. and  Tateo D. and  Bou-Ammar, H. and  Peters, J.",
  year =        "2021",
  title =       "Robot Reinforcement Learning on the Constraint Manifold",
  booktitle =   "Proceedings of the Conference on Robot Learning (CoRL)",
  key =	        "robot learning, constrained reinforcement learning, safe exploration",
}
```
