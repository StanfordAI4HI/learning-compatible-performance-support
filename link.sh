rm src/baselines/baselines/deepq/build_graph.py
ln -s ../../../../build_graph.py src/baselines/baselines/deepq/
rm src/gym/gym/envs/box2d/lunar_lander.py
ln -s ../../../../../lunar_lander.py src/gym/gym/envs/box2d/
rm src/gym/gym/envs/box2d/__init__.py
ln -s ../../../../../gym_box2d_init.py src/gym/gym/envs/box2d/__init__.py
rm src/gym/gym/envs/toy_text/cliffwalking.py
ln -s ../../../../../cliffwalking.py src/gym/gym/envs/toy_text/
rm src/gym/gym/envs/__init__.py
ln -s ../../../../gym_init.py src/gym/gym/envs/__init__.py
