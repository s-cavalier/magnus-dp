# Software Package for Efficient Numeric Magnus Integration
Currenlty WIP, so bugs may exist. However, the main branch should otherwise be stable to use.

### Requirements
The only requirement is NumPy for building the nodes in `gen_nodes.py` and usage by passing in NumPy arrays. The test/ scripts use scipy, but it is not necessary for the package. 

### How To Use
Run `chmod +x build.sh`, and then run `./build.sh`. It should build everything and run a smoke-test that it built properly.
Once built, the folder `magnus` in test/ should operate as a self-contained python package.