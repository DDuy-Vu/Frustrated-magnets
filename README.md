Ground state simulation of frustrated anti-ferromagnetic Heisenberg model on kagome lattice.
Key challenge: the nontrivial sign structure of wavefunction arising from singlet pairings.
Solution: use the exchange move from a reference point to the current configuration as the input for the neural network.
To run the code
```
python GS.py --L $$L --n_features $$n_features
```
where $$L is the linear size of lattice by the number of unit cells, and $$n_features is the number of features for the neural network.
