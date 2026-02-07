Ground state simulation of frustrated anti-ferromagnetic Heisenberg model on kagome lattice $H = \sum_{\{i, j \}} \vec{s}_i \cdot \vec{s}_j$.

<img width="584" height="530" alt="image" src="https://github.com/user-attachments/assets/bc38236c-7f84-4904-a9c0-f8dce1e0761b" />


Key challenge: the nontrivial sign structure of wavefunction arising from singlet pairings.
Solution: use the exchange move from a reference point to the current configuration as the input for the neural network.
To run the code
```
python GS.py --L $$L --n_features $$n_features
```
where $$L is the linear size of lattice by the number of unit cells, and $$n_features is the number of features for the neural network.
