# Bachelor's Thesis: Equivariant Neural Diffusion for Molecule Generation
Welcome to Equivariant Diffusion Models for Molecule Generation (we didn't get to the "neural" part). This is a from-scratch implementation of a generative model for molecule generation that combines a simplified EDM framework [Hoogeboom et al.] with a PaiNN [Schütt et al.] implementation. This was written as part of my Bachelor's project - it's been fun!

This started out as a super neat repository. Nice class structure, useful inheritance, good folder structure, and more. And then the deadlines hit.. And all sorts of funny stuff needed to be changed slightly. I'll admit, I cut a lot of corners in the end, which is why this repo isn't as nice as I'd like it to be. Perhaps in a future version? For now I need to finish writing. 

Below you'll find a guide for installation and training some models - feel free to play around!

## Installation
Add `src` to path:
```
pip3 install -e .
```

## Training


## Benchmarking
Benchmark performance of pretrained property predictor


## Visualisations
Histograms and categorical distribution visualisations for pretrained models
```
python3 src/edm/visualise/edm_qm9_comparison.py --edm_path models/edm.pt --prop_pred_paths models/mu.pt models/alpha.pt models/U0.pt --samples 5000
```
