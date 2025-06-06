

Add `src` to path:
```
pip3 install -e .
```

Benchmark performance of pretrained property predictor

Histograms and categorical distribution visualisations for pretrained models
```
python3 src/edm/visualise/edm_qm9_comparison.py --edm_path models/edm.pt --prop_pred_paths models/mu.pt models/alpha.pt models/U0.pt --samples 5000
```