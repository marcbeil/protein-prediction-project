# Changes
- masked out padded residues for loss

# Problems
- Dataset:
  - domains of the same protein are not necessary in the same split -> leads to inconsistent predictions
    - Solution: dataset v5
- need to adjust dataset class because domains of the same protein need to be fed to the model at the same time otherwise loss calculation is bad
  - example: ![1dcqA02.png](visualization_results/1dcqA02.png)
- Fragmentation of domains
  - Solution: Post-processing
    - Rules: use the same rules as protenn2