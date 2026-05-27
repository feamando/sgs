# Refinement Comparison: SGS-native vs. External

Generated: refinement_comparison_v3.md

## Metrics

| Metric | SGS-native | External | Winner |
|--------|-----------|----------|--------|
| Gaussian count | 51721.0000 | 56561.0000 | External |
| Avg NN distance | 0.0008 | 0.0006 | External |
| Uniformity (std/mean NN) | 3.2357 | 0.8040 | External |
| Avg opacity | 0.8822 | 0.8774 | SGS |
| Low-opacity fraction | 0.0000 | 0.0000 | Tie |
| Color variance | 0.0296 | 0.0295 | SGS |
| Density (G/vol) | 1589.4052 | 1738.1396 | External |

## Bounding Box

- SGS: [4.05, 1.98, 4.06] (vol=32.54)
- Ext: [4.05, 1.98, 4.06] (vol=32.54)

## Interpretation

- **Avg NN distance**: lower = denser packing, more solid surfaces
- **Uniformity ratio**: lower = more even distribution (fewer gaps/clusters)
- **Low-opacity fraction**: lower = fewer invisible/useless Gaussians
- **Color variance**: higher = more visual diversity (not all same color)
- **Density**: higher = more detail per unit of space
