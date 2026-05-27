# Refinement Comparison: SGS-native vs. External

Generated: refinement_comparison_v3.md

## Metrics

| Metric | SGS-native | External | Winner |
|--------|-----------|----------|--------|
| Gaussian count | 52018.0000 | 56561.0000 | External |
| Avg NN distance (packing) | 0.0007 | 0.0006 | External |
| Surface coverage (3D fill %) | 0.0110 | 0.0091 | SGS |
| Silhouette fill (top-down %) | 0.2258 | 0.2146 | SGS |
| Local compactness | 0.0010 | 0.0011 | SGS |
| Avg opacity | 0.8822 | 0.8774 | SGS |
| Low-opacity fraction | 0.0000 | 0.0000 | Tie |
| Color variance | 0.0297 | 0.0295 | SGS |
| Density (G/vol) | 1583.4376 | 1738.1396 | External |

## Bounding Box

- SGS: [4.14, 1.98, 4.01] (vol=32.85)
- Ext: [4.05, 1.98, 4.06] (vol=32.54)

## Interpretation

- **Avg NN distance**: lower = denser packing, more solid surfaces
- **Surface coverage**: higher = more of the 3D volume is occupied (solid, not hollow)
- **Silhouette fill**: higher = the scene fills more of its footprint from above
- **Local compactness**: lower = local neighborhoods are tighter (solid objects)
- **Low-opacity fraction**: lower = fewer invisible/useless Gaussians
- **Color variance**: higher = more visual diversity (not all same color)
- **Density**: higher = more detail per unit of space
