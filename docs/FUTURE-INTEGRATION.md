# Future Integration: ternary-projection

## Current State
Implements dimensionality reduction for ternary data: PCA via power iteration with covariance matrix computation, random projection with ternary preservation (Johnson-Lindenstrauss), t-SNE-like embedding using fixed-point math, and variance-explained ratios.

## Integration Opportunities

### With ternary-cell / room-as-codespace
Project high-dimensional room state to 2D/3D for visualization and similarity search. A room with 48 ternary parameters projects to a 3-trit summary via `pca_project()`. Two rooms with similar projections have similar overall states. The t-SNE embedding (`tsne_embedding()`) produces 2D coordinates for a fleet dashboard — rooms cluster visually by function.

### With ternary-pca
Both crates implement PCA. `ternary-pca` uses fixed-point arithmetic (i32); `ternary-projection` uses f64. Unify under a shared trait: `TernaryProjection` with methods `fit()`, `transform()`, `explained_variance()`. The fixed-point variant for ESP32, the float variant for analysis. Random projection (`random_projection()`) provides a cheaper alternative when PCA's eigendecomposition is too expensive.

### With ternary-matrix
`TernaryMatrix` should provide the backing storage and linear algebra. Currently `ternary-projection` implements its own matrix-vector multiply and covariance computation. Delegate to `TernaryMatrix::multiply()` for consistency and potential SIMD acceleration.

## Potential in Mature Systems
In PLATO, projection is the bridge between Layer 0 and Layer 1. The ESP32 runs `random_projection()` (fixed matrix, just matrix-vector multiply — no training needed) to compress its 48-trit state to 8 trits for transmission. Layer 1 receives the compressed states from all rooms and runs `pca_project()` for fleet-level analysis. The `explained_variance_ratio()` tells the operator how much information is preserved at each compression level.

## Cross-Pollination Ideas
**Music × Projection:** Project high-dimensional ternary musical features (rhythm, harmony, timbre, dynamics) to a 2D "genre space." Similar pieces cluster in this space. `tsne_embedding()` creates a visual map of a music library. New compositions project to a point — their nearest neighbors in the embedding are their stylistic relatives. Connects to `ternary-music`.

**Topology × Projection:** Random projection preserves pairwise distances (Johnson-Lindenstrauss guarantee). But for ternary data, it also preserves some topological structure. The Betti numbers of the original and projected data should be similar for sufficiently high target dimension. `ternary-topology` can validate this empirically.

## Dependencies for Next Steps
- Shared `TernaryProjection` trait with `ternary-pca`
- `TernaryMatrix` integration for backing linear algebra
- Incremental PCA that updates with streaming data
