# ternary-projection

Dimensionality reduction techniques adapted for ternary data (`-1`, `0`, `+1`).

## Why This Exists

Ternary data has a natural three-valued structure that gets destroyed when you naively apply standard dimensionality reduction. PCA assumes continuous variance; random projections assume real-valued signals. This crate provides PCA via power iteration, random projection with ternary structure preservation, and t-SNE-like embedding — all designed to respect that your data lives in {-1, 0, +1}^d. The algorithms round projections back to ternary values where appropriate, and use Hamming distance as the natural metric for t-SNE's affinity computation.

## Core Concepts

- **`Ternary`** — Three-valued enum: `Neg` (-1), `Zero` (0), `Pos` (+1).
- **`PcaResult`** — Contains principal components (eigenvectors), eigenvalues, variance ratios, and cumulative variance.
- **Ternary PCA** — Principal Component Analysis using power iteration on the covariance matrix of ternary data converted to reals, with deflation to extract multiple components.
- **Ternary random projection** — Johnson-Lindenstrauss-style projection using a random ±1 sign matrix, preserving ternary structure.
- **Ternary t-SNE** — Neighborhood-preserving embedding using Hamming distances for high-dimensional affinities, with Gaussian kernel bandwidth tuned via perplexity.

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-projection = "0.1"
```

```rust
use ternary_projection::*;

fn main() {
    let data = vec![
        vec![Ternary::Pos, Ternary::Pos, Ternary::Zero],
        vec![Ternary::Pos, Ternary::Zero, Ternary::Zero],
        vec![Ternary::Neg, Ternary::Neg, Ternary::Zero],
        vec![Ternary::Neg, Ternary::Zero, Ternary::Zero],
    ];

    // PCA: extract 2 principal components
    let result = pca(&data, 2);
    println!("Variance ratios: {:?}", result.variance_ratios);
    println!("Cumulative variance: {:?}", result.cumulative_variance);

    // Project data onto components
    let projected = pca_transform(&data, &result);
    println!("Projected: {:?}", projected);

    // Random projection to 2 dimensions
    let rp = random_projection(&data, 2, 42);
    println!("Random projection: {:?}", rp);

    // t-SNE embedding to 2D
    let embed = tsne_embed(&data, 2, 5.0, 300);
    println!("t-SNE embedding: {:?}", embed);
}
```

## API Overview

### PCA
- `pca(data, n_components)` — Compute `n_components` principal components. Returns `PcaResult`.
- `pca_transform(data, pca_result)` — Project data onto the computed components.
- `PcaResult` — Fields: `components`, `eigenvalues`, `variance_ratios`, `cumulative_variance`.

### Random Projection
- `random_projection(data, n_components, seed)` — Deterministic random projection using a seeded LCG PRNG. Returns real-valued projections.
- `round_projection(projected)` — Round projected values back to ternary.

### t-SNE
- `tsne_embed(data, target_dim, perplexity, max_iters)` — Compute a low-dimensional embedding preserving neighborhood structure. Uses Hamming distance for high-dimensional affinities.

### Utilities
- `to_matrix(data)` — Convert ternary data to f64 matrix
- `covariance_matrix(data)` — Compute feature × feature covariance matrix
- `total_variance(data)` — Total variance of a ternary dataset

## How It Works

**PCA** converts ternary data to real values, computes the covariance matrix, then uses power iteration to extract the dominant eigenvector/eigenvalue pair. The matrix is deflated (subtracting the outer product) before extracting the next component. This avoids full eigendecomposition and works well for extracting the top few components.

**Random projection** generates a random matrix with ±1 entries scaled by 1/√k (where k is the target dimension), following the Johnson-Lindenstrauss lemma. This preserves approximate pairwise distances while drastically reducing dimensionality. The seed makes projections reproducible.

**t-SNE** computes pairwise Hamming distances in the high-dimensional ternary space, converts them to probability affinities using Gaussian kernels with bandwidth calibrated to the target perplexity, then optimizes a low-dimensional embedding using gradient descent with momentum and adaptive gains. The low-dimensional affinities use the Student t-distribution kernel (1/(1+d²)).

## Use Cases

1. **Visualization of ternary survey data** — Reduce 50-feature sentiment surveys (agree/neutral/disagree) to 2D for visual cluster inspection.
2. **Preprocessing for classification** — Use PCA to reduce a 200-dimension ternary feature space to 20 components before feeding into a downstream classifier.
3. **Noise-resistant similarity search** — Random project ternary feature vectors for fast approximate nearest-neighbor lookup with distance guarantees.

## Ecosystem

- [`ternary-clustering`](https://github.com/user/ternary-clustering) — Clustering algorithms for ternary data
- [`ternary-streaming`](https://github.com/user/ternary-streaming) — Streaming processing for ternary signals
- [`ternary-signals`](https://github.com/user/ternary-signals) — Fourier analysis and signal processing for ternary data

## License

MIT
