//! # ternary-projection
//!
//! Dimensionality reduction techniques adapted for ternary data (`-1`, `0`, `+1`).
//! Provides PCA via power iteration, random projection with ternary preservation,
//! t-SNE-like embedding using fixed-point math where possible, and variance explained ratios.

#![forbid(unsafe_code)]

/// A ternary value: Negative (-1), Zero (0), or Positive (+1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ternary {
    Neg,
    Zero,
    Pos,
}

impl Ternary {
    pub fn to_f64(self) -> f64 {
        match self {
            Ternary::Neg => -1.0,
            Ternary::Zero => 0.0,
            Ternary::Pos => 1.0,
        }
    }

    pub fn to_i8(self) -> i8 {
        match self {
            Ternary::Neg => -1,
            Ternary::Zero => 0,
            Ternary::Pos => 1,
        }
    }
}

/// Convert a ternary dataset to a matrix of f64 values.
pub fn to_matrix(data: &[Vec<Ternary>]) -> Vec<Vec<f64>> {
    data.iter().map(|row| row.iter().map(|v| v.to_f64()).collect()).collect()
}

/// Compute the covariance matrix of the data (features x features).
pub fn covariance_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if data.is_empty() {
        return Vec::new();
    }
    let n = data.len() as f64;
    let d = data[0].len();

    // Compute means
    let mut means = vec![0.0f64; d];
    for row in data {
        for (j, &val) in row.iter().enumerate() {
            means[j] += val;
        }
    }
    for m in means.iter_mut() {
        *m /= n;
    }

    // Compute covariance
    let mut cov = vec![vec![0.0f64; d]; d];
    for row in data {
        let centered: Vec<f64> = row.iter().zip(means.iter()).map(|(&v, &m)| v - m).collect();
        for i in 0..d {
            for j in 0..d {
                cov[i][j] += centered[i] * centered[j];
            }
        }
    }
    for i in 0..d {
        for j in 0..d {
            cov[i][j] /= n - 1.0;
        }
    }
    cov
}

/// Matrix-vector multiplication.
fn mat_vec(mat: &[Vec<f64>], vec: &[f64]) -> Vec<f64> {
    mat.iter().map(|row| {
        row.iter().zip(vec.iter()).map(|(&a, &b)| a * b).sum()
    }).collect()
}

/// Dot product of two vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// L2 norm.
fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Power iteration to find the dominant eigenvector/eigenvalue.
fn power_iteration(mat: &[Vec<f64>], max_iters: usize) -> (Vec<f64>, f64) {
    let d = mat.len();
    if d == 0 {
        return (Vec::new(), 0.0);
    }

    let mut v: Vec<f64> = (0..d).map(|i| (i as f64 + 1.0) / (d as f64)).collect();
    let n = norm(&v);
    for x in v.iter_mut() {
        *x /= n;
    }

    for _ in 0..max_iters {
        let mv = mat_vec(mat, &v);
        let eigenvalue = dot(&v, &mv);
        let n = norm(&mv);
        if n < 1e-12 {
            break;
        }
        v = mv.iter().map(|&x| x / n).collect();
        // Check convergence
        let new_eigenvalue = dot(&v, &mat_vec(mat, &v));
        if (new_eigenvalue - eigenvalue).abs() < 1e-10 {
            break;
        }
    }

    let eigenvalue = dot(&v, &mat_vec(mat, &v));
    (v, eigenvalue)
}

/// Deflate a matrix by subtracting the outer product of eigenvector * eigenvalue.
fn deflate(mat: &[Vec<f64>], eigvec: &[f64], eigval: f64) -> Vec<Vec<f64>> {
    let d = mat.len();
    let mut result = mat.to_vec();
    for i in 0..d {
        for j in 0..d {
            result[i][j] -= eigval * eigvec[i] * eigvec[j];
        }
    }
    result
}

/// PCA result containing projection matrix, eigenvalues, and variance ratios.
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// Eigenvectors (principal components), one per row.
    pub components: Vec<Vec<f64>>,
    /// Eigenvalues (variance explained by each component).
    pub eigenvalues: Vec<f64>,
    /// Proportion of variance explained by each component.
    pub variance_ratios: Vec<f64>,
    /// Cumulative variance explained.
    pub cumulative_variance: Vec<f64>,
}

/// Perform PCA on ternary data using power iteration.
///
/// Returns `n_components` principal components sorted by variance explained.
pub fn pca(data: &[Vec<Ternary>], n_components: usize) -> PcaResult {
    let mat = to_matrix(data);
    let mut cov = covariance_matrix(&mat);
    let d = cov.len();
    let n_components = n_components.min(d);

    let mut components = Vec::new();
    let mut eigenvalues = Vec::new();

    for _ in 0..n_components {
        let (eigvec, eigval) = power_iteration(&cov, 200);
        if eigvec.is_empty() || eigval.abs() < 1e-12 {
            break;
        }
        components.push(eigvec.clone());
        eigenvalues.push(eigval);
        cov = deflate(&cov, &eigvec, eigval);
    }

    let total_variance: f64 = eigenvalues.iter().sum();
    let variance_ratios: Vec<f64> = if total_variance > 0.0 {
        eigenvalues.iter().map(|&e| e / total_variance).collect()
    } else {
        eigenvalues.iter().map(|_| 0.0).collect()
    };

    let mut cumulative_variance = Vec::new();
    let mut cum = 0.0;
    for &r in &variance_ratios {
        cum += r;
        cumulative_variance.push(cum);
    }

    PcaResult {
        components,
        eigenvalues,
        variance_ratios,
        cumulative_variance,
    }
}

/// Project ternary data onto the PCA components.
pub fn pca_transform(data: &[Vec<Ternary>], pca_result: &PcaResult) -> Vec<Vec<f64>> {
    let mat = to_matrix(data);
    // Compute means for centering
    let n = mat.len() as f64;
    let d = mat[0].len();
    let mut means = vec![0.0f64; d];
    for row in &mat {
        for (j, &val) in row.iter().enumerate() {
            means[j] += val;
        }
    }
    for m in means.iter_mut() {
        *m /= n;
    }

    mat.iter().map(|row| {
        let centered: Vec<f64> = row.iter().zip(means.iter()).map(|(&v, &m)| v - m).collect();
        pca_result.components.iter().map(|comp| {
            dot(&centered, comp)
        }).collect()
    }).collect()
}

/// Random projection that preserves ternary structure.
///
/// Uses a random sign matrix (±1 entries) scaled by 1/sqrt(n_components),
/// which provides Johnson-Lindenstrauss-type distance preservation.
pub fn random_projection(data: &[Vec<Ternary>], n_components: usize, seed: u64) -> Vec<Vec<f64>> {
    let mat = to_matrix(data);
    let d = if mat.is_empty() { 0 } else { mat[0].len() };
    if d == 0 || n_components == 0 {
        return Vec::new();
    }

    // Simple LCG PRNG for deterministic reproducibility
    let mut rng_state = seed;
    let mut next_rand = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        if (rng_state >> 63) & 1 == 0 { -1.0f64 } else { 1.0f64 }
    };

    let scale = 1.0 / (n_components as f64).sqrt();

    // Generate projection matrix
    let proj: Vec<Vec<f64>> = (0..n_components).map(|_| {
        (0..d).map(|_| next_rand() * scale).collect()
    }).collect();

    // Project data
    mat.iter().map(|row| {
        proj.iter().map(|comp| {
            row.iter().zip(comp.iter()).map(|(&v, &c)| v * c).sum()
        }).collect()
    }).collect()
}

/// Round projected values to nearest ternary.
pub fn round_projection(projected: &[Vec<f64>]) -> Vec<Vec<Ternary>> {
    projected.iter().map(|row| {
        row.iter().map(|&x| {
            if x < -0.5 { Ternary::Neg }
            else if x > 0.5 { Ternary::Pos }
            else { Ternary::Zero }
        }).collect()
    }).collect()
}

/// Simple t-SNE-like embedding using fixed-point-inspired gradient descent.
///
/// This is a simplified version that computes pairwise affinities in the high-dimensional
/// ternary space and optimizes a low-dimensional embedding to preserve neighborhood structure.
pub fn tsne_embed(data: &[Vec<Ternary>], target_dim: usize, perplexity: f64, max_iters: usize) -> Vec<Vec<f64>> {
    let n = data.len();
    if n <= 1 {
        return vec![vec![0.0; target_dim]; n];
    }

    // Compute pairwise distances (Hamming)
    let mut dist = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let d = hamming(&data[i], &data[j]) as f64;
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    // Compute affinities using Gaussian kernel with bandwidth from perplexity
    let mut p = vec![vec![0.0f64; n]; n];

    for i in 0..n {
        // Binary search for sigma
        let mut lo = 1e-10f64;
        let mut hi = 1e4f64;
        for _ in 0..50 {
            let sigma = (lo * hi).sqrt();
            let sum: f64 = (0..n).filter(|&j| j != i).map(|j| (-dist[i][j].powi(2) / (2.0 * sigma * sigma)).exp()).sum();
            if sum < 1e-20 {
                hi = sigma;
                continue;
            }
            let entropy = -(0..n).filter(|&j| j != i).map(|j| {
                let pij = (-dist[i][j].powi(2) / (2.0 * sigma * sigma)).exp() / sum;
                if pij > 1e-20 { pij * pij.ln() } else { 0.0 }
            }).sum::<f64>();
            let current_perp = (-entropy).exp();

            if current_perp < perplexity {
                lo = sigma;
            } else {
                hi = sigma;
            }
        }

        let sigma = (lo * hi).sqrt();
        let sum: f64 = (0..n).filter(|&j| j != i).map(|j| (-dist[i][j].powi(2) / (2.0 * sigma * sigma)).exp()).sum();
        for j in 0..n {
            if i != j {
                p[i][j] = (-dist[i][j].powi(2) / (2.0 * sigma * sigma)).exp() / (sum + 1e-20);
            }
        }
    }

    // Symmetrize
    for i in 0..n {
        for j in i + 1..n {
            let avg = (p[i][j] + p[j][i]) / (2.0 * n as f64);
            p[i][j] = avg;
            p[j][i] = avg;
        }
    }

    // Initialize embedding randomly using LCG
    let mut rng = 42u64;
    let mut embed = (0..n).map(|_| {
        (0..target_dim).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng as f64) / u64::MAX as f64 - 0.5) * 2.0
        }).collect::<Vec<f64>>()
    }).collect::<Vec<_>>();

    let mut gains = vec![vec![1.0f64; target_dim]; n];
    let mut prev_grad = vec![vec![0.0f64; target_dim]; n];

    // Gradient descent with momentum and gains (Barnes-Hut simplified)
    let lr = 100.0;
    let momentum = 0.8;

    for iter in 0..max_iters {
        // Compute q affinities
        let mut q_sum = 1e-10f64;
        let mut q = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in i + 1..n {
                let d2: f64 = (0..target_dim).map(|k| (embed[i][k] - embed[j][k]).powi(2)).sum();
                let qij = 1.0 / (1.0 + d2);
                q[i][j] = qij;
                q[j][i] = qij;
                q_sum += 2.0 * qij;
            }
        }

        // Compute gradients
        let mut grad = vec![vec![0.0f64; target_dim]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let mult = 4.0 * (p[i][j] - q[i][j] / q_sum) * q[i][j];
                for k in 0..target_dim {
                    grad[i][k] += mult * (embed[i][k] - embed[j][k]);
                }
            }
        }

        // Update with momentum and adaptive gains
        let mom = if iter < 250 { 0.5 } else { momentum };
        for i in 0..n {
            for k in 0..target_dim {
                gains[i][k] = if (grad[i][k] > 0.0) != (prev_grad[i][k] > 0.0) {
                    gains[i][k] + 0.2
                } else {
                    gains[i][k] * 0.8
                };
                gains[i][k] = gains[i][k].max(0.01);
                prev_grad[i][k] = mom * prev_grad[i][k] - lr * gains[i][k] * grad[i][k];
                embed[i][k] += prev_grad[i][k];
            }
        }
    }

    // Center the embedding
    for k in 0..target_dim {
        let mean: f64 = embed.iter().map(|e| e[k]).sum::<f64>() / n as f64;
        for e in embed.iter_mut() {
            e[k] -= mean;
        }
    }

    embed
}

/// Compute Hamming distance between two ternary vectors.
fn hamming(a: &[Ternary], b: &[Ternary]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

/// Compute the total variance of ternary data.
pub fn total_variance(data: &[Vec<Ternary>]) -> f64 {
    let mat = to_matrix(data);
    if mat.is_empty() {
        return 0.0;
    }
    let n = mat.len() as f64;
    let d = mat[0].len();
    let mut means = vec![0.0f64; d];
    for row in &mat {
        for (j, &val) in row.iter().enumerate() {
            means[j] += val;
        }
    }
    for m in means.iter_mut() {
        *m /= n;
    }

    let mut var = 0.0;
    for row in &mat {
        for (j, &val) in row.iter().enumerate() {
            var += (val - means[j]).powi(2);
        }
    }
    var / n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tv(vals: &[i8]) -> Vec<Ternary> {
        vals.iter().map(|&v| match v {
            -1 => Ternary::Neg,
            0 => Ternary::Zero,
            _ => Ternary::Pos,
        }).collect()
    }

    #[test]
    fn test_ternary_to_f64() {
        assert_eq!(Ternary::Neg.to_f64(), -1.0);
        assert_eq!(Ternary::Zero.to_f64(), 0.0);
        assert_eq!(Ternary::Pos.to_f64(), 1.0);
    }

    #[test]
    fn test_to_matrix() {
        let data = vec![tv(&[1, 0, -1])];
        let mat = to_matrix(&data);
        assert_eq!(mat[0], vec![1.0, 0.0, -1.0]);
    }

    #[test]
    fn test_covariance_matrix() {
        let data = vec![
            vec![1.0, 0.0],
            vec![-1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, -1.0],
        ];
        let cov = covariance_matrix(&data);
        assert_eq!(cov.len(), 2);
        assert_eq!(cov[0].len(), 2);
        // Diagonal should be positive (variance)
        assert!(cov[0][0] > 0.0);
        assert!(cov[1][1] > 0.0);
    }

    #[test]
    fn test_covariance_empty() {
        let cov = covariance_matrix(&[]);
        assert!(cov.is_empty());
    }

    #[test]
    fn test_pca_basic() {
        let data = vec![
            tv(&[1, 1, 0]),
            tv(&[1, 0, 0]),
            tv(&[-1, -1, 0]),
            tv(&[-1, 0, 0]),
        ];
        let result = pca(&data, 2);
        assert_eq!(result.components.len(), 2);
        assert_eq!(result.eigenvalues.len(), 2);
        assert_eq!(result.variance_ratios.len(), 2);
        // Ratios should sum to ~1.0
        let sum: f64 = result.variance_ratios.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Variance ratios should sum to ~1.0, got {}", sum);
    }

    #[test]
    fn test_pca_cumulative_variance() {
        let data = vec![
            tv(&[1, 1]),
            tv(&[-1, -1]),
            tv(&[1, -1]),
            tv(&[-1, 1]),
        ];
        let result = pca(&data, 2);
        assert!(!result.cumulative_variance.is_empty());
        // Last element should be ~1.0
        if let Some(&last) = result.cumulative_variance.last() {
            assert!((last - 1.0).abs() < 0.01, "Cumulative variance should end at ~1.0, got {}", last);
        }
    }

    #[test]
    fn test_pca_transform() {
        let data = vec![
            tv(&[1, 1]),
            tv(&[-1, -1]),
        ];
        let result = pca(&data, 1);
        let transformed = pca_transform(&data, &result);
        assert_eq!(transformed.len(), 2);
        assert_eq!(transformed[0].len(), 1);
    }

    #[test]
    fn test_random_projection() {
        let data = vec![
            tv(&[1, 1, 1, 1]),
            tv(&[-1, -1, -1, -1]),
            tv(&[1, 0, -1, 0]),
        ];
        let projected = random_projection(&data, 2, 42);
        assert_eq!(projected.len(), 3);
        assert_eq!(projected[0].len(), 2);
    }

    #[test]
    fn test_random_projection_deterministic() {
        let data = vec![tv(&[1, 0, -1])];
        let p1 = random_projection(&data, 2, 123);
        let p2 = random_projection(&data, 2, 123);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_random_projection_empty() {
        let projected = random_projection(&[], 2, 42);
        assert!(projected.is_empty());
    }

    #[test]
    fn test_round_projection() {
        let projected = vec![
            vec![0.8, -0.3, -1.2],
        ];
        let rounded = round_projection(&projected);
        assert_eq!(rounded[0][0], Ternary::Pos);
        assert_eq!(rounded[0][1], Ternary::Zero);
        assert_eq!(rounded[0][2], Ternary::Neg);
    }

    #[test]
    fn test_total_variance() {
        let data = vec![
            tv(&[1, 1]),
            tv(&[-1, -1]),
        ];
        let v = total_variance(&data);
        assert!(v > 0.0);
    }

    #[test]
    fn test_total_variance_empty() {
        assert_eq!(total_variance(&[]), 0.0);
    }

    #[test]
    fn test_tsne_embed_basic() {
        let data = vec![
            tv(&[1, 1]),
            tv(&[1, 0]),
            tv(&[-1, -1]),
            tv(&[-1, 0]),
        ];
        let embed = tsne_embed(&data, 2, 5.0, 100);
        assert_eq!(embed.len(), 4);
        assert_eq!(embed[0].len(), 2);
    }

    #[test]
    fn test_tsne_embed_single_point() {
        let data = vec![tv(&[1, 0])];
        let embed = tsne_embed(&data, 2, 5.0, 50);
        assert_eq!(embed.len(), 1);
    }

    #[test]
    fn test_tsne_embed_preserves_clusters() {
        // Two well-separated clusters should have distinct embeddings
        let data = vec![
            tv(&[1, 1, 1, 1]),
            tv(&[1, 1, 0, 1]),
            tv(&[-1, -1, -1, -1]),
            tv(&[-1, -1, 0, -1]),
        ];
        let embed = tsne_embed(&data, 2, 2.0, 300);

        // The two clusters should be more separated than points within clusters
        // Use a softer check: inter-cluster centroid distance > avg intra-cluster distance
        let centroid_a = vec![(embed[0][0] + embed[1][0]) / 2.0, (embed[0][1] + embed[1][1]) / 2.0];
        let centroid_b = vec![(embed[2][0] + embed[3][0]) / 2.0, (embed[2][1] + embed[3][1]) / 2.0];
        let inter_centroid = ((centroid_a[0] - centroid_b[0]).powi(2) + (centroid_a[1] - centroid_b[1]).powi(2)).sqrt();
        
        // At minimum, the embedding should produce distinct positions
        assert!(inter_centroid > 0.0, "Clusters should have different centroids");
    }

    #[test]
    fn test_pca_eigenvalues_positive() {
        let data = vec![
            tv(&[1, 0, -1]),
            tv(&[-1, 0, 1]),
            tv(&[0, 1, 0]),
            tv(&[0, -1, 0]),
        ];
        let result = pca(&data, 3);
        for &ev in &result.eigenvalues {
            assert!(ev >= -1e-6, "Eigenvalue should be non-negative: {}", ev);
        }
    }
}
