param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

foreach ($seed in @(42, 123, 3407)) {
    $runDir = "runs/qiyuhan_cifar_seed$seed"
    $pool = "$runDir/candidate_pool.npz"
    $detail = "$runDir/discovery_detail.json"
    $common = @(
        $pool,
        "--cluster-k", "oracle",
        "--num-novel", "40",
        "--seed", "$seed",
        "--evaluation-detail", $detail
    )

    & $Python scripts/analyze_frozen_candidate_pool.py @common `
        --output-dir "$runDir/cluster_feature_pca_kmeans" `
        --feature feature_pca --pca-dim 32 --method kmeans

    & $Python scripts/analyze_frozen_candidate_pool.py @common `
        --output-dir "$runDir/cluster_feature_normalized_kmeans" `
        --feature feature --normalize --method kmeans

    & $Python scripts/analyze_frozen_candidate_pool.py @common `
        --output-dir "$runDir/cluster_feature_kmeans" `
        --feature feature --method kmeans --skip-all-unknown-oracle

    & $Python scripts/analyze_frozen_candidate_pool.py @common `
        --output-dir "$runDir/cluster_projection_pca_kmeans" `
        --feature projection_pca --pca-dim 32 --method kmeans `
        --skip-all-unknown-oracle

    & $Python scripts/analyze_frozen_candidate_pool.py @common `
        --output-dir "$runDir/cluster_feature_pca_agglomerative" `
        --feature feature_pca --pca-dim 32 --method agglomerative `
        --skip-all-unknown-oracle

    & $Python scripts/analyze_frozen_candidate_pool.py @common `
        --output-dir "$runDir/cluster_feature_pca_spectral" `
        --feature feature_pca --pca-dim 32 --method spectral `
        --skip-all-unknown-oracle
}
