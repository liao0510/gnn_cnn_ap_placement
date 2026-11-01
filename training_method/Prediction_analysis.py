"""Final-deployment metrics and paired, trial-level bootstrap summaries."""
import numpy as np


# key, plot label, percentage axis. All metrics are higher-is-better.
ANALYSIS_METRICS = (
    ('coverage', 'Coverage', True),
    ('mean_satisfaction', 'Mean throughput satisfaction', True),
    ('mesh_connectivity', 'APs in largest mesh / all APs', True),
    ('demand_met_ratio', 'Clients meeting demand', True),
    ('total_throughput_mbps', 'Total delivered throughput (Mbps)', False),
    ('client_throughput_p05_mbps', '5th-percentile client throughput (Mbps)', False),
    ('satisfaction_fairness', 'Jain fairness of demand achievement', True),
    ('associated_client_ratio', 'Client association rate', True),
    ('mesh_connected', 'Fully connected mesh', True),
)
METHOD_LABELS = {'gnn_cnn': 'GNN + CNN', 'random': 'Random AP'}


def final_metrics(coverage, network):
    """Include unassociated clients (zero throughput); never filter failures.

    Jain fairness uses demand-normalized rates for positive-demand clients.
    When all these rates are zero it is defined as zero, rather than 0/0.
    No-client metrics are undefined (None), not artificial perfect scores.
    """
    rows = network['allocation_results']
    throughput = np.asarray([row['throughput_mbps'] for row in rows], dtype=float)
    satisfaction = np.asarray([row['satisfaction_ratio'] for row in rows
                               if row['demand_mbps'] > 0], dtype=float)
    denominator = float(np.sum(satisfaction ** 2))
    fairness = (float(satisfaction.sum() ** 2 / (len(satisfaction) * denominator))
                if denominator > 0 else 0.0) if len(satisfaction) else None
    summary = network['network_summary']
    return {
        'coverage': float(coverage),
        'mean_satisfaction': float(summary['mean_satisfaction']) if rows else None,
        'mesh_connectivity': float(network['mesh_connectivity']),
        'demand_met_ratio': float(summary['demand_met_ratio']) if rows else None,
        'total_throughput_mbps': float(summary['total_throughput_mbps']),
        'client_throughput_p05_mbps': float(np.percentile(throughput, 5)) if rows else None,
        'satisfaction_fairness': fairness,
        'associated_client_ratio': float(summary['associated_clients'] / len(rows)) if rows else None,
        'mesh_connected': float(network['mesh_connected']),
    }


def describe(values, rng, bootstrap_samples):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(values):
        return {'n': 0, 'mean': None, 'std': None, 'median': None,
                'p05': None, 'p95': None, 'mean_ci95': None}
    if not np.all(np.isfinite(values)):
        raise ValueError('Analysis values must be finite or None')
    interval = None
    if len(values) >= 2:
        # Chunk draws so a larger trial count does not create a huge tensor.
        means = []
        for start in range(0, bootstrap_samples, 128):
            indices = rng.integers(len(values), size=(min(128, bootstrap_samples - start), len(values)))
            means.extend(values[indices].mean(axis=1))
        interval = np.percentile(means, [2.5, 97.5]).tolist()
    return {'n': len(values), 'mean': float(values.mean()),
            'std': float(values.std(ddof=1)) if len(values) >= 2 else None,
            'median': float(np.median(values)), 'p05': float(np.percentile(values, 5)),
            'p95': float(np.percentile(values, 95)), 'mean_ci95': interval}


def summarize_trials(trials, bootstrap_samples=2000, seed=2026):
    if not trials:
        raise ValueError('At least one paired trial is required')
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 100:
        raise ValueError('Use at least 100 bootstrap resamples')
    rng = np.random.default_rng(seed)
    summary = {'methods': {}, 'paired_difference_gnn_minus_random': {},
               'bootstrap_samples': bootstrap_samples,
               'resampling_unit': 'paired trial, not individual client',
               'ci_method': '95% percentile bootstrap confidence interval of the mean',
               'scope': 'conditional on this trained model and the sampled evaluation scenarios'}
    for method in METHOD_LABELS:
        summary['methods'][method] = {
            key: describe([trial[method]['metrics'][key] for trial in trials], rng, bootstrap_samples)
            for key, _, _ in ANALYSIS_METRICS}
    for key, _, _ in ANALYSIS_METRICS:
        differences = [trial['gnn_cnn']['metrics'][key] - trial['random']['metrics'][key]
                       for trial in trials if trial['gnn_cnn']['metrics'][key] is not None
                       and trial['random']['metrics'][key] is not None]
        stats = describe(differences, rng, bootstrap_samples)
        stats.update(win_rate=float(np.mean(np.asarray(differences) > 1e-9)) if differences else None,
                     tie_rate=float(np.mean(np.abs(differences) <= 1e-9)) if differences else None)
        summary['paired_difference_gnn_minus_random'][key] = stats
    return summary
