"""AP-AP reference SNR mesh, AP-client SINR and OFDMA results."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import font_manager
from matplotlib.patches import Patch


# Material code, Chinese label, fallback label, and opaque wall color.
WALL_MATERIALS = (
    (10, '混凝土', 'Concrete', '#80858C'),
    (6, '木頭', 'Wood', '#654321'),
    (3, '木隔板', 'Wood partition', '#D2B48C'),
    (1, '玻璃', 'Glass', '#BFE7F5'),
)


def plot_prediction_results(walls, forbidden, rssi_map, report, output_dir=None, show=True):
    """Draw every AP/client and qualified link with a detailed right sidebar.

    Client throughput values come from the environment's completed OFDMA
    allocation. Long sidebars are paginated so no client details are omitted.
    """
    if not show:
        plt.switch_backend('Agg')
    net = report['network']
    aps, clients = report['ap_positions'], report['client_positions']
    destination = Path(output_dir) if output_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    figures = []

    def finish(figure, name):
        figure.tight_layout()
        if destination is not None:
            figure.savefig(destination / name, dpi=150, bbox_inches='tight')
        figures.append(figure)

    def ap_label(ap):
        return f'AP{ap}' if ap is not None and ap >= 0 else 'Unassociated'

    def number(value):
        return '-' if value is None else f'{value:.1f}'

    # Map sparse material IDs to categorical colors, independently of RSSI.
    wall_values = np.asarray(walls)
    wall_indices = np.full(wall_values.shape, -1, dtype=np.int8)
    wall_indices[wall_values > 0] = len(WALL_MATERIALS)
    for index, (code, _, _, _) in enumerate(WALL_MATERIALS):
        wall_indices[wall_values == code] = index
    wall_cmap = ListedColormap([material[3] for material in WALL_MATERIALS] + ['#9CA3AF'])
    wall_norm = BoundaryNorm(np.arange(-0.5, len(WALL_MATERIALS) + 1, 1), wall_cmap.N)
    wall_layer = np.ma.masked_where(wall_indices < 0, wall_indices)
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    chinese_font = next((name for name in ('Microsoft JhengHei', 'Microsoft YaHei',
                        'Noto Sans CJK TC', 'SimHei') if name in available_fonts), None)
    material_font = font_manager.FontProperties(family=chinese_font or 'DejaVu Sans', size=9)
    material_handles = [Patch(facecolor=color, edgecolor='#E5E7EB',
                              label=f'{code} = {chinese if chinese_font else english}')
                        for code, chinese, english, color in WALL_MATERIALS]
    if np.any(wall_indices == len(WALL_MATERIALS)):
        material_handles.append(Patch(facecolor='#9CA3AF', edgecolor='#E5E7EB',
                                      label='其他材質' if chinese_font else 'Other material'))

    # One topology with a right-hand detail panel; long panels are paginated.
    import matplotlib.patheffects as effects
    from matplotlib.lines import Line2D
    summary = net['network_summary']
    demand = float(sum(report['client_demands']))
    total_throughput = float(summary['total_throughput_mbps'])
    aggregate_ratio = total_throughput / demand if demand > 0 else 1.0
    header = [
        'COVERAGE / THROUGHPUT',
        f'Coverage:                 {report["coverage"]:8.2%}',
        f'Mean throughput satisfaction: {summary["mean_satisfaction"]:8.2%}',
        f'Clients meeting demand:   {summary["demand_met_ratio"]:8.2%}',
        f'Delivered / total demand: {aggregate_ratio:8.2%}',
        f'Throughput: {total_throughput:.2f} / {demand:.2f} Mbps',
        f'APs in serving mesh: {sum(net["in_mesh"])} / {len(aps)}',
        f'Mesh edges: {len(net["mesh_edges"])} | components: {len(net["mesh_components"])}',
        f'Clients associated: {summary["associated_clients"]} / {len(clients)}',
        f'Coverage threshold: {report["coverage_threshold_dbm"]:g} dBm',
        '',
    ]
    details = ['AP DETAILS']
    for ap, position in enumerate(aps):
        resource = net['ap_resources'][ap]
        peers = [edge['ap_b'] if edge['ap_a'] == ap else edge['ap_a']
                 for edge in net['mesh_edges'] if ap in (edge['ap_a'], edge['ap_b'])]
        membership = 'mesh' if net['in_mesh'][ap] else 'outside mesh'
        details.extend([
            f'AP{ap} {tuple(position)} [{membership}] peers={peers}',
            f'  clients={len(resource["associated_clients"])} | RU={resource["ru_tones"]} | '
            f'{resource["throughput_mbps"]:.2f} Mbps',
        ])
    details.extend(['', 'CLIENT DETAILS (actual throughput / demand, Mbps)'])
    for client, row in enumerate(net['allocation_results']):
        sinr = number(row['sinr_db'])
        details.extend([
            f'C{client} {tuple(clients[client])} -> {ap_label(row["ap"])} | '
            f'SINR {sinr} dB | MCS {row["mcs"] if row["mcs"] is not None else "-"}',
            f'  RU {row["ru_tones"]} | time {row["time_fraction"]:.0%} | '
            f'{row["throughput_mbps"]:.1f}/{row["demand_mbps"]:.1f} Mbps | '
            f'sat {row["satisfaction_ratio"]:.1%}',
            f'  RSSI {number(row.get("rssi_dbm"))} dBm | '
            f'I {number(row.get("interference_dbm"))} dBm | '
            f'N {number(row.get("noise_dbm"))} dBm',
        ])
    details.extend(['', 'AP-AP SNR LINKS (both directions)'])
    for edge in net['mesh_edges']:
        details.append(f'AP{edge["ap_a"]} <-> AP{edge["ap_b"]}: '
                       f'{edge["snr_a_to_b_db"]:.1f} / {edge["snr_b_to_a_db"]:.1f} dB')
    if not net['mesh_edges']:
        details.append('No qualified AP-AP links')

    line_effect = [effects.Stroke(linewidth=3.3, foreground='#111827'), effects.Normal()]
    text_box = dict(boxstyle='round,pad=0.18', facecolor='#101827', edgecolor='none', alpha=0.78)
    for page, offset in enumerate(range(0, max(len(details), 1), 44)):
        try:
            fig, (ax, panel) = plt.subplots(1, 2, figsize=(20, 11),
                                             gridspec_kw={'width_ratios': [1.22, 1]})
        except Exception as error:
            if type(error).__name__ != 'TclError':
                raise
            plt.switch_backend('Agg')
            show = False
            fig, (ax, panel) = plt.subplots(1, 2, figsize=(20, 11),
                                             gridspec_kw={'width_ratios': [1.22, 1]})
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#161e2b')
        heat = ax.imshow(np.ma.masked_invalid(rssi_map), cmap='jet', vmin=-100, vmax=-20)
        ax.imshow(np.ma.masked_where(np.asarray(forbidden) == 0, forbidden),
                  cmap=ListedColormap(['orange']), alpha=0.24)
        # Draw walls last so forbidden-cell shading cannot alter material colors.
        ax.imshow(wall_layer, cmap=wall_cmap, norm=wall_norm,
                  interpolation='nearest', alpha=1.0)
        colorbar = fig.colorbar(heat, ax=ax, orientation='horizontal', pad=0.06, shrink=0.85)
        colorbar.set_label('Serving mesh RSSI (dBm)', color='white')
        colorbar.ax.tick_params(colors='white')
        ax.set_title('AP-AP SNR mesh / AP-client SINR | white lines = links', color='white')
        for edge_index, edge in enumerate(net['mesh_edges']):
            x1, y1 = aps[edge['ap_a']]
            x2, y2 = aps[edge['ap_b']]
            ax.plot([x1, x2], [y1, y2], '-', color='white', linewidth=1.6,
                     path_effects=line_effect, zorder=2)
            fraction = (0.3, 0.45, 0.65, 0.8)[edge_index % 4]
            ax.annotate(f'AP{edge["ap_a"]}-AP{edge["ap_b"]}: '
                         f'{edge["snr_a_to_b_db"]:.1f}/{edge["snr_b_to_a_db"]:.1f} dB',
                         (x1 + fraction * (x2 - x1), y1 + fraction * (y2 - y1)),
                         xytext=(0, 8 if edge_index % 2 == 0 else -9), textcoords='offset points',
                         fontsize=7, color='white', ha='center', va='center', bbox=text_box, zorder=3)
        for ap, (x, y) in enumerate(aps):
            ax.scatter(x, y, marker='^', s=150, c='lime' if net['in_mesh'][ap] else 'red',
                        edgecolors='black', linewidths=1, zorder=6)
            ax.annotate(ap_label(ap), (x, y), xytext=(5, 9), textcoords='offset points',
                         fontsize=9, color='white', weight='bold', bbox=text_box, zorder=7)
        for client, (x, y) in enumerate(clients):
            row = net['allocation_results'][client]
            ap = row['ap']
            if ap >= 0:
                px, py = aps[ap]
                ax.plot([x, px], [y, py], '--', color='white', linewidth=1.15,
                         path_effects=line_effect, zorder=2)
                fraction = 0.35 + 0.12 * (client % 4)
                ax.annotate(f'C{client}: SINR {row["sinr_db"]:.1f} dB\n{row["throughput_mbps"]:.1f} Mbps',
                             (x + fraction * (px - x), y + fraction * (py - y)),
                             xytext=(9 if client % 2 == 0 else -9, 0), textcoords='offset points',
                             fontsize=7, color='white', ha='center', va='center', bbox=text_box, zorder=3)
            ax.scatter(x, y, s=48, marker='o' if ap >= 0 else 'x',
                        c='cyan' if ap >= 0 else '#ff7c7c', zorder=5)
            ax.annotate(f'C{client}', (x, y), xytext=(4, -13), textcoords='offset points',
                         fontsize=8, color='white', bbox=text_box, zorder=7)
        ax.set(xlabel='X (grid)', ylabel='Y (grid)',
               xlim=(-0.5, walls.shape[1] - 0.5), ylim=(walls.shape[0] - 0.5, -0.5))
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        legend = ax.legend(handles=[Line2D([0], [0], color='white', label='AP-AP mesh'),
                                   Line2D([0], [0], color='white', linestyle='--', label='AP-client')],
                            loc='upper right', facecolor='#111827', edgecolor='#8892a3', fontsize=8)
        for label in legend.get_texts():
            label.set_color('white')
        ax.add_artist(legend)
        material_legend = ax.legend(handles=material_handles, loc='lower left', ncol=2,
                                     title='牆體材質' if chinese_font else 'Wall materials',
                                     prop=material_font, facecolor='#111827',
                                     edgecolor='#8892a3', framealpha=0.95)
        material_legend.get_title().set_fontproperties(material_font)
        material_legend.get_title().set_color('white')
        for label in material_legend.get_texts():
            label.set_color('white')
        panel.set_facecolor('#111827')
        panel.axis('off')
        lines = header + details[offset:offset + 44]
        panel.text(0.01, 0.99, '\n'.join(lines), transform=panel.transAxes,
                    ha='left', va='top', family='monospace', fontsize=9,
                    linespacing=1.25, color='#edf2f7')
        if len(details) > 44:
            panel.set_title(f'Details page {page + 1}/{(len(details) + 43) // 44}', color='white')
        finish(fig, 'network_maps.png' if page == 0 else f'network_maps_{page + 1}.png')

    def tables(rows, columns, title, stem):
        for start in range(0, max(len(rows), 1), 20):
            page = rows[start:start + 20]
            fig, ax = plt.subplots(figsize=(max(10, len(columns) * 1.45), max(3, len(page) * 0.36 + 1.5)))
            ax.axis('off')
            ax.set_title(f'{title} (page {start // 20 + 1})', pad=18)
            if page:
                table = ax.table(cellText=page, colLabels=columns, loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
            else:
                ax.text(0.5, 0.5, 'No devices', ha='center', va='center')
            finish(fig, f'{stem}_{start // 20 + 1}.png')

    summary = net['network_summary']
    rows = [['APs / serving mesh', f'{len(aps)} / {sum(net["in_mesh"])}'],
            ['Mesh edges / components', f'{len(net["mesh_edges"])} / {len(net["mesh_components"])}'],
            ['Fully connected mesh', str(net['mesh_connected'])],
            ['Clients / associated', f'{len(clients)} / {summary["associated_clients"]}'],
            ['Total demand (Mbps)', number(sum(report['client_demands']))],
            ['OFDMA throughput (Mbps)', number(summary['total_throughput_mbps'])],
            ['Mean satisfaction', f'{summary["mean_satisfaction"]:.1%}'],
            ['Demand met', f'{summary["demand_met_ratio"]:.1%}'],
            ['Coverage', f'{report["initial_coverage"]:.1%} -> {report["coverage"]:.1%}']]
    tables(rows, ['Metric', 'Value'], 'AP mesh and OFDMA summary', 'summary')
    ap_rows = []
    for ap, position in enumerate(aps):
        resource = net['ap_resources'][ap]
        neighbors = [edge['ap_b'] if edge['ap_a'] == ap else edge['ap_a']
                     for edge in net['mesh_edges'] if ap in (edge['ap_a'], edge['ap_b'])]
        ap_rows.append([ap_label(ap), str(tuple(position)), str(net['in_mesh'][ap]),
                        ','.join(map(str, neighbors)) or '-', len(resource['associated_clients']),
                        resource['ru_tones'], resource['concurrent_clients'], number(resource['throughput_mbps'])])
    tables(ap_rows, ['AP', '(x,y)', 'In mesh', 'Neighbors', 'Clients', 'RU tones',
                     'Concurrent', 'Throughput\n(Mbps)'], 'AP resource allocation', 'ap_details')
    client_rows = []
    for client, row in enumerate(net['allocation_results']):
        client_rows.append([f'C{client}', ap_label(row['ap']), number(row['sinr_db']),
                            row['mcs'], row['ru_tones'], f'{row["time_fraction"]:.1%}',
                            number(row['demand_mbps']), number(row['throughput_mbps']),
                            f'{row["satisfaction_ratio"]:.1%}'])
    tables(client_rows, ['Client', 'AP', 'SINR (dB)', 'MCS', 'RU tones', 'Time share',
                         'Demand\n(Mbps)', 'Throughput\n(Mbps)', 'Satisfaction'],
           'Client SINR and OFDMA allocation', 'client_details')
    mesh_rows = [[edge['ap_a'], edge['ap_b'], number(edge['snr_a_to_b_db']),
                  number(edge['snr_b_to_a_db'])] for edge in net['mesh_edges']]
    tables(mesh_rows, ['AP A', 'AP B', 'A to B SNR (dB)', 'B to A SNR (dB)'],
           'All threshold-qualified mesh edges', 'mesh_links')
    if show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)
    return figures


def plot_validation_history(history, best_episode, save_path, show=False):
    """Final-deployment metrics on the same validation scenarios at every check."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    episodes = [row['episode'] for row in history]
    axes[0].plot(episodes, [row['score'] for row in history], marker='o', label='Validation objective')
    for key, label in (('coverage', 'Coverage'), ('mean_satisfaction', 'Throughput satisfaction'),
                       ('mesh_connectivity', 'Mesh connectivity'), ('demand_met_ratio', 'Clients meeting demand')):
        axes[1].plot(episodes, [np.nan if row[key] is None else row[key] for row in history], label=label)
    for ax in axes:
        ax.axvline(best_episode, color='black', linestyle='--', alpha=.6, label=f'Best: EP {best_episode}')
        ax.set_xlabel('Completed training episodes')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Weighted final objective')
    axes[1].set_ylabel('Ratio')
    axes[1].set_ylim(0, 1.05)
    fig.savefig(save_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_training_history(history, save_path=None, show=True):
    if not show:
        plt.switch_backend('Agg')
    episodes = np.asarray(history['episode'])
    if not episodes.size:
        return None
    unique, groups = np.unique(episodes, return_inverse=True)
    counts = np.bincount(groups)
    metrics = [('actor_loss', 'Actor loss'), ('critic_loss', 'Critic loss'),
               ('rewards', 'Reward'), ('coverage', 'Coverage'),
               ('mean_satisfaction', 'OFDMA satisfaction'), ('connectivity', 'Mesh join reward (0/1)')]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for ax, (key, label) in zip(axes.flat, metrics):
        values = np.asarray(history[key], dtype=float)
        if values.shape != episodes.shape:
            raise ValueError('History metrics must be aligned one-dimensional arrays')
        ax.plot(unique, np.bincount(groups, weights=values) / counts, linewidth=1)
        ax.set(ylabel=label, xlabel='Episode')
        # sharex=True hides tick labels on the upper rows by default.
        ax.tick_params(axis='x', labelbottom=True)
        ax.grid(True, alpha=0.3)
    fig.suptitle('AP mesh / OFDMA training (mean per placement)')
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_batch_prediction_analysis(trials, summary, output_dir, show=False):
    """Export paired final-deployment comparisons as PNG and vector PDF.

    Run-level CIs resample whole paired trials. Pooled client ECDFs are
    descriptive distributions, not independent-client significance tests.
    """
    from matplotlib.ticker import PercentFormatter
    from textwrap import fill
    from training_method.Prediction_analysis import ANALYSIS_METRICS, METHOD_LABELS
    if not trials:
        raise ValueError('At least one paired trial is required')
    if not show:
        plt.switch_backend('Agg')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    colors = {'gnn_cnn': '#2563EB', 'random': '#EA580C'}
    methods = tuple(METHOD_LABELS)
    primary = ANALYSIS_METRICS[:3]
    saved = []

    def finish(fig, name):
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        for extension in ('png', 'pdf'):
            path = output / f'{name}.{extension}'
            fig.savefig(path, dpi=180, bbox_inches='tight')
            saved.append(str(path))
        if show:
            plt.show()
        plt.close(fig)

    def values(method, key):
        return np.asarray([trial[method]['metrics'][key] for trial in trials], dtype=float)

    def decorate(ax, label, percentage=False):
        ax.set_ylabel(fill(label, 38))
        if percentage:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylim(-0.03, 1.05)
        ax.grid(True, alpha=0.22)
        ax.set_axisbelow(True)

    def ecdf(ax, data, method, label=None):
        ordered = np.sort(np.asarray(data, dtype=float))
        ordered = ordered[np.isfinite(ordered)]
        if len(ordered):
            ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where='post',
                    color=colors[method], label=label or METHOD_LABELS[method], linewidth=1.7)
        return len(ordered)

    # The three requested quantities, one point per completed deployment.
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (key, label, _) in zip(axes, primary):
        for method in methods:
            ax.plot([trial['trial'] for trial in trials], values(method, key),
                    color=colors[method], label=METHOD_LABELS[method], alpha=0.85,
                    marker='o', markersize=2.5, linewidth=1)
        decorate(ax, label, True)
    axes[0].legend(loc='best')
    axes[-1].set_xlabel('Paired trial (same map and clients for both methods)')
    fig.suptitle(f'Final deployment metrics | {len(trials)} paired trials')
    finish(fig, 'per_trial_metrics')

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for column, (key, label, _) in enumerate(primary):
        data = [values(method, key) for method in methods]
        finite = [array[np.isfinite(array)] for array in data]
        if any(len(array) for array in finite):
            boxes = axes[0, column].boxplot(finite, labels=[METHOD_LABELS[m] for m in methods],
                                           patch_artist=True, showmeans=True)
            for box, method in zip(boxes['boxes'], methods):
                box.set(facecolor=colors[method], alpha=0.45)
        decorate(axes[0, column], label, True)
        for method, array in zip(methods, data):
            ecdf(axes[1, column], array, method)
        axes[1, column].set(xlabel=fill(label, 32), ylabel='Empirical cumulative probability', ylim=(0, 1.02))
        axes[1, column].xaxis.set_major_formatter(PercentFormatter(1))
        axes[1, column].grid(True, alpha=0.22)
    if axes[1, 0].lines:
        axes[1, 0].legend()
    fig.suptitle('Trial distributions | box: median / IQR, triangle: mean; ECDF: all trials')
    finish(fig, 'metric_distributions')

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, (key, label, percentage) in zip(axes.flat, ANALYSIS_METRICS):
        for index, method in enumerate(methods):
            stats = summary['methods'][method][key]
            if stats['mean'] is None:
                continue
            ax.bar(index, stats['mean'], color=colors[method], alpha=0.8, width=0.55)
            if stats['mean_ci95'] is not None:
                low, high = stats['mean_ci95']
                ax.vlines(index, low, high, color='#111827', linewidth=1.7)
                ax.hlines([low, high], index - 0.10, index + 0.10, color='#111827', linewidth=1.7)
        ax.set_xticks(range(2), [METHOD_LABELS[method] for method in methods])
        decorate(ax, label, percentage)
    fig.suptitle('Trial means with 95% bootstrap confidence intervals | not client-level CIs')
    finish(fig, 'summary_metrics')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, label, _) in zip(axes, primary):
        delta = (values('gnn_cnn', key) - values('random', key)) * 100
        delta = delta[np.isfinite(delta)]
        if len(delta):
            ax.hist(delta, bins=min(15, max(1, int(np.sqrt(len(delta))))),
                    color=colors['gnn_cnn'], alpha=0.75, edgecolor='white')
            stats = summary['paired_difference_gnn_minus_random'][key]
            ax.axvline(stats['mean'] * 100, color=colors['gnn_cnn'], linestyle='--', label='Mean difference')
            interval = stats['mean_ci95']
            note = (f'Mean: {stats["mean"] * 100:+.2f} pp\n'
                    f'Win: {stats["win_rate"]:.1%} | Tie: {stats["tie_rate"]:.1%}')
            if interval is not None:
                note += f'\n95% CI: [{interval[0] * 100:+.2f}, {interval[1] * 100:+.2f}] pp'
            ax.text(0.03, 0.96, note, transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='#D1D5DB'))
        ax.axvline(0, color='#111827', linewidth=1)
        ax.set(title=fill(label, 32), xlabel='GNN + CNN minus Random AP (percentage points)', ylabel='Trials')
        ax.grid(True, alpha=0.22)
    fig.suptitle('Paired differences | positive favors GNN + CNN')
    finish(fig, 'paired_differences')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, label, percentage) in zip(axes, (
            ('throughput_mbps', 'Client actual throughput (Mbps)', False),
            ('satisfaction_ratio', 'Client throughput / demand', True),
            ('sinr_db', 'Associated-client SINR (dB)', False))):
        for method in methods:
            data = [row[key] for trial in trials for row in trial[method]['allocation_results']
                    if row[key] is not None]
            ecdf(ax, data, method, f'{METHOD_LABELS[method]} (n={len(data)})')
        ax.set(xlabel=label, ylabel='Empirical cumulative probability', ylim=(0, 1.02))
        if percentage:
            ax.xaxis.set_major_formatter(PercentFormatter(1))
        if ax.lines:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No clients / valid samples', ha='center', transform=ax.transAxes)
        ax.grid(True, alpha=0.22)
    fig.suptitle('Pooled client ECDFs | throughput includes unassociated zeros; SINR excludes unassociated clients')
    finish(fig, 'client_distributions')
    return saved
