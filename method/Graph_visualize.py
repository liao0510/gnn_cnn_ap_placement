"""AP/client mesh, SNR links and OFDMA results."""
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
        snr = number(row['snr_db'])
        details.extend([
            f'C{client} {tuple(clients[client])} -> {ap_label(row["ap"])} | '
            f'SNR {snr} dB | MCS {row["mcs"] if row["mcs"] is not None else "-"}',
            f'  RU {row["ru_tones"]} | time {row["time_fraction"]:.0%} | '
            f'{row["throughput_mbps"]:.1f}/{row["demand_mbps"]:.1f} Mbps | '
            f'sat {row["satisfaction_ratio"]:.1%}',
        ])
    details.extend(['', 'AP-AP LINKS (both directions)'])
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
        ax.set_title('AP mesh and client associations | white lines = active links', color='white')
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
                ax.annotate(f'C{client}: {row["snr_db"]:.1f} dB\n{row["throughput_mbps"]:.1f} Mbps',
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
        client_rows.append([f'C{client}', ap_label(row['ap']), number(row['snr_db']),
                            row['mcs'], row['ru_tones'], f'{row["time_fraction"]:.1%}',
                            number(row['demand_mbps']), number(row['throughput_mbps']),
                            f'{row["satisfaction_ratio"]:.1%}'])
    tables(client_rows, ['Client', 'AP', 'SNR (dB)', 'MCS', 'RU tones', 'Time share',
                         'Demand\n(Mbps)', 'Throughput\n(Mbps)', 'Satisfaction'],
           'Client SNR and OFDMA allocation', 'client_details')
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
               ('mean_satisfaction', 'OFDMA satisfaction'), ('connectivity', 'New AP connected (0/1)')]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for ax, (key, label) in zip(axes.flat, metrics):
        values = np.asarray(history[key], dtype=float)
        if values.shape != episodes.shape:
            raise ValueError('History metrics must be aligned one-dimensional arrays')
        ax.plot(unique, np.bincount(groups, weights=values) / counts, linewidth=1)
        ax.set(ylabel=label, xlabel='Episode')
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
