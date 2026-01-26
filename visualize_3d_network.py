"""
Beautiful 3D Visualizations for Cortical Column Spiking Neural Network

This module creates presentation-ready 3D visualizations including:
1. 3D Network Architecture with neuron positions
2. Animated spike activity propagation
3. Connectivity visualization (inter-layer and intra-layer)
4. LFP field visualization in 3D
5. Interactive visualizations using plotly
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import brian2 as b2
from brian2 import *
from config.config2 import CONFIG
from src.column import CorticalColumn
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


# Color scheme for different neuron types
NEURON_COLORS = {
    'E': '#2ECC71',      # Green for Excitatory
    'PV': '#E74C3C',     # Red for Parvalbumin
    'SOM': '#3498DB',    # Blue for Somatostatin
    'VIP': '#F39C12',    # Gold for VIP
}

LAYER_COLORS = {
    'L1': '#E8DAEF',
    'L23': '#D7BDE2',
    'L4AB': '#C39BD3',
    'L4C': '#AF7AC5',
    'L5': '#9B59B6',
    'L6': '#7D3C98',
}


def create_3d_network_structure(column, figsize=(16, 12), elevation=20, azimuth=45):
    """
    Create a beautiful 3D visualization of the entire cortical column structure
    showing all neurons positioned in 3D space, color-coded by type.

    Parameters
    ----------
    column : CorticalColumn
        The cortical column object with all layers
    figsize : tuple
        Figure size in inches
    elevation : float
        Elevation angle for 3D view
    azimuth : float
        Azimuth angle for 3D view

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='#F8F9F9')

    layer_order = ['L23', 'L4AB', 'L4C', 'L5', 'L6']

    # Plot neurons for each layer
    for layer_name in layer_order:
        if layer_name not in column.layers:
            continue

        layer = column.layers[layer_name]
        layer_config = CONFIG['layers'][layer_name]

        for pop_name, neuron_group in layer.neuron_groups.items():
            # Get neuron positions
            x = neuron_group.x / b2.mm
            y = neuron_group.y / b2.mm
            z = neuron_group.z / b2.mm

            # Plot with appropriate color and size
            color = NEURON_COLORS.get(pop_name, '#95A5A6')
            alpha = 0.6 if pop_name == 'E' else 0.8
            size = 5 if pop_name == 'E' else 15

            ax.scatter(x, y, z, c=color, s=size, alpha=alpha,
                      label=f'{layer_name}-{pop_name}',
                      edgecolors='none')

    # Add layer boundaries
    for layer_name in layer_order:
        if layer_name not in CONFIG['layers']:
            continue
        z_range = CONFIG['layers'][layer_name]['coordinates']['z']
        x_range = CONFIG['layers'][layer_name]['coordinates']['x']
        y_range = CONFIG['layers'][layer_name]['coordinates']['y']

        # Create semi-transparent layer boxes
        xx, yy = np.meshgrid([x_range[0], x_range[1]],
                             [y_range[0], y_range[1]])

        # Bottom and top of layer
        for z_val in z_range:
            ax.plot_surface(xx, yy, np.full_like(xx, z_val),
                          alpha=0.05, color=LAYER_COLORS[layer_name])

    # Add electrode positions
    electrode_positions = CONFIG['electrode_positions']
    elec_x = [pos[0] for pos in electrode_positions]
    elec_y = [pos[1] for pos in electrode_positions]
    elec_z = [pos[2] for pos in electrode_positions]
    ax.scatter(elec_x, elec_y, elec_z, c='black', s=50, marker='D',
              label='Electrodes', edgecolors='gold', linewidths=2, alpha=0.9)

    # Connect electrodes with a line
    ax.plot(elec_x, elec_y, elec_z, 'k--', linewidth=1.5, alpha=0.5)

    # Styling
    ax.set_xlabel('X (mm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=14, fontweight='bold')
    ax.set_zlabel('Z - Cortical Depth (mm)', fontsize=14, fontweight='bold')
    ax.set_title('3D Cortical Column Architecture\nMulti-Layer Spiking Neural Network',
                fontsize=18, fontweight='bold', pad=20)

    # Set viewing angle
    ax.view_init(elev=elevation, azim=azimuth)

    # Legend with better positioning
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1),
             fontsize=10, framealpha=0.9)

    # Grid styling
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Set aspect ratio to match actual column dimensions
    # X: 0.3mm, Y: 0.3mm, Z: ~1.72mm (from -0.62 to 1.1)
    ax.set_box_aspect([0.3, 0.3, 1.72])

    plt.tight_layout()
    return fig


def create_interactive_3d_network(column):
    """
    Create an interactive 3D visualization using Plotly for web/presentation viewing.

    Parameters
    ----------
    column : CorticalColumn
        The cortical column object

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    fig = go.Figure()

    layer_order = ['L23', 'L4AB', 'L4C', 'L5', 'L6']

    # Add neurons
    for layer_name in layer_order:
        if layer_name not in column.layers:
            continue

        layer = column.layers[layer_name]

        for pop_name, neuron_group in layer.neuron_groups.items():
            x = neuron_group.x / b2.mm
            y = neuron_group.y / b2.mm
            z = neuron_group.z / b2.mm

            color = NEURON_COLORS.get(pop_name, '#95A5A6')
            size = 2 if pop_name == 'E' else 4

            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                name=f'{layer_name}-{pop_name}',
                marker=dict(
                    size=size,
                    color=color,
                    opacity=0.7,
                ),
                text=[f'{layer_name}-{pop_name}-{i}' for i in range(len(x))],
                hoverinfo='text'
            ))

    # Add electrodes
    electrode_positions = CONFIG['electrode_positions']
    elec_x = [pos[0] for pos in electrode_positions]
    elec_y = [pos[1] for pos in electrode_positions]
    elec_z = [pos[2] for pos in electrode_positions]

    fig.add_trace(go.Scatter3d(
        x=elec_x, y=elec_y, z=elec_z,
        mode='markers+lines',
        name='Electrodes',
        marker=dict(size=6, color='black', symbol='diamond'),
        line=dict(color='black', width=2, dash='dash')
    ))

    # Layout with correct aspect ratio
    # X: 0.3mm, Y: 0.3mm, Z: ~1.72mm - normalize to make Z=1
    aspect_x = 0.3 / 1.72
    aspect_y = 0.3 / 1.72
    aspect_z = 1.0

    fig.update_layout(
        title=dict(
            text='Interactive 3D Cortical Column Network',
            font=dict(size=20, family='Arial Black')
        ),
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Cortical Depth (mm)',
            aspectmode='manual',
            aspectratio=dict(x=aspect_x, y=aspect_y, z=aspect_z),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)
            ),
            bgcolor='#F8F9F9'
        ),
        width=1200,
        height=900,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )

    return fig


def visualize_spike_propagation_3d(column, spike_monitors, time_window=(200, 400),
                                   num_frames=50, save_animation=True,
                                   output_file='spike_animation.mp4'):
    """
    Create an animated 3D visualization of spike propagation through the network.

    Parameters
    ----------
    column : CorticalColumn
        The cortical column object
    spike_monitors : dict
        Dictionary of spike monitors from simulation
    time_window : tuple
        (start_ms, end_ms) time window to visualize
    num_frames : int
        Number of animation frames
    save_animation : bool
        Whether to save as video file
    output_file : str
        Output filename for animation

    Returns
    -------
    fig : matplotlib.figure.Figure
    anim : matplotlib.animation.FuncAnimation (if save_animation is False)
    """
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='#F8F9F9')

    # Collect all neuron positions and spike times
    neuron_data = []
    for layer_name, monitors in spike_monitors.items():
        if layer_name not in column.layers:
            continue

        layer = column.layers[layer_name]

        for pop_name, mon in monitors.items():
            if 'spikes' not in mon.name:
                continue

            pop_key = pop_name.replace('_spikes', '')
            if pop_key not in layer.neuron_groups:
                continue

            neuron_group = layer.neuron_groups[pop_key]

            # Get spike times and neuron indices
            spike_times = np.array(mon.t / b2.ms)
            spike_indices = np.array(mon.i)

            # Filter to time window
            mask = (spike_times >= time_window[0]) & (spike_times <= time_window[1])
            spike_times = spike_times[mask]
            spike_indices = spike_indices[mask]

            # Get positions of spiking neurons
            x = neuron_group.x / b2.mm
            y = neuron_group.y / b2.mm
            z = neuron_group.z / b2.mm

            color = NEURON_COLORS.get(pop_key, '#95A5A6')

            neuron_data.append({
                'x': x, 'y': y, 'z': z,
                'spike_times': spike_times,
                'spike_indices': spike_indices,
                'color': color,
                'layer': layer_name,
                'pop': pop_key
            })

    # Plot all neurons as background (gray, small)
    for data in neuron_data:
        ax.scatter(data['x'], data['y'], data['z'],
                  c='lightgray', s=1, alpha=0.3, edgecolors='none')

    # Time points for animation
    time_points = np.linspace(time_window[0], time_window[1], num_frames)
    spike_window = (time_window[1] - time_window[0]) / num_frames * 3  # Show spikes for 3 frames

    scatter_plots = []

    def init():
        ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Cortical Depth (mm)', fontsize=12, fontweight='bold')
        ax.set_title('Spike Propagation in Cortical Column',
                    fontsize=16, fontweight='bold')
        # Set aspect ratio to match actual column dimensions
        ax.set_box_aspect([0.3, 0.3, 1.72])
        return []

    def update(frame):
        # Clear previous active spikes
        for scatter in scatter_plots:
            scatter.remove()
        scatter_plots.clear()

        current_time = time_points[frame]

        # Plot active spikes
        for data in neuron_data:
            # Find spikes in current window
            mask = (data['spike_times'] >= current_time - spike_window/2) & \
                   (data['spike_times'] <= current_time + spike_window/2)

            active_indices = data['spike_indices'][mask]

            if len(active_indices) > 0:
                active_x = data['x'][active_indices]
                active_y = data['y'][active_indices]
                active_z = data['z'][active_indices]

                scatter = ax.scatter(active_x, active_y, active_z,
                                   c=data['color'], s=50, alpha=0.9,
                                   edgecolors='white', linewidths=1)
                scatter_plots.append(scatter)

        # Update title with current time
        ax.set_title(f'Spike Propagation in Cortical Column\nTime: {current_time:.1f} ms',
                    fontsize=16, fontweight='bold')

        return scatter_plots

    anim = FuncAnimation(fig, update, init_func=init, frames=num_frames,
                        interval=100, blit=False, repeat=True)

    if save_animation:
        print(f"Saving animation to {output_file}...")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Cortical Column Viz'), bitrate=1800)
        anim.save(output_file, writer=writer)
        print("Animation saved!")
        return fig, None
    else:
        return fig, anim


def visualize_lfp_field_3d(lfp_signals, time_array, electrode_positions,
                           time_points=[200, 500, 800, 1200]):
    """
    Create beautiful 3D field potential visualizations at multiple time points.

    Parameters
    ----------
    lfp_signals : dict
        LFP signals from electrodes
    time_array : array
        Time array in ms
    electrode_positions : list
        List of (x, y, z) electrode positions
    time_points : list
        Time points (in ms) to visualize

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_timepoints = len(time_points)
    fig = plt.figure(figsize=(18, 4*n_timepoints))

    # Create interpolation grid
    xi = np.linspace(-0.2, 0.2, 50)
    yi = np.linspace(-0.2, 0.2, 50)
    zi = np.linspace(-1.0, 1.3, 80)
    Xi, Yi, Zi = np.meshgrid(xi, yi, zi)

    for idx, time_ms in enumerate(time_points):
        ax = fig.add_subplot(n_timepoints, 1, idx+1, projection='3d')

        # Find closest time index
        time_idx = np.argmin(np.abs(time_array - time_ms))

        # Get LFP values at this time
        lfp_values = np.array([lfp_signals[i][time_idx] for i in range(len(lfp_signals))])

        # Electrode positions
        elec_coords = np.array(electrode_positions)

        # Interpolate LFP field
        Vi = griddata(elec_coords, lfp_values, (Xi, Yi, Zi), method='linear')

        # Create isosurfaces
        # Normalize LFP for visualization
        vmin, vmax = np.nanpercentile(Vi, [10, 90])
        levels = np.linspace(vmin, vmax, 8)

        # Plot isosurfaces with different colors
        cmap = cm.get_cmap('RdBu_r')

        for i, level in enumerate(levels[1:-1]):
            try:
                from skimage import measure
                verts, faces, _, _ = measure.marching_cubes(Vi, level=level)

                # Scale vertices back to real coordinates
                verts_scaled = verts.copy()
                verts_scaled[:, 0] = xi[0] + (verts[:, 0] / Vi.shape[0]) * (xi[-1] - xi[0])
                verts_scaled[:, 1] = yi[0] + (verts[:, 1] / Vi.shape[1]) * (yi[-1] - yi[0])
                verts_scaled[:, 2] = zi[0] + (verts[:, 2] / Vi.shape[2]) * (zi[-1] - zi[0])

                color = cmap((level - vmin) / (vmax - vmin))

                ax.plot_trisurf(verts_scaled[:, 0], verts_scaled[:, 1], faces,
                              verts_scaled[:, 2], color=color, alpha=0.3)
            except:
                pass

        # Plot electrodes
        ax.scatter(elec_coords[:, 0], elec_coords[:, 1], elec_coords[:, 2],
                  c='black', s=80, marker='D', edgecolors='gold', linewidths=2)

        # Styling
        ax.set_xlabel('X (mm)', fontsize=11)
        ax.set_ylabel('Y (mm)', fontsize=11)
        ax.set_zlabel('Cortical Depth (mm)', fontsize=11)
        ax.set_title(f'LFP Field at t = {time_ms} ms', fontsize=14, fontweight='bold')
        ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig


def visualize_connectivity_3d(column, sample_fraction=0.01, connection_type='inter'):
    """
    Visualize synaptic connections in 3D space.

    Parameters
    ----------
    column : CorticalColumn
        The cortical column object
    sample_fraction : float
        Fraction of connections to visualize (to avoid clutter)
    connection_type : str
        'inter' for inter-layer, 'intra' for intra-layer connections

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='#F8F9F9')

    # Plot all neurons first (as background)
    for layer_name, layer in column.layers.items():
        for pop_name, neuron_group in layer.neuron_groups.items():
            x = neuron_group.x / b2.mm
            y = neuron_group.y / b2.mm
            z = neuron_group.z / b2.mm

            color = NEURON_COLORS.get(pop_name, '#95A5A6')
            ax.scatter(x, y, z, c=color, s=3, alpha=0.3, edgecolors='none')

    # Plot connections
    if connection_type == 'inter':
        synapses_dict = column.inter_layer_synapses
        title = 'Inter-Layer Synaptic Connections'
    else:
        # Collect intra-layer connections
        synapses_dict = {}
        for layer_name, layer in column.layers.items():
            for syn_name, syn in layer.synapses.items():
                synapses_dict[f"{layer_name}_{syn_name}"] = syn
        title = 'Intra-Layer Synaptic Connections'

    for conn_name, synapse in synapses_dict.items():
        if len(synapse.i) == 0:
            continue

        # Sample connections
        n_conns = len(synapse.i)
        n_sample = max(1, int(n_conns * sample_fraction))
        sample_indices = np.random.choice(n_conns, size=n_sample, replace=False)

        # Get pre and post neuron positions
        pre_group = synapse.source
        post_group = synapse.target

        pre_indices = synapse.i[sample_indices]
        post_indices = synapse.j[sample_indices]

        pre_x = pre_group.x[pre_indices] / b2.mm
        pre_y = pre_group.y[pre_indices] / b2.mm
        pre_z = pre_group.z[pre_indices] / b2.mm

        post_x = post_group.x[post_indices] / b2.mm
        post_y = post_group.y[post_indices] / b2.mm
        post_z = post_group.z[post_indices] / b2.mm

        # Determine if excitatory or inhibitory
        if '_E_' in conn_name or conn_name.endswith('_E'):
            color = '#2ECC71'  # Green for excitatory
            alpha = 0.15
        else:
            color = '#E74C3C'  # Red for inhibitory
            alpha = 0.2

        # Plot connections as lines
        for i in range(n_sample):
            ax.plot([pre_x[i], post_x[i]],
                   [pre_y[i], post_y[i]],
                   [pre_z[i], post_z[i]],
                   color=color, alpha=alpha, linewidth=0.5)

    # Styling
    ax.set_xlabel('X (mm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=14, fontweight='bold')
    ax.set_zlabel('Cortical Depth (mm)', fontsize=14, fontweight='bold')
    ax.set_title(f'{title}\n(sampled at {sample_fraction*100:.1f}%)',
                fontsize=16, fontweight='bold', pad=20)
    ax.view_init(elev=20, azim=45)

    # Set aspect ratio to match actual column dimensions
    ax.set_box_aspect([0.3, 0.3, 1.72])

    plt.tight_layout()
    return fig


def create_comprehensive_3d_figure(column, spike_monitors, lfp_signals,
                                   time_array, electrode_positions):
    """
    Create a comprehensive multi-panel 3D figure showing multiple aspects.

    Parameters
    ----------
    column : CorticalColumn
        The cortical column object
    spike_monitors : dict
        Spike monitors from simulation
    lfp_signals : dict
        LFP signals
    time_array : array
        Time array
    electrode_positions : list
        Electrode positions

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(20, 10))

    # Panel 1: Network structure
    ax1 = fig.add_subplot(131, projection='3d')
    layer_order = ['L23', 'L4AB', 'L4C', 'L5', 'L6']

    for layer_name in layer_order:
        if layer_name not in column.layers:
            continue
        layer = column.layers[layer_name]
        for pop_name, neuron_group in layer.neuron_groups.items():
            x = neuron_group.x / b2.mm
            y = neuron_group.y / b2.mm
            z = neuron_group.z / b2.mm
            color = NEURON_COLORS.get(pop_name, '#95A5A6')
            size = 2 if pop_name == 'E' else 8
            ax1.scatter(x, y, z, c=color, s=size, alpha=0.6, edgecolors='none')

    ax1.set_xlabel('X (mm)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Y (mm)', fontsize=10, fontweight='bold')
    ax1.set_zlabel('Depth (mm)', fontsize=10, fontweight='bold')
    ax1.set_title('Network Structure', fontsize=12, fontweight='bold')
    ax1.view_init(elev=20, azim=45)
    ax1.set_box_aspect([0.3, 0.3, 1.72])

    # Panel 2: Spike activity snapshot
    ax2 = fig.add_subplot(132, projection='3d')
    time_point = 500  # ms
    time_window = 50  # ms

    for layer_name, monitors in spike_monitors.items():
        if layer_name not in column.layers:
            continue
        layer = column.layers[layer_name]

        for pop_name, mon in monitors.items():
            if 'spikes' not in mon.name:
                continue
            pop_key = pop_name.replace('_spikes', '')
            if pop_key not in layer.neuron_groups:
                continue

            neuron_group = layer.neuron_groups[pop_key]
            spike_times = np.array(mon.t / b2.ms)
            spike_indices = np.array(mon.i)

            mask = (spike_times >= time_point - time_window/2) & \
                   (spike_times <= time_point + time_window/2)
            active_indices = spike_indices[mask]

            if len(active_indices) > 0:
                x = neuron_group.x[active_indices] / b2.mm
                y = neuron_group.y[active_indices] / b2.mm
                z = neuron_group.z[active_indices] / b2.mm
                color = NEURON_COLORS.get(pop_key, '#95A5A6')
                ax2.scatter(x, y, z, c=color, s=30, alpha=0.8, edgecolors='white')

    ax2.set_xlabel('X (mm)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Y (mm)', fontsize=10, fontweight='bold')
    ax2.set_zlabel('Depth (mm)', fontsize=10, fontweight='bold')
    ax2.set_title(f'Active Spikes at {time_point}±{time_window/2}ms',
                 fontsize=12, fontweight='bold')
    ax2.view_init(elev=20, azim=45)
    ax2.set_box_aspect([0.3, 0.3, 1.72])

    # Panel 3: LFP across depth
    ax3 = fig.add_subplot(133)
    elec_coords = np.array(electrode_positions)

    # Create a pleasing color gradient for LFP traces (deep purple to teal to gold)
    n_electrodes = len(lfp_signals)
    lfp_colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_electrodes))

    # Plot LFP traces stacked by depth
    for i in range(len(lfp_signals)):
        lfp = lfp_signals[i]
        depth = elec_coords[i, 2]

        # Subsample for visualization
        time_mask = (time_array >= 200) & (time_array <= 600)
        t_plot = time_array[time_mask]
        lfp_plot = lfp[time_mask]

        # Normalize and offset
        if np.std(lfp_plot) > 0:
            lfp_norm = (lfp_plot - np.mean(lfp_plot)) / np.std(lfp_plot)
        else:
            lfp_norm = lfp_plot

        offset = i * 3
        ax3.plot(t_plot, lfp_norm + offset, linewidth=1.2, alpha=0.9, color=lfp_colors[i])
        ax3.text(t_plot[0] - 20, offset, f'{depth:.2f}mm',
                fontsize=9, va='center', color=lfp_colors[i])

    ax3.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('LFP by Depth', fontsize=11, fontweight='bold')
    ax3.set_title('Local Field Potentials', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yticks([])

    plt.suptitle('Comprehensive 3D Visualization of Cortical Column Dynamics',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def save_all_visualizations(column, spike_monitors, lfp_signals,
                            time_array, electrode_positions,
                            output_dir='3d_visualizations'):
    """
    Generate and save all visualization types.

    Parameters
    ----------
    column : CorticalColumn
        The cortical column object
    spike_monitors : dict
        Spike monitors from simulation
    lfp_signals : dict
        LFP signals
    time_array : array
        Time array in ms
    electrode_positions : list
        List of electrode positions
    output_dir : str
        Directory to save visualizations

    Returns
    -------
    None
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("Generating 3D network structure...")
    fig1 = create_3d_network_structure(column)
    fig1.savefig(f'{output_dir}/network_structure_3d.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {output_dir}/network_structure_3d.png")

    print("\nGenerating interactive 3D network...")
    fig2 = create_interactive_3d_network(column)
    fig2.write_html(f'{output_dir}/interactive_network_3d.html')
    print(f"Saved: {output_dir}/interactive_network_3d.html")

    print("\nGenerating connectivity visualization...")
    fig3 = visualize_connectivity_3d(column, sample_fraction=0.005, connection_type='inter')
    fig3.savefig(f'{output_dir}/connectivity_inter_layer_3d.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: {output_dir}/connectivity_inter_layer_3d.png")

    fig4 = visualize_connectivity_3d(column, sample_fraction=0.005, connection_type='intra')
    fig4.savefig(f'{output_dir}/connectivity_intra_layer_3d.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print(f"Saved: {output_dir}/connectivity_intra_layer_3d.png")

    print("\nGenerating comprehensive figure...")
    fig5 = create_comprehensive_3d_figure(column, spike_monitors, lfp_signals,
                                         time_array, electrode_positions)
    fig5.savefig(f'{output_dir}/comprehensive_3d_view.png', dpi=300, bbox_inches='tight')
    plt.close(fig5)
    print(f"Saved: {output_dir}/comprehensive_3d_view.png")

    print("\nAll visualizations saved successfully!")
    print(f"Check the '{output_dir}/' directory for all files.")


if __name__ == "__main__":
    print("=" * 60)
    print("Beautiful 3D Visualizations for Cortical Column")
    print("=" * 60)

    # Create column
    print("\nCreating cortical column...")
    np.random.seed(CONFIG['simulation']['RANDOM_SEED'])
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']

    column = CorticalColumn(column_id=0, config=CONFIG)

    # Run a short simulation to get spike data
    print("Running simulation...")
    baseline_ms = 500
    stim_ms = 500

    column.network.run(baseline_ms * b2.ms)

    # Add some stimulus
    w_ext_AMPA = CONFIG['synapses']['Q']['EXT_AMPA']
    L4C = column.layers['L4C']
    L4C_E_grp = L4C.neuron_groups['E']
    L4C_E_stim = PoissonInput(L4C_E_grp, 'gE_AMPA',
                              N=60, rate=10*b2.Hz, weight=w_ext_AMPA/2)
    column.network.add(L4C_E_stim)
    column.network.run(stim_ms * b2.ms)

    print("Simulation complete!")

    # Get monitors and data
    all_monitors = column.get_all_monitors()
    spike_monitors = {}
    neuron_groups = {}

    for layer_name, monitors in all_monitors.items():
        spike_monitors[layer_name] = {k: v for k, v in monitors.items() if 'spikes' in k}
        neuron_groups[layer_name] = column.layers[layer_name].neuron_groups

    # Calculate LFP
    print("\nCalculating LFP signals...")
    from src.analysis import calculate_lfp_kernel_method

    electrode_positions = CONFIG['electrode_positions']
    total_sim_ms = baseline_ms + stim_ms

    lfp_signals, time_array = calculate_lfp_kernel_method(
        spike_monitors,
        neuron_groups,
        CONFIG['layers'],
        electrode_positions,
        fs=10000,
        sim_duration_ms=total_sim_ms,
    )

    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    # Generate and save all visualizations
    save_all_visualizations(column, spike_monitors, lfp_signals,
                           time_array, electrode_positions,
                           output_dir='3d_visualizations')

    print("\n" + "=" * 60)
    print("DONE! All visualizations are ready for your presentation.")
    print("=" * 60)
    print("\nTip: Open 'interactive_network_3d.html' in a web browser")
    print("     for an interactive 3D view you can rotate and zoom!")
