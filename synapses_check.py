import pandas as pd
from collections import defaultdict

from config.config_test2 import CONFIG

def count_synapses():

    layers = CONFIG['layers']
    inter_layer_conn = CONFIG['inter_layer_connections']
    results = {
        'intra_layer': defaultdict(lambda: defaultdict(int)),
        'inter_layer': defaultdict(lambda: defaultdict(int)),
        'by_type': {'excitatory': 0, 'inhibitory': 0},
        'by_source': defaultdict(int),
        'by_target': defaultdict(int),
        'total': 0
    }
    incoming_recurrent_by_layer = defaultdict(int)
    incoming_recurrent_by_layer_type = defaultdict(lambda: defaultdict(int))

    external_by_layer = defaultdict(int)
    external_by_layer_type = defaultdict(lambda: defaultdict(int))
    cell_types = ['E', 'PV', 'SOM', 'VIP']

    print("="*80)
    print("V1 COLUMN SYNAPSE ANALYSIS")
    print("="*80)

    print("\n### INTRA-LAYER CONNECTIONS ###\n")
    for layer_name, layer_config in layers.items():
        print(f"\n{layer_name}:")
        layer_total = 0

        for src_type in cell_types:
            if src_type not in layer_config['neuron_counts']:
                continue
            n_src = layer_config['neuron_counts'][src_type]

            for tgt_type in cell_types:
                if tgt_type not in layer_config['neuron_counts']:
                    continue
                n_tgt = layer_config['neuron_counts'][tgt_type]

                conn_key = f'{src_type}_{tgt_type}'
                prob = layer_config['connection_prob'].get(conn_key, 0)

                n_synapses = int(n_src * n_tgt * prob)

                if n_synapses > 0:
                    print(f"  {src_type} → {tgt_type}: {n_synapses:,} synapses "
                          f"({n_src} × {n_tgt} × {prob:.4f})")

                    results['intra_layer'][layer_name][conn_key] = n_synapses
                    layer_total += n_synapses

                    if src_type == 'E':
                        results['by_type']['excitatory'] += n_synapses
                    else:
                        results['by_type']['inhibitory'] += n_synapses

                    results['by_source'][f'{layer_name}_{src_type}'] += n_synapses
                    results['by_target'][f'{layer_name}_{tgt_type}'] += n_synapses
                    results['total'] += n_synapses

                    incoming_recurrent_by_layer[layer_name] += n_synapses
                    incoming_recurrent_by_layer_type[layer_name][tgt_type] += n_synapses

        print(f"  → {layer_name} TOTAL: {layer_total:,} synapses")

    print("\n\n### INTER-LAYER CONNECTIONS ###\n")
    inter_total = 0

    for (src_layer, tgt_layer), connections in inter_layer_conn.items():
        print(f"\n{src_layer} → {tgt_layer}:")
        pair_total = 0

        src_config = layers[src_layer]
        tgt_config = layers[tgt_layer]

        for src_type in cell_types:
            if src_type not in src_config['neuron_counts']:
                continue
            n_src = src_config['neuron_counts'][src_type]

            for tgt_type in cell_types:
                if tgt_type not in tgt_config['neuron_counts']:
                    continue
                n_tgt = tgt_config['neuron_counts'][tgt_type]

                conn_key = f'{src_type}_{tgt_type}'
                prob = connections.get(conn_key, 0)

                n_synapses = int(n_src * n_tgt * prob)

                if n_synapses > 0:
                    print(f"  {src_type} → {tgt_type}: {n_synapses:,} synapses "
                          f"({n_src} × {n_tgt} × {prob:.4f})")

                    results['inter_layer'][(src_layer, tgt_layer)][conn_key] = n_synapses
                    pair_total += n_synapses

                    if src_type == 'E':
                        results['by_type']['excitatory'] += n_synapses
                    else:
                        results['by_type']['inhibitory'] += n_synapses

                    results['by_source'][f'{src_layer}_{src_type}'] += n_synapses
                    results['by_target'][f'{tgt_layer}_{tgt_type}'] += n_synapses
                    results['total'] += n_synapses

                    incoming_recurrent_by_layer[tgt_layer] += n_synapses
                    incoming_recurrent_by_layer_type[tgt_layer][tgt_type] += n_synapses

        print(f"  → {src_layer}→{tgt_layer} TOTAL: {pair_total:,} synapses")
        inter_total += pair_total

    print("\n\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    intra_total = sum(sum(conns.values()) for conns in results['intra_layer'].values())

    print(f"\nTotal recurrent synapses: {results['total']:,}")
    print(f"  Intra-layer:  {intra_total:,} ({100*intra_total/results['total']:.1f}%)")
    print(f"  Inter-layer:  {inter_total:,} ({100*inter_total/results['total']:.1f}%)")

    print(f"\nBy synapse type:")
    print(f"  Excitatory:   {results['by_type']['excitatory']:,} "
          f"({100*results['by_type']['excitatory']/results['total']:.1f}%)")
    print(f"  Inhibitory:   {results['by_type']['inhibitory']:,} "
          f"({100*results['by_type']['inhibitory']/results['total']:.1f}%)")

    print(f"\nTotal neurons by layer:")
    neuron_counts_by_layer = {}
    total_neurons = 0
    for layer_name, layer_config in layers.items():
        layer_n = sum(layer_config['neuron_counts'].values())
        neuron_counts_by_layer[layer_name] = layer_n
        total_neurons += layer_n
        print(f"  {layer_name}: {layer_n:,}")
    print(f"  TOTAL: {total_neurons:,}")

    print(f"\nAverage synapses per neuron (global, recurrent only): {results['total']/total_neurons:.1f}")

    print("\n\n" + "="*80)
    print("EXTERNAL INPUTS (Poisson)")
    print("="*80)

    total_ext = 0
    for layer_name, layer_config in layers.items():
        print(f"\n{layer_name}:")
        for cell_type, poisson_config in layer_config['poisson_inputs'].items():
            if cell_type not in layer_config['neuron_counts']:
                continue
            n_cells = layer_config['neuron_counts'][cell_type]
            n_inputs = poisson_config['N']
            n_ext_synapses = n_cells * n_inputs
            total_ext += n_ext_synapses
            external_by_layer[layer_name] += n_ext_synapses
            external_by_layer_type[layer_name][cell_type] += n_ext_synapses
            print(f"  {cell_type}: {n_ext_synapses:,} external synapses "
                  f"({n_cells} cells × {n_inputs} inputs)")

    print(f"\nTotal external synapses: {total_ext:,}")
    print(f"Total recurrent synapses: {results['total']:,}")
    print(f"Grand total (external + recurrent): {total_ext + results['total']:,}")

    print("\n\n" + "="*80)
    print("PER-LAYER SYNAPSES PER NEURON (incoming)")
    print("="*80)

    rows = []
    for layer_name in layers.keys():
        n_neurons = neuron_counts_by_layer[layer_name]
        rec_in = incoming_recurrent_by_layer[layer_name]
        ext_in = external_by_layer[layer_name]
        rec_per_neuron = rec_in / n_neurons if n_neurons else 0.0
        total_per_neuron = (rec_in + ext_in) / n_neurons if n_neurons else 0.0
        rows.append({
            'Layer': layer_name,
            'Neurons': n_neurons,
            'Incoming recurrent synapses': rec_in,
            'Incoming external synapses': ext_in,
            'Synapses/neuron (recurrent only)': round(rec_per_neuron, 2),
            'Synapses/neuron (recurrent + external)': round(total_per_neuron, 2)
        })

    df_layer = pd.DataFrame(rows).set_index('Layer').sort_index()
    print("\n>>> Per-layer incoming synapses & synapses/neuron (recurrent + external):\n")
    print(df_layer.to_string())

    print("\n\n" + "="*80)
    print("PER-LAYER, PER-CELL-TYPE SYNAPSES PER NEURON (incoming)")
    print("="*80)
    rows_ct = []
    for layer_name, layer_config in layers.items():
        for ct in cell_types:
            if ct not in layer_config['neuron_counts']:
                continue
            n_ct = layer_config['neuron_counts'][ct]
            rec_ct = incoming_recurrent_by_layer_type[layer_name][ct]
            ext_ct = external_by_layer_type[layer_name][ct]
            rec_ct_per_neuron = rec_ct / n_ct if n_ct else 0.0
            total_ct_per_neuron = (rec_ct + ext_ct) / n_ct if n_ct else 0.0
            rows_ct.append({
                'Layer': layer_name,
                'CellType': ct,
                'Neurons': n_ct,
                'Incoming recurrent synapses': rec_ct,
                'Incoming external synapses': ext_ct,
                'Synapses/neuron (recurrent only)': round(rec_ct_per_neuron, 2),
                'Synapses/neuron (recurrent + external)': round(total_ct_per_neuron, 2)
            })

    if rows_ct:
        df_ct = pd.DataFrame(rows_ct).set_index(['Layer', 'CellType']).sort_index()
        print("\n>>> Per-layer, per-cell-type incoming synapses & synapses/neuron:\n")
        print(df_ct.to_string())
    else:
        print("No per-cell-type data available in neuron_counts; skipping.")

    print("\n\n" + "="*80)
    print("REALISM CHECK")
    print("="*80)
    print("\nTypical values for cortical neurons:")
    print("  - Excitatory neurons: 5,000-10,000 synapses per neuron")
    print("  - Inhibitory neurons: 1,000-5,000 synapses per neuron")
    print("  - E/I ratio: typically 80/20 to 85/15")

    ei_ratio = 100 * results['by_type']['excitatory'] / results['total'] if results['total'] else 0
    print(f"\nYour model:")
    print(f"  - E/I ratio: {ei_ratio:.1f}/{100-ei_ratio:.1f}")
    print(f"  - Avg synapses/neuron (recurrent only, global): {results['total']/total_neurons:.1f}")
    print(f"  - Avg synapses/neuron (incl. external, global): {(results['total']+total_ext)/total_neurons:.1f}")

    results['incoming_recurrent_by_layer'] = dict(incoming_recurrent_by_layer)
    results['incoming_recurrent_by_layer_type'] = {L: dict(cts) for L, cts in incoming_recurrent_by_layer_type.items()}
    results['external_by_layer'] = dict(external_by_layer)
    results['external_by_layer_type'] = {L: dict(cts) for L, cts in external_by_layer_type.items()}
    results['per_layer_summary'] = df_layer
    results['per_layer_celltype_summary'] = df_ct if rows_ct else None

    return results

if __name__ == "__main__":
    results = count_synapses()
