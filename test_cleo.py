import pandas as pd
from itertools import product

# --- Load raw CSV data ---
conn_df = pd.read_csv('conn_prob.csv', index_col=0, skipinitialspace=True)
cond_df = pd.read_csv('conductances.csv', index_col=0, skipinitialspace=True)

layers = ['L23', 'L4', 'L5', 'L6']
pops   = ['E', 'PV', 'SOM', 'VIP']

_layer_csv = {
    'L23': {'E_row': 'E2/3',   'PV_row': 'i2/3Pvalb','SOM_row': 'i2/3Sst',  'VIP_row': 'i2/3Htr3a',
            'E_col': 'E2/3',   'PV_col': 'i2/3Pvalb','SOM_col': 'i2/3Sst',  'VIP_col': 'i2/3Htr3a'},
    'L4' : {'E_row': 'E4',     'PV_row': 'i4Pvalb',  'SOM_row': 'i4Sst',    'VIP_row': 'i4Htr3a',
            'E_col': 'E4',     'PV_col': 'i4Pvalb',  'SOM_col': 'i4Sst',    'VIP_col': 'i4Htr3a'},
    'L5' : {'E_row': 'E5',     'PV_row': 'i5Pvalb',  'SOM_row': 'i5Sst',    'VIP_row': 'i5Htr3a',
            'E_col': 'E5',     'PV_col': 'i5Pvalb',  'SOM_col': 'i5Sst',    'VIP_col': 'i5Htr3a'},
    'L6' : {'E_row': 'E6',     'PV_row': 'i6Pvalb',  'SOM_row': 'i6Sst',    'VIP_row': 'i6Htr3a',
            'E_col': 'E6',     'PV_col': 'i6Pvalb',  'SOM_col': 'i6Sst',    'VIP_col': 'i6Htr3a'},
}

def get_prob(layer, pre, post):
    row = _layer_csv[layer][f'{pre}_row']
    col = _layer_csv[layer][f'{post}_col']
    return conn_df.loc[row, col]

def get_cond(layer, pre, post):
    row = _layer_csv[layer][f'{pre}_row']
    col = _layer_csv[layer][f'{post}_col']
    return cond_df.loc[row, col]

# --- scaling functions: old vs new ---

def scale_old_cond(pre, post):
    if pre == 'E' and post == 'E':
        return 2.0
    elif pre == 'E':
        return 1.3
    else:
        return 0.6

def scale_old_prob(pre, post):
    if pre == 'E' and post == 'E':
        return 1.5
    elif pre in ('PV', 'SOM', 'VIP'):
        return 0.8
    else:
        return 1.0

def scale_new_cond(pre, post):
    if pre == 'E' and post == 'E':
        return 0.7
    elif pre == 'E' and post in ('PV', 'SOM', 'VIP'):
        return 1.0
    elif pre in ('PV', 'SOM') and post == 'E':
        return 2.0
    elif pre in ('PV', 'SOM') and post in ('PV', 'SOM', 'VIP'):
        return 1.5
    elif pre == 'VIP' and post in ('PV', 'SOM'):
        return 1.2
    else:
        return 0.8

def scale_new_prob(pre, post):
    if pre == 'E' and post == 'E':
        return 0.7
    elif pre == 'E' and post in ('PV', 'SOM', 'VIP'):
        return 1.1
    elif pre in ('PV', 'SOM') and post == 'E':
        return 1.2
    elif pre in ('PV', 'SOM') and post in ('PV', 'SOM', 'VIP'):
        return 1.0
    elif pre == 'VIP' and post in ('PV', 'SOM'):
        return 1.1
    else:
        return 0.9

# --- build summary table ---

rows = []
for layer in layers:
    for pre, post in product(pops, pops):
        p0 = get_prob(layer, pre, post)
        g0 = get_cond(layer, pre, post)

        # Baseline (from CSV)
        eff0 = p0 * g0

        # Old scaling
        p_old = min(p0 * scale_old_prob(pre, post), 1.0)
        g_old = g0 * scale_old_cond(pre, post)
        eff_old = p_old * g_old

        # New scaling
        p_new = min(p0 * scale_new_prob(pre, post), 1.0)
        g_new = g0 * scale_new_cond(pre, post)
        eff_new = p_new * g_new

        kind = 'E' if pre == 'E' else 'I'

        rows.append({
            'layer': layer,
            'pre': pre,
            'post': post,
            'kind_pre': kind,
            'p0': p0,
            'g0': g0,
            'eff0': eff0,
            'p_old': p_old,
            'g_old': g_old,
            'eff_old': eff_old,
            'p_new': p_new,
            'g_new': g_new,
            'eff_new': eff_new,
        })

df = pd.DataFrame(rows)

# --- summarize E vs I input per postsynaptic population ---

for layer in layers:
    print(f'\n===== {layer} =====')
    for post in pops:
        sub = df[(df['layer'] == layer) & (df['post'] == post)]

        # Baseline
        e0 = sub[sub['kind_pre'] == 'E']['eff0'].sum()
        i0 = sub[sub['kind_pre'] == 'I']['eff0'].sum()

        # Old
        e_old = sub[sub['kind_pre'] == 'E']['eff_old'].sum()
        i_old = sub[sub['kind_pre'] == 'I']['eff_old'].sum()

        # New
        e_new = sub[sub['kind_pre'] == 'E']['eff_new'].sum()
        i_new = sub[sub['kind_pre'] == 'I']['eff_new'].sum()

        def safe_ratio(i, e):
            return i/e if e > 0 else float('inf')

        print(f'Post {post}: '
              f'baseline I/E={safe_ratio(i0, e0):.2f}, '
              f'old I/E={safe_ratio(i_old, e_old):.2f}, '
              f'new I/E={safe_ratio(i_new, e_new):.2f}')
