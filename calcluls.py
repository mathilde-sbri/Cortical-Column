import pandas as pd
import numpy as np


scale_conductances = True
if scale_conductances:
    try:
        cond_df = pd.read_csv('conductance2.csv', index_col=0)
        

        CONDUCTANCE_SCALE = 0.3  
        
        cond_df_scaled = cond_df * CONDUCTANCE_SCALE
        cond_df_scaled.to_csv('conductance_scaled_realistic.csv')
        
        print(f"\n✓ Also processed conductances (scaled by {CONDUCTANCE_SCALE}x)")
        print(f"  Saved to: conductance_scaled_realistic.csv")
    except FileNotFoundError:
        print("\n⚠ conductance_scaled.csv not found - skipping conductance scaling")

print("\n✓ Done!")