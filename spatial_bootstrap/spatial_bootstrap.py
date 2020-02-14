import pandas as pd
import geopandas as gpd
import fiona

def spatial_bootstrap(df,w,n):
    # ```
    # Spatial Bootsrapping (random draws with replacement, including spatial neighbours at each draw)
    #
    # Input:
    # df = pandas dataframe.
    # w = spatial weight matrix with indices corresponding to df.
    # n = number of bootstrapped samples to be returned
    #
    # Output:
    # (1) The sampled pandas dataframe
    #
    # Example: sboot_df = spatial_bootstrap(df,w,1000,degree=2)
    # ```
    
    indices = df.index.values
    
    sb_indices = []
    len_sb_indices = 0
    
    while len_sb_indices < n:
        
        random_idx = np.random.choice(indices)
        neighs_random_idx = w.neighbors[random_idx]
        
        idx_list_iteration = neighs_random_idx
        idx_list_iteration.append(random_idx)
        idx_list_iteration = list(set(pd.core.common.flatten(idx_list_iteration)))
        
        sb_indices.append(idx_list_iteration)
        sb_indices = list(pd.core.common.flatten(sb_indices))
        
        len_sb_indices = len(sb_indices)
        
    sb_indices = sb_indices[:n]
    
    return df.iloc[sb_indices]