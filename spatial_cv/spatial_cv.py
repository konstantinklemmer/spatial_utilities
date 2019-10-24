import numpy as np
import pandas as pd
import libpysal
import pysal
import pysal.lib

ef spatial_cv_slicing(df,coords,slices=5,k=8,discrete=False,disc_neighbourhood="queen"):
    # ```
    # Spatial Cross-Validation using slicing
    #
    # Input:
    # df = pandas dataframe to be sliced 
    # coords = names of the spatial coordiante columns
    # slices = no. of slices across each spatial dimension
    # k = no. of nearest-neighbours to define the boundary size
    #
    # Output:
    # (1) The sliced dataframe with columns for each slice named "fold_1" - "fold_n" (where n = slices *2).
    #     The value 1 indicates test set, 2 indicates training set and 0 indicates omitted observation
    # (2) Array of names of columns, ["fold_1",...,"fold_n"]
    #
    # Example: sliced_df = spatial_cv_slicing(df,["latitude","longitude"],slices=6,discrete=True,disc_neighbourhood="king")
    # ```
    
    l = [i+1 for i in range(slices)]
    df["temp_id"] = [i+1 for i in range(df.shape[0])]
    df["x_group"] = pd.cut(df[coords[0]],bins=slices,labels=l)
    df["y_group"] = pd.cut(df[coords[1]],bins=slices,labels=l)
    
    kd = pysal.lib.cg.kdtree.KDTree(np.array(df[coords]))
        
    if discrete is False:
        w = pysal.lib.weights.KNN(kd, k)
        
        for q in list(df)[-2::]: #Loop over the two slicing label columns 
            df["s_id"] = df[q] #Define which label column to use for slicing

            for j in np.unique(df["s_id"]): #Loop over the unique labels in the slicing column 

                df[q+str(j)] = 0

                test = df[df["s_id"]==j] #Define test data 
                df.loc[df["temp_id"].isin(np.array(test["temp_id"])),q+str(j)] = 1

                temp_id = [] #Create empty neighbourhood index

                for h in test.index: #Fill neighborhood index using first degree neighbors of test data
                    temp_id = np.unique(np.concatenate([temp_id,w.neighbors[h]]).ravel().astype(np.int32))

                train = df[df["s_id"]!=j] #Define train data 
                train = train.drop(temp_id,errors="ignore") #Exclude neighbors from index
                df.loc[df["temp_id"].isin(np.array(train["temp_id"])),q+str(j)] = 2
        
    else:
        dist = pysal.lib.cg.distance_matrix(np.array(df[coords]))
        u_dist = np.unique(dist)
        k_min_dist = np.sort(u_dist.flatten())[:k]
        
        if disc_neighbourhood in ["queen","queen_2nd"]:
            w = pysal.lib.weights.distance.DistanceBand(kd, threshold=k_min_dist[2],binary=True,p=2)
        else:
            w = pysal.lib.weights.distance.DistanceBand(kd, threshold=k_min_dist[1],binary=True,p=2)
        
        for q in list(df)[-2::]: #Loop over the two slicing label columns 
            df["s_id"] = df[q] #Define which label column to use for slicing
    
            for j in np.unique(df["s_id"]): #Loop over the unique labels in the slicing column 
        
                df[q+str(j)] = 0

                test = df[df["s_id"]==j] #Define test data 
                df.loc[df["temp_id"].isin(np.array(test["temp_id"])),q+str(j)] = 1

                temp_id = [] #Create empty neighbourhood index
                
                if disc_neighbourhood=="queen":
                    for h in test.index: #Fill neighborhood index using first degree neighbors of test data
                        temp_id = np.unique(np.concatenate([temp_id,w.neighbors[h]]).ravel().astype(np.int32))
                
                elif disc_neighbourhood=="queen_2nd":
                    for h in test.index: #Fill neighborhood index using first degree neighbors of test data
                        temp_id = np.unique(np.concatenate([temp_id,w.neighbors[h]]).ravel().astype(np.int32))
                    for u in temp_id: #Include second degree neighbors
                        temp_id = np.unique(np.concatenate([temp_id,w.neighbors[u]]).ravel().astype(np.int32))
                
                elif disc_neighbourhood=="king":
                    for h in test.index: #Fill neighborhood index using first degree neighbors of test data
                        temp_id = np.unique(np.concatenate([temp_id,w.neighbors[h]]).ravel().astype(np.int32))
                
                else:
                    print("Invalid neighbourhood definition for discrete data")

                train = df[df["s_id"]!=j] #Define train data 
                train = train.drop(temp_id,errors="ignore") #Exclude neighbors from index
                df.loc[df["temp_id"].isin(np.array(train["temp_id"])),q+str(j)] = 2
    
    #Drop helper columns
    df = df.drop(columns=["x_group","y_group","s_id","temp_id"])
    # Rename columns
    col_names = []
    for i in [i+1 for i in range(slices*2)]:
        col_names += ["fold_" + str(i)]
    df.rename(columns=dict(zip(df.filter(regex='_group').columns,col_names)), inplace=True)
    #Return dataframe
    return df, col_names 