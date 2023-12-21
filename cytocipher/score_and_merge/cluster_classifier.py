import pandas as pd
import numpy as np
from itertools import combinations
import scanpy as sc   
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .cluster_score import get_markers
from .cluster_merge import merge_neighbours_v2

pd.options.mode.chained_assignment = None
#from plotnine import ggplot, aes, geom_point, geom_line, geom_tile, scale_color_discrete, theme_classic

################################################################################
             # Functions related to  Classifier #
################################################################################
def get_curves(y_test, y_prob, mod):
    res=[] 
    for i, cell_type in enumerate(mod.classes_):
        fpr, tpr, thresholds = roc_curve(y_test == cell_type, y_prob[:, i], drop_intermediate=False)
        precision, recall, thresholds = precision_recall_curve(y_test == cell_type, y_prob[:, i], drop_intermediate=False)
        curve = pd.DataFrame({'cluster': cell_type, 'fpr': fpr, 'tpr': tpr, 'prec': precision, 'recall': recall})
        res.append(curve)
    res = pd.concat(res)
    return(res)

def get_metrics(y_prob, y_test, mod):
    """ calculates prediction metrics

    Parameters
    ----------
    y_prob: predicted class probability
    y_test: vector of class labels
    mod: model
    """
    res=[]
    for i, cell_type in enumerate(mod.classes_):
        roc = roc_auc_score(y_test == cell_type, y_prob[:, i])
        pr  = precision_recall_curve(y_test == cell_type, y_prob[:, i])
        prc = auc(pr[1], pr[0])
        res.append(pd.DataFrame({'cluster': cell_type, 'auroc':roc, 'auprc': prc}, index=[0]))
    res = pd.concat(res)
    res = res.reset_index(drop=True)
    return(res)

def get_pred(X, group):
    """ trains predictor

    Parameters
    ----------
    X: matrix of predictors
    group: vector of class labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, group, test_size=0.5, random_state=0)
    pipe = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=100, solver="sag", penalty="l2", C=0.02))
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)
    return(y_prob, y_test, pipe)


def pairwise_classifiers(data,
                         groupby: str,
                         markers_key:str = None,
                         calc_curves: bool = False):
    """ trains classifiers for each pair of clusters

    Parameters
    ----------
    data: sc.AnnData
        Single cell RNA-seq anndata, QC'd a preprocessed to log-cpm in data.X
    groupby: str
        Specifies the clusters to merge, defined in data.obs[groupby]. Must
        be categorical type.
    """
    
    ## init
    if markers_key is None:
        markers_key = f"{groupby}_markers"
    all_res=[]
    all_models={}
    all_curves={}
    ## iterate over pairs
    pairs = list(combinations(data.obs[groupby].cat.categories, 2))
    for pair in pairs:
        cells1 = data.obs[groupby] == pair[0]
        cells2 = data.obs[groupby] == pair[1]
        cells  = cells1 | cells2
        ## get cluster markers
        markers1 = data.uns[markers_key][pair[0]].tolist()
        markers2 = data.uns[markers_key][pair[1]].tolist()
        markers  = markers1 + markers2
        # subset data
        dat = data.X[ cells ][ :  ,  data.var_names.isin(markers)]
        lab = data.obs[groupby][cells]
        ## predict
        y_prob, y_test, mod = get_pred(dat, lab)
        res = get_metrics(y_prob, y_test, mod)
        res['other_cluster'] = pair[::-1] if pair[0] == res.loc[0, 'cluster'] else pair
        all_res.append(res)
        all_models[f"{pair[0]}__{pair[1]}"] = mod
        ## curves
        if calc_curves:
            all_curves[f"{pair[0]}__{pair[1]}"] = get_curves(y_test, y_prob, mod)
    
    # format and return
    all_res = pd.concat(all_res).reset_index(drop=True)
    # make cluster pairs unique
    all_res['c1'] = np.where(all_res['cluster'] < all_res['other_cluster'], all_res['cluster'], all_res['other_cluster'])
    all_res['c2'] = np.where(all_res['other_cluster'] == all_res['c1'], all_res['cluster'], all_res['other_cluster'])
    all_res = all_res[ ['c1', 'c2', 'auroc', 'auprc'] ]
    # keep only worst model per pair
    all_res = all_res.groupby(['c1', 'c2']).min()
    all_res = all_res.reset_index()
    data.uns[f'{groupby}_classifier_metrics'] = all_res
    data.uns[f'{groupby}_classifier_models' ] = all_models
    if calc_curves:
        data.uns[f'{groupby}_classifier_curves' ] = all_curves




def oneagainstall_classifiers(data,
                              groupby: str,
                              markers_key:str = None):
    """ trains classifiers for each cluster against all others. Only used to
        populate the obsm for heatmap plotting

    Parameters
    ----------
    data: sc.AnnData
        Single cell RNA-seq anndata, QC'd a preprocessed to log-cpm in data.X
    groupby: str
        Specifies the clusters to merge, defined in data.obs[groupby]. Must
        be categorical type.
    """
    
    ## init
    if markers_key is None:
        markers_key = f"{groupby}_markers"
    
    ## get cluster markers
    markers = [value for sublist in data.uns[markers_key].values() for value in sublist]

    ## train & predict
    _, _, mod = get_pred(data.X[ :  ,  data.var_names.isin(markers)], data.obs[groupby])
    ## predict for all and save
    y_prob = mod.predict_proba(data.X[ :  ,  data.var_names.isin(markers)])
    data.obsm[f'{groupby}_predictions'] = pd.DataFrame(y_prob, index=data.obs_names, columns=mod.classes_).reindex(columns=data.obs[groupby].cat.categories.tolist())
    


def plot_curves(data, groupby, 
                figsize=(20,15),
                which_curve:str='roc',
                colour_map:str="Set1"):
    
    # prep            
    clusters = data.obs[groupby].cat.categories
    fig, axs = plt.subplots(len(clusters), len(clusters)-1, figsize=figsize)
    
    # iterate over cluster pairs
    for i1,c1 in enumerate(clusters):
        for i2,c2 in enumerate(clusters):
            
            ## x labels on top
            if i1==0:
                axs[i1,i2-1].set_title(c2, fontsize=8)
            
            ## plot curve
            n = f"{c1}__{c2}"
            if n in data.uns[f'{groupby}_classifier_curves'].keys():
                dat = data.uns[f'{groupby}_classifier_curves'][n]
                cols = plt.colormaps[colour_map]
                for c, clust in enumerate(np.unique(dat.cluster)):
                    subdat = dat[dat.cluster == clust]
                    if which_curve == 'roc':
                        axs[i1,i2-1].plot(subdat['fpr'], subdat['tpr'], c=cols(c))
                    elif which_curve == "prc":
                        axs[i1,i2-1].plot(subdat['prec'], subdat['recall'], c=cols(c))
            
            elif (i1>0) and (i2>0):
                ## empty everything else
                axs[i1,i2-1].set_xticks([])
                axs[i1,i2-1].set_yticks([])
                axs[i1,i2-1].axis('off')

        ## y labels on right side
        axs[i1,len(clusters)-2].set_ylabel(c1, rotation=270, fontsize=8, labelpad=12)
        axs[i1,len(clusters)-2].yaxis.set_label_position("right")


# The key function #
def merge_clusters_by_classifier(data: sc.AnnData, 
                                 groupby: str,
                                 var_groups: str=None, 
                                 ## markers
                                 marker_method: str="t-test_overestim_var", marker_tie_correct:bool=True,
                                 n_top_genes: int = 6, t_cutoff: int=3,
                                 marker_padj_cutoff: float=.05, min_de: int=1,
                                 gene_order=None,
                                 ## classification
                                 auroc_thresh:float=0.95,  auprc_thresh:float=0.95,
                                 ## other
                                 max_iter: int = 10,
                                 verbose: bool = True):
    """ Merges the clusters following an expectation maximisation approach.

    Parameters
    ----------
    data: sc.AnnData
        Single cell RNA-seq anndata, QC'd a preprocessed to log-cpm in data.X
    groupby: str
        Specifies the clusters to merge, defined in data.obs[groupby]. Must
        be categorical type.
    marker_method: str
        method for marker detection (see scanpy.tl.rank_genes_groups)
    marker_tie_correct: bool
        method for marker detection (see scanpy.tl.rank_genes_groups)    
    var_groups: str
        Specifies a column in data.var of type boolean, with True indicating
        the candidate genes to use when determining marker genes per cluster.
        Useful to, for example, remove ribosomal and mitochondrial genes.
        None indicates use all genes in data.var_names as candidates.
    n_top_genes: int
        The maximimum no. of marker genes per cluster.
    t_cutoff: float
        The minimum t-value a gene must have to be considered a marker gene
        (Welch's t-statistic with one-versus-rest comparison).
    marker_padj_cutoff: float
        Adjusted p-value (Benjamini-Hochberg correction) below which a gene
        can be considered a marker gene.
    min_de: int
        Minimum no. of marker genes to use for each cluster.
    verbose: bool
        Print statements during computation (True) or silent run (False).
    Returns
    --------
        data.obs[f'{groupby}_merged']
            New cell type labels with non-sigificant clusters merged.
        data.uns[f'{groupby}_merged_markers']
            Dictionary with merged cluster names as keys, and list of marker
            genes as values.
        data.obsm[f'{groupby}_merged_enrich_scores']
            Dataframe with cells as rows and merged clusters as columns.
            Values are the enrichment scores for each cluster, using the
            marker genes in data.uns[f'{groupby}_merged_markers']
    """

    ### Initialize ##
    if verbose:
        print( f"Initialize", flush=True)
    data.obs[f'{groupby}_merged'] = data.obs[groupby]

    ## Merging per iteration until convergence ##
    for i in range(max_iter):
        if verbose:
            print( f"iteration {i}", flush=True)
            print( f"-- markers", flush=True)
        # Running marker gene determination #
        get_markers(data, f'{groupby}_merged', 
                    method=marker_method, tie_correct=marker_tie_correct,
                    n_top=n_top_genes,
                    verbose=False, var_groups=var_groups, t_cutoff=t_cutoff,
                    padj_cutoff=marker_padj_cutoff,
                    gene_order=gene_order, min_de=min_de)

        # Running the enrichment scoring #
        if verbose:
            print( f"-- classifier", flush=True)
        pairwise_classifiers(data = data, groupby=f'{groupby}_merged')
        
       
        ## get all_resormance 
        all_res = data.uns[f'{groupby}_merged_classifier_metrics'].copy()
        merge = all_res[ (all_res.auroc < auroc_thresh) | (all_res.auprc < auprc_thresh)]
        ## format
        merge['merge'] = merge.apply(lambda row: (row['c1'], row['c2']), axis=1)
        merge_pairs = merge['merge'].tolist()
        # Running new merge operation #
        if len(merge_pairs) > 0:
            if verbose:
                print( f"-- merge", flush=True)
            #### Now merging..
            labels = data.obs[f'{groupby}_merged'].values.astype(str)
            cluster_map, merge_cluster_labels = merge_neighbours_v2(labels, merge_pairs)
            if verbose:
                print( f"-- [before: {len(np.unique(labels))}] [after: {len(np.unique(merge_cluster_labels))}]", flush=True)
            data.obs[f'{groupby}_merged'] = merge_cluster_labels
            data.obs[f'{groupby}_merged'] = data.obs[f'{groupby}_merged'].astype('category')
            data.obs[f'{groupby}_merged'] = data.obs[f'{groupby}_merged'].cat.set_categories(np.unique(merge_cluster_labels.astype(int)).astype(str))
            #### Recording the merge pairs
            data.uns[f'{groupby}_mutualpairs'] = merge_pairs
        else:   
            print(f"Added data.obs[f'{groupby}_merged']")
            ## final classifier with curves
            pairwise_classifiers(data, groupby = f'{groupby}_merged', calc_curves = True)
            oneagainstall_classifiers(data, groupby = f'{groupby}_merged')
            print("Exiting due to convergence.")
            return

    ## final classifier with curves
    pairwise_classifiers(data, groupby = f'{groupby}_merged', calc_curves = True)
    oneagainstall_classifiers(data, groupby = f'{groupby}_merged')
        
    if verbose:
        print(f"Added data.obs[f'{groupby}_merged']")
        print(f"Exiting due to reaching max_iter {max_iter}")

