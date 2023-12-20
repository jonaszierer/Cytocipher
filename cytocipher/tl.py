#### For cluster scoring ####
from .score_and_merge.cluster_score import get_markers, \
                                  code_enrich, coexpr_enrich, giotto_page_enrich

#### For cluster merging ####
from .score_and_merge.cluster_merge import merge_clusters, merge

#### For cluster classification ####
from .score_and_merge.cluster_classifier import merge_clusters_by_classifier, pairwise_classifiers