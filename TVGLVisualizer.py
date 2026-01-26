import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

class Visualizer():
    def __init__(self, precision_matrices: np.ndarray, gname: list[str], k: int):
        self.k = k
        self.ts = [0, 1, 2, 6, 12, 18, 22, 26, 27, 29, 30, 40, 50, 60, 70, 80, 90]
        self.gname = gname
        if precision_matrices.ndim == 2:
            self.pcorrs = self.precision_to_partial_corr([precision_matrices])
        elif precision_matrices.ndim == 3:
            self.pcorrs = self.precision_to_partial_corr(precision_matrices)
        elif precision_matrices.ndim == 4:
            self.temporal_deviation_ratio(precision_matrices)            

    def precision_to_partial_corr(self, thetas):
        pcorrs = []
        for theta in thetas:
            d = np.sqrt(np.diag(theta))
            pcorr = -theta / np.outer(d, d)
            np.fill_diagonal(pcorr, 1.0)
            pcorrs.append(pcorr)
        return pcorrs
    
    def gene_by_module(self):
        gname = np.array(self.gname)
        gene_by_module_list = []
        unique_modules = np.unique(self.modules)
        for m in unique_modules:
            mask = np.where(self.modules == m)
            gene_by_module_list.append(gname[mask].tolist())
        return gene_by_module_list

    def top_k_pcorrs(self, pcorr: np.ndarray, k: int, t: int) -> pd.DataFrame:
        """
        Returns (rows, cols, values) of the top-k largest absolute partial correlations
        in the strictly lower triangular part of precision matrix.
        """
        # row column indices of strict triangular part: rows, cols: ndarray, dim=1
        rows, cols = np.tril_indices(pcorr.shape[0], k=-1)
        # values of strict triangular part: vals: ndarray, dim=1
        vals = pcorr[rows, cols]
        # sort by descending absolute value: np.argsort sorts ascendingly and returns indices.
        order = np.argsort(np.abs(vals))[::-1] # [::-1]: take the entire sequence and reverse it. General slice form sequence[start : stop : step].
        rows_sorted = rows[order][:k]
        cols_sorted = cols[order][:k]
        vals_sorted = vals[order][:k]

        gname = np.array(self.gname)
        pairs = np.char.add(np.char.add(gname[rows_sorted], "-"), gname[cols_sorted])
        ranks = np.array([i for i in range(1, k+1)])
        ts = t * np.ones(k)

        df = pd.DataFrame({"pair": pairs,
                           "pcorr": vals_sorted,
                           "rank": ranks,
                           "time": ts})
        return df
    
    def riverplot(self, min_times: int=5, source: str='line'): 
        """Use gene pairs that partial correlations get into the top-k rank at least min_times across all time stamps.
           Plot the river plot of the rank changes of those gene pairs."""
        if source == 'line':
            raw_rank_dfs = pd.concat([self.top_k_pcorrs(pcorr, self.k, t) for pcorr, t in zip(self.pcorrs, self.ts)])
            # Filter to pairs that persist across time
            valid_pairs = (raw_rank_dfs.groupby("pair")["time"].nunique().loc[lambda x: x >= min_times].index)
            rank_dfs = raw_rank_dfs[raw_rank_dfs["pair"].isin(valid_pairs)]
            # Convert ranks into vertical positions: Lower rank = higher on plot, so invert
            rank_dfs["y"] = self.k - rank_dfs["rank"]

            fig, ax = plt.subplots(figsize=(12, 6))

            time_to_x = {t: i for i, t in enumerate(self.ts)}
            pairs = []
            for pair, g in rank_dfs.groupby("pair"):
                g = g.sort_values("time")
                x = g["time"].map(time_to_x)
                y = g["y"]
                pairs.append(pair)
                ax.plot(
                    x,
                    y,
                    alpha=0.3,
                    linewidth=1
                )
            ax.legend(labels=pairs)
            ax.set_xticks(range(len(self.ts)))
            ax.set_xticklabels(self.ts)
            ax.set_ylabel("Rank (higher = stronger partial correlation)")
            ax.set_title("Gene-Gene Partial Correlation Rank Dynamics")

            ax.invert_yaxis()
            plt.tight_layout()
            plt.show()
        elif source == 'pa':
            import pylluvial as pa
            raw_rank_dfs = pd.concat([self.top_k_pcorrs(pcorr, self.k, t) for pcorr, t in zip(self.pcorrs, self.ts)])
            # Filter to pairs that persist across time
            valid_pairs = (raw_rank_dfs.groupby("pair")["time"].nunique().loc[lambda x: x >= min_times].index)
            rank_dfs = raw_rank_dfs[raw_rank_dfs["pair"].isin(valid_pairs)]
            flows = rank_dfs[["time", "pair", "rank"]]
            fig, ax = plt.subplots(figsize=(10, 5))
            pa.alluvial(
                x="time",
                stratum="rank",
                alluvium="pair",
                data=flows,
                ax=ax,
                show_labels=True
            )
            plt.show()
        elif source == 'sankey':
            import plotly.graph_objects as go
            raw_rank_dfs = pd.concat([self.top_k_pcorrs(pcorr, self.k, t) for pcorr, t in zip(self.pcorrs, self.ts)])
            times = sorted(raw_rank_dfs["time"].unique())
            pairs_in_consecutive = set()
            for t0, t1 in zip(times[:-1], times[1:]):
                p0 = set(raw_rank_dfs[raw_rank_dfs["time"] == t0]["pair"])
                p1 = set(raw_rank_dfs[raw_rank_dfs["time"] == t1]["pair"])
                pairs_in_consecutive |= (p0 & p1)
            df = raw_rank_dfs[raw_rank_dfs["pair"].isin(pairs_in_consecutive)]

            node_df = (
                df[["time", "rank"]]
                .drop_duplicates()
                .sort_values(["time", "rank"])
                .reset_index(drop=True)
            )
            node_df["node_id"] = range(len(node_df))
            node_lookup = {
                (row.time, row.rank): row.node_id
                for row in node_df.itertuples()
            }
            sources = []
            targets = []
            values = []
            labels = []

            for pair, g in df.groupby("pair"):
                g = g.sort_values("time")

                for (_, r0), (_, r1) in zip(g.iloc[:-1].iterrows(), g.iloc[1:].iterrows()):
                    src = node_lookup[(r0.time, r0["rank"])]
                    tgt = node_lookup[(r1.time, r1["rank"])]

                    sources.append(src)
                    targets.append(tgt)

                    # thickness: use absolute partial correlation
                    values.append(abs(r1["pcorr"]))

                    labels.append(pair)
            values = np.array(values)
            values = 1 + 9 * (values - values.min()) / (values.max() - values.min())

            fig = go.Figure(
                data=[
                    go.Sankey(
                        arrangement="fixed",
                        node=dict(
                            pad=8,
                            thickness=10,
                            label=[
                                f"{row.time} | rank {row.rank}"
                                for row in node_df.itertuples()
                            ],
                            color="lightgray",
                        ),
                        link=dict(
                            source=sources,
                            target=targets,
                            value=values,
                            label=labels,
                            color="rgba(50, 100, 200, 0.35)"
                        ),
                    )
                ]
            )
            fig.update_layout(
                title="Gene-Gene Partial Correlation Rank Dynamics",
                font_size=10,
                height=600,
            )
            fig.show()

    def heatmap(self, pcorr: np.ndarray, beta: int=1, style: str='Dendrogram'):
        adjacency = np.abs(pcorr) ** beta 
        if style == 'Dendrogram':
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            dissimilarity = 1 - adjacency
            d = squareform(dissimilarity) 
            Z = linkage(d, method="average") 
            modules = fcluster(Z, t=0.996, criterion="distance") 
            self.modules = modules

            unique_modules = np.unique(modules)
            palette = list(mcolors.TABLEAU_COLORS.values()) 
            module_colors = {m: palette[i % len(palette)] for i, m in enumerate(unique_modules)}

            row_colors = [module_colors[m] for m in modules]
            # facilitate showing clusters
            A = adjacency.copy()
            thr = np.percentile(A[A > 0], 95)
            A_plot = np.where(A >= thr, A, 0)
            np.fill_diagonal(A_plot, 0)
            A_plot = np.log10(A_plot + 1e-8)
            # get gene order
            cg = sns.clustermap(A_plot, row_linkage=Z, col_linkage=Z, row_colors=row_colors)
            # order = cg.dendrogram_row.reordered_ind
            # ordered_modules = modules[order]

            # # find boundaries
            # boundaries = np.where(np.diff(ordered_modules) != 0)[0]

            # for b in boundaries:
            #     plt.axhline(b, color="black", linewidth=0.5)
            #     plt.axvline(b, color="black", linewidth=0.5)
            plt.show()
        elif style == 'Leiden':
            import igraph as ig
            import leidenalg
            g = ig.Graph.Weighted_Adjacency(adjacency.tolist(), mode="undirected")
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                n_iterations=-1
            )
            modules = np.array(partition.membership)
            self.modules = modules
            unique_modules = np.unique(modules)

            order = []
            for m in unique_modules:
                idx = np.where(modules == m)[0]
                # sort genes inside module by connectivity
                strength = adjacency[np.ix_(idx, idx)].sum(axis=1)
                idx = idx[np.argsort(-strength)]
                order.extend(idx)

            order = np.array(order)
            A = adjacency.copy()
            thr = np.percentile(A[A > 0], 95)
            A_plot = np.where(A >= thr, A, 0)
            np.fill_diagonal(A_plot, 0)
            A_plot = np.log10(A_plot + 1e-8)
            adj_ord = A_plot[np.ix_(order, order)]
    
            sns.heatmap(
                adj_ord,
                cmap="RdBu_r",
                center=0,
                xticklabels=False,
                yticklabels=False
            )
            plt.show()

        # module level network
        K = len(np.unique(modules))
        M = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                M[i, j] = A_plot[np.ix_(modules == i+1, modules == j+1)].mean() # mix module i+1 and j+1 and calculate the mean
        sns.heatmap(M, cmap="viridis")
        plt.title('Module level heatmap')
        plt.show()

    def temporal_deviation_ratio(self, thetas_seq):
        TD_ratio_seq = []
        for thetas in thetas_seq:
            TD_ratio = []
            total_TD_ratio = 0
            cur_TD_ratios = []
            for i in range(len(thetas)-1):
                cur_TD_ratio = np.linalg.norm(thetas[i+1] - thetas[i], 'fro')
                total_TD_ratio += cur_TD_ratio
                cur_TD_ratios.append(cur_TD_ratio)
            for i in range(len(cur_TD_ratios)):
                TD_ratio.append(cur_TD_ratios[i] / total_TD_ratio * (len(thetas)-1))
            TD_ratio_seq.append(TD_ratio)
        TD_ratio_seq = np.array(TD_ratio_seq)
        TD_ratios_mean = np.mean(TD_ratio_seq, axis=0)
        TD_ratios_std = np.std(TD_ratio_seq, axis=0) 

        # plot
        ages = np.array([str(t) for t in self.ts])
        age_shifts = np.char.add(np.char.add(ages[:-1], "-"), ages[1:])
        plt.figure(figsize=(10, 5))
        plt.plot(age_shifts, TD_ratios_mean, label="Mean")
        plt.axhline(y=1, linestyle='--', linewidth=1, color='red')
        plt.fill_between(
            age_shifts,
            TD_ratios_mean - TD_ratios_std,
            TD_ratios_mean + TD_ratios_std,
            alpha=0.3,
            label="±1 Std Dev"
        )
        plt.errorbar(
            age_shifts,
            TD_ratios_mean,
            yerr=TD_ratios_std,
            fmt='-o',
            capsize=3
        )
        plt.xlabel("Age shifts")
        plt.ylabel("TD Ratios")
        plt.title("Temporal deviation ratios")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
if __name__ == "__main__":
    import gseapy as gp

    precision_matrices = np.load("precision_matrices_CD4T_all_samples.npy", allow_pickle=True)
    gname = pd.read_csv("PBMC_metabolic_gene.csv", sep=",").columns.tolist()

    model = Visualizer(precision_matrices, gname, k=100)
    pcorrs = model.pcorrs
    """Sankey plot"""
    # model.riverplot(source="sankey")

    pcorr = pcorrs[0]

    """Dendrogram clustering and enrichment analysis"""
    # model.heatmap(pcorr, style='Dendrogram')
    # gmlist = model.gene_by_module()
    # count = 0; loc = 0
    # for i, _ in enumerate(gmlist):
    #     if len(_) > count:
    #         count = len(_)
    #         loc = i
    #     print(i, len(_))
    # print(i, count)

    # # list of gene symbols of the biggest module
    # gene_list2 = gmlist[i] 

    # enr = gp.enrichr(
    #     gene_list=gene_list2,
    #     gene_sets=["KEGG_2021_Human", "GO_Biological_Process_2021", "Reactome_2022", "MSigDB_Hallmark_2020"],
    #     organism="Human",
    #     background=gname,
    #     cutoff=0.05,
    #     outdir=None
    # )
    # # print(enr.results.head())
    # gp.dotplot(enr.results, 
    #            x='Gene_set',
    #            cutoff=0.05,                # cutoff filters enriched terms by adjusted p-value (FDR)
    #            column="Adjusted P-value", 
    #            xticklabels_rot=45, # rotate xtick labels
    #            show_ring=True, # set to False to revmove outer ring
    #            marker='o',
    #            ofname="dot_CD4T_age0_dendrogram")
    
    # gp.barplot(enr.results, 
    #            cutoff=0.05,
    #            column="Adjusted P-value",
    #            ofname="bar_CD4T_age0_dendrogram")
    
    """Leiden network community identification and enrichment analysis"""
    model.heatmap(pcorr, style='Leiden')
    gmlist = model.gene_by_module()
    for i, _ in enumerate(gmlist):
        print(i, len(_))

    gene_list = gmlist[1]
    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=["KEGG_2021_Human", "GO_Biological_Process_2021", "Reactome_2022", "MSigDB_Hallmark_2020"],
        organism="Human",
        background=gname,
        cutoff=0.05,
        outdir=None
    )
    # print(enr.results.head())
    gp.dotplot(enr.results, 
               x='Gene_set',
               cutoff=0.1,                # cutoff filters enriched terms by adjusted p-value (FDR)
               column="Adjusted P-value", 
               xticklabels_rot=45, # rotate xtick labels
               show_ring=True, # set to False to revmove outer ring
               marker='o',
               ofname="dot_CD4T_age0_leiden")
    
    gp.barplot(enr.results, 
               cutoff=0.1,
               column="Adjusted P-value",
               ofname="bar_CD4T_age0_leiden")