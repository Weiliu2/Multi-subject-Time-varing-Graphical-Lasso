import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

class Visualizer():
    def __init__(self, precision_matrices: np.ndarray, gname: list[str]):
        self.gname = gname
        self.precision_matrices = precision_matrices
        if precision_matrices.ndim == 2:
            self.pcorrs = self.precision_to_partial_corr([precision_matrices])
        elif precision_matrices.ndim == 3:
            self.pcorrs = self.precision_to_partial_corr(precision_matrices)
        else:
            raise ValueError("Precision_matrices must be a NdArray with shape (T,p,p) or (p,p).")

    def precision_to_partial_corr(self, thetas: list[np.ndarray] | np.ndarray) -> list[np.ndarray]:
        """
        Transfer precision matrices to partial correlation matrices.

        Parameters
        ----------
        thetas : list[np.ndarray] with shape (p,p) and length T or np.ndarray with shape (T,p,p)."""
        pcorrs = []
        for theta in thetas:
            d = np.sqrt(np.diag(theta))
            pcorr = -theta / np.outer(d, d)
            np.fill_diagonal(pcorr, 1.0)
            pcorrs.append(pcorr)
        return pcorrs
    
    def partial_corr_to_difference(self, pcorrs: list[np.ndarray]) -> list[np.ndarray]: 
        """
        Compute difference between adjacent partial correlation matrices.

        Parameters
        ----------
        pcorrs : list[np.ndarray]
            Partial correlation matrices
        
        Return
        ------
        d_pcorrs : list[np.ndarray]
            A list of difference between adjacent partial correlation matrices.
        """
        d_pcorrs = [b - a for a, b in zip(pcorrs, pcorrs[1:])] 
        return d_pcorrs

    def symmetric_log_transform(self, x: np.ndarray, thr: float=0.01) -> np.ndarray:
        """
        Log-transformation to increase contrast of heatmap blocks.
        
        Parameters
        ----------
        x : array like
        thr : float
            Correction factor to magnify or reduce elements of x, defaulted to 0.01
        """
        sign = np.sign(x)
        return sign * np.log10(1 + np.abs(x)/thr)

    def subject_by_module(self):
        """Acquire all the subjects in each module/cluster."""
        gname = np.array(self.gname)
        subject_by_module_list = []
        unique_modules = np.unique(self.modules)
        for m in unique_modules:
            mask = np.where(self.modules == m)
            subject_by_module_list.append(gname[mask].tolist())
        return subject_by_module_list

    def top_k_pcorrs(self, pcorr: np.ndarray, k: int, t: int | str, sign: str="positive") -> pd.DataFrame:
        """
        Give the top-k largest partial correlations of negative or positive values
        in the strictly lower triangular part of a precision matrix.

        Parameters
        ----------
        pcorr : np.ndArray
            Partial correlation matrix
        k : int 
            Rank size
        t : int | str
            Current time label
        sign : str
            Take value between "positive" and "negative"

        Return
        -------
        A DataFrame with 4 columns

        "pair" : name of the subject pair
        "pcorr" : value of partial correlation of this pair
        "rank" : the rank this pair locates at
        "time" : the time stamp this pair belongs to
        """
        # row column indices of strict triangular part: rows, cols: ndarray, dim=1
        rows, cols = np.tril_indices(pcorr.shape[0], k=-1)
        # values of strict triangular part: vals: ndarray, dim=1
        vals = pcorr[rows, cols]
        # sort by descending value: np.argsort sorts ascendingly and returns indices.
        if sign == "positive":
            order = np.argsort(vals)[::-1] # [::-1]: take the entire sequence and reverse it. General slice form sequence[start : stop : step].
        elif sign == "negative":
            order = np.argsort(vals)
        rows_sorted = rows[order][:k]
        cols_sorted = cols[order][:k]
        vals_sorted = vals[order][:k]

        gname = np.array(self.gname)
        pairs = np.char.add(np.char.add(gname[rows_sorted], "-"), gname[cols_sorted])
        ranks = np.array([i for i in range(1, k+1)])
        ts = k * [t]

        df = pd.DataFrame({"pair": pairs,
                           "pcorr": vals_sorted,
                           "rank": ranks,
                           "time": ts})
        return df
    
    # def top_k_precs(self, theta: np.ndarray, k: int, t: int) -> pd.DataFrame:
    #     """
    #     Returns (rows, cols, values) of the top-k largest absolute precision matrix entries 
    #     in the strictly lower triangular part of precision matrix.
    #     """
    #     # row column indices of strict triangular part: rows, cols: ndarray, dim=1
    #     rows, cols = np.tril_indices(theta.shape[0], k=-1)
    #     # values of strict triangular part: vals: ndarray, dim=1
    #     vals = theta[rows, cols]
    #     # sort by descending absolute value: np.argsort sorts ascendingly and returns indices.
    #     order = np.argsort(np.abs(vals))[::-1] # [::-1]: take the entire sequence and reverse it. General slice form sequence[start : stop : step].
    #     rows_sorted = rows[order][:k]
    #     cols_sorted = cols[order][:k]
    #     vals_sorted = vals[order][:k]

    #     gname = np.array(self.gname)
    #     pairs = np.char.add(np.char.add(gname[rows_sorted], "-"), gname[cols_sorted])
    #     ranks = np.array([i for i in range(1, k+1)])
    #     ts = t * np.ones(k)

    #     df = pd.DataFrame({"pair": pairs,
    #                        "pcorr": vals_sorted,
    #                        "rank": ranks,
    #                        "time": ts})
    #     return df
    
    def riverplot(self, k: int, ts: list, min_times: int=5, source: str='line'): 
        """
        Use subject pairs that partial correlations get into the top-k rank at least min_times across all time stamps.
           Plot the river plot of the rank changes of those gene pairs.
        
        Parameters
        ----------
        k : int
            Rank size
        ts : list
            The list of time labels
        min_time : int 
            The least total count of a pair appears in the top-k rank of partial correlations, defaulted to 5.
        source : str
            The style of the riverplot, which is chosen among "line"(line graph), "pa"(Alluvial diagram) and "sankey"(Sankey graph), defaulted to "line".
        """
        if source == 'line':
            raw_rank_dfs = pd.concat([self.top_k_pcorrs(pcorr, k, t) for pcorr, t in zip(self.pcorrs, ts)])
            # Filter to pairs that persist across time
            valid_pairs = (raw_rank_dfs.groupby("pair")["time"].nunique().loc[lambda x: x >= min_times].index)
            rank_dfs = raw_rank_dfs[raw_rank_dfs["pair"].isin(valid_pairs)]
            # Convert ranks into vertical positions: Lower rank = higher on plot, so invert
            rank_dfs["y"] = k - rank_dfs["rank"]

            fig, ax = plt.subplots(figsize=(12, 6))

            time_to_x = {t: i for i, t in enumerate(ts)}
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
            ax.set_xticks(range(len(ts)))
            ax.set_xticklabels(ts)
            ax.set_ylabel("Rank (higher = stronger partial correlation)")
            ax.set_title("Partial Correlation Rank Dynamics")

            ax.invert_yaxis()
            plt.tight_layout()
            plt.show()
        elif source == 'pa':
            import pylluvial as pa
            raw_rank_dfs = pd.concat([self.top_k_pcorrs(pcorr, k, t) for pcorr, t in zip(self.pcorrs, ts)])
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
            raw_rank_dfs = pd.concat([self.top_k_pcorrs(pcorr, k, t) for pcorr, t in zip(self.pcorrs, ts)])
            times = sorted(raw_rank_dfs["time"].unique())
            pairs_in_consecutive = set() # Initialize an empty set to store persistent gene pairs
            # collect all pairs that appear in at least one consecutive time transition
            for t0, t1 in zip(times[:-1], times[1:]):
                p0 = set(raw_rank_dfs[raw_rank_dfs["time"] == t0]["pair"]) # set of gene pair names belonging to t0
                p1 = set(raw_rank_dfs[raw_rank_dfs["time"] == t1]["pair"])
                pairs_in_consecutive |= (p0 & p1) # |= union accumulate, & intersection
            df = raw_rank_dfs[raw_rank_dfs["pair"].isin(pairs_in_consecutive)] 
            # Create Sankey nodes: Nodes = unique (time, rank) combinations
            node_df = (
                df[["time", "rank"]] # Only time and rank matter for Sankey nodes
                .drop_duplicates() # in case there are same (time, rank) combinations but actually no
                .sort_values(["time", "rank"])
                .reset_index(drop=True)
            )
            node_df["node_id"] = range(len(node_df))
            # Create a dictionary with mapping (time, rank) → node_id
            node_lookup = {
                (row.time, row.rank): row.node_id
                for row in node_df.itertuples() # .itertuples() iterates over DataFrame rows as namedtuples
            }
            # Initialize link containers
            sources = []
            targets = []
            values = []
            labels = []
            # Build links
            for pair, g in df.groupby("pair"): # pair is the gene pair name and g is the grouped dataframe of this pair
                g = g.sort_values("time")

                for (_, r0), (_, r1) in zip(g.iloc[:-1].iterrows(), g.iloc[1:].iterrows()): # .iterrows() returns the row as a Series which is r0 or r1
                    src = node_lookup[(r0.time, r0["rank"])] # node_id of start point of the link
                    tgt = node_lookup[(r1.time, r1["rank"])] # node_id of end point of the link

                    sources.append(src)
                    targets.append(tgt)

                    # thickness: use absolute partial correlation
                    values.append(abs(r1["pcorr"]))

                    labels.append(pair)
            values = np.array(values)
            # values = 1 + 9 * (values - values.min()) / (values.max() - values.min()) # normalize thickness to avoid extremely small or large numbers

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
                title="Partial Correlation Rank Dynamics",
                font_size=10,
                height=600,
            )
            fig.show()

    def heatmap(self, pcorr: np.ndarray, title: str='Heatmap', style: str='difference', figsize=(12,12), show_boundary: bool=True, show_clusters: bool=True):
        """
        Perform hierarchical clustering of subjects based on a distance matrix calculated from a partial correlation matrix. 
        Plot heatmap of the network based on the clustering result. 

        Parameters
        ---------
        pcorr : np.ndarray
            Partial correlation matrix
        label_names : list[str]
            Name of all subjects
        title : str
            Title of the heatmap
        style : str
            The method of calculating distance from partial correlation, which is chosen between "difference" and "original", defaulted to ""difference. 
            "difference" uses the difference between two partial correlation matrix and "original" use a partial correlation matrix directly. 
        figsize : Tuple[int, int]
            The size of the heatmap
        show_boundary : bool
            Whether to show the boundaries between clusters on the heatmap. 
            Only works when show_clusters is True.
        show_clusters : bool
            Whether to show dendrogram on the heatmap. 
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        from sklearn.metrics import silhouette_score
        from matplotlib.colors import TwoSlopeNorm

        if style == 'original':
            adjacency = pcorr # 1 diagonal, lies in [-1,1]
            dissimilarity = 1 - adjacency
        elif style == 'difference':
            adjacency = pcorr / 2 # 0 diagonal, lies in [-1,1]
            dissimilarity = 1 - adjacency - np.eye(pcorr.shape[0]) # subtracting a diagonal matrix to make dissimilarity of the same region 0
        else:
            raise ValueError("Choose \'style\' from \'original\' and \'difference\'.")
        d = squareform(dissimilarity)
        Z = linkage(d, method="average") 

        # choose best clustering threshold with solhouette score
        best_k = 2
        best_score = -1
        for k in range(2, min(20, pcorr.shape[0])):
            labels = fcluster(Z, k, criterion='maxclust')
            if len(set(labels)) > 1:
                score = silhouette_score(dissimilarity, labels.reshape(-1,1), metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_k = k
        modules = fcluster(Z, best_k, criterion='maxclust')
        self.modules = modules

        # set color of each cluster
        unique_modules = np.unique(modules)
        palette = list(mcolors.TABLEAU_COLORS.values()) 
        module_colors = {m: palette[i % len(palette)] for i, m in enumerate(unique_modules)}
        row_colors = [module_colors[m] for m in modules]

        # log-transform adjacency to increase contrast of heatmap blocks
        A_plot = self.symmetric_log_transform(adjacency)
        # adjust color of blocks in heatmap for easier comparison
        vmax = max(abs(A_plot.min()), abs(A_plot.max()))
        vmin = -vmax

        # plot heatmap
        if show_clusters == True:
            cg = sns.clustermap(A_plot, 
                                row_linkage=Z, 
                                col_linkage=Z, 
                                row_colors=row_colors,
                                cmap='RdBu_r',
                                norm=TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax),
                                xticklabels=self.gname,
                                yticklabels=self.gname,
                                figsize=figsize)         
            cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xticklabels(), 
                                rotation=90, ha='right', fontsize=10)
            cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_yticklabels(), 
                                        rotation=0, fontsize=10)
            cg.figure.suptitle(title, fontsize=16, y=1.02)

            # highlight boundaries between clusters
            if show_boundary == True:
                order = cg.dendrogram_row.reordered_ind
                ordered_modules = modules[order]
                boundaries = np.where(np.diff(ordered_modules) != 0)[0]
                for b in boundaries:
                    cg.ax_heatmap.axhline(b + 0.5, color="black", linewidth=1.5, alpha=0.8)
                    cg.ax_heatmap.axvline(b + 0.5, color="black", linewidth=1.5, alpha=0.8)
            plt.show()
        else:
            plt.figure(figsize=figsize)
            cg = sns.heatmap(A_plot,
                             cmap='RdBu_r',
                             center=0,
                             vmin=-vmax, 
                             vmax=vmax,
                             xticklabels=self.gname,
                             yticklabels=self.gname,
                             cbar_kws={'label': 'Value', 'shrink': 0.8})
            plt.xticks(rotation=90, ha='right', fontsize=10)
            plt.yticks(rotation=0, fontsize=10)
            plt.title(title, fontsize=16, pad=20)            
            plt.tight_layout()
            plt.show()

    def plot_cell_network(self, adj_matrix: np.ndarray, cell_labels: list | None=None, title: str="Cell-Cell Interaction Network", 
                            figsize : tuple[float, float]=(10, 10), node_size: float=300, 
                            max_edge_width: float=5.0, min_edge_width: float=0.5, arrow_style: str='->', 
                            font_size: float=10, remove_isolates: bool=True, top_edges: int | None=None, 
                            show_edge_colorbar: bool=True):
        """
        Plot cell-cell interaction network using a communication matrix.

        Parameters
        ----------

        adj_matrix : np.ndarray
            Communication matrix of cells. A 2-dim matrix whose cols and rows correspond to a same group of cells with the same order.
            Positive values represent excitatory/activating interactions, negative values inhibitory/suppressing interactions.
        cell_labels : list | None
            (Optional) The label of cells, defaulted to None. 
        title : str
            Title of the figure.
        figsize : tuple
            The size of the figure, defaulted to (10, 10).
        node_size : float
            Scaling factor for the size of the node, defaulted to 300.
        max_edge_width : float
            Maximal width of the edge (applied to the maximum absolute weight).
        min_edge_width : float
            Minimal width of the edge (applied to the minimum absolute weight).
        arrow_style : str
            Style of the arrows in the edge, including '->', '-|>', '-[', '-', 
            'simple', 'fancy'. head_width and head_length defines arrowhead width and length. 
            The complete control sentence is given by '->,head_width=0.3,head_length=0.5' for example.
        font_size : int
            Font size of the node label.
        remove_isolates : bool
            Whether to remove isolated nodes or not.
        top_edges : int | None
            (Optional) Keep the strongest edges (based on absolute weight) and hide others.
        show_edge_colorbar : bool
            Whether to show color bar of the weight of the edges (positive/negative diverging colors).
        """
        import networkx as nx
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.patches import Circle as MPLCircle
        n_nodes = adj_matrix.shape[0]
        
        # Create a directed graph and add edges (include both positive and negative weights)
        G = nx.DiGraph()
        for i in range(n_nodes):
            for j in range(n_nodes):
                weight = adj_matrix[i, j]
                if weight != 0:                     # include both positive and negative
                    G.add_edge(i, j, weight=weight)
        
        # Remove isolated nodes (nodes with no incoming or outgoing edges)
        if remove_isolates:
            isolated_nodes = [n for n in G.nodes if G.degree(n) == 0]
            G.remove_nodes_from(isolated_nodes)
            # Update mapping: The new node sequence is the remaining nodes in the graph
            original_nodes = sorted(G.nodes)
            # Re-number from 0 to k-1 to facilitate subsequent processing
            node_remap = {orig: new for new, orig in enumerate(original_nodes)}
            G = nx.relabel_nodes(G, node_remap, copy=True)
            n_nodes = len(G.nodes)
            if n_nodes == 0:
                print("Warning: There are 0 nodes in the graph.")
                return plt.subplots(figsize=figsize)

        # Limit the number of edges by only showing the top strongest edges (based on absolute weight)
        if top_edges is not None and top_edges > 0:
            edges = list(G.edges(data=True))
            edges.sort(key=lambda x: abs(x[2]['weight']), reverse=True)
            top_edges_list = edges[:top_edges]
            # Create a new graph containing the kept nodes 
            H = nx.DiGraph()
            H.add_nodes_from(G.nodes(data=True))
            for u, v, d in top_edges_list:
                H.add_edge(u, v, weight=d['weight'])
            G = H

        pos = nx.circular_layout(G) # Circular node layout

        # Define node color
        node_colors = plt.cm.tab20(np.linspace(0, 1, n_nodes))

        # Create a figure and its axes
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        ax.axis('off')

        # Draw edges
        edges = G.edges(data=True)
        if edges:
            # Extract raw weights and absolute weights
            raw_weights = [data['weight'] for (_, _, data) in edges]
            abs_weights = [abs(w) for w in raw_weights]
            min_abs = min(abs_weights)
            max_abs = max(abs_weights)
            min_raw = min(raw_weights)
            max_raw = max(raw_weights)
            
            # Color mapping: diverging colormap (negative -> cool, positive -> warm)
            norm_color = plt.Normalize(vmin=min_raw, vmax=max_raw)
            cmap = plt.get_cmap('coolwarm')    # strong contrast: blue (negative) ↔ red (positive)
            
            # Edge width mapping: based on absolute weight
            for u, v, data in edges:
                w_raw = data['weight']
                w_abs = abs(w_raw)
                # Compute width
                if max_abs == min_abs:
                    width = (min_edge_width + max_edge_width) / 2
                else:
                    width = min_edge_width + (w_abs - min_abs) / (max_abs - min_abs) * (max_edge_width - min_edge_width)
                # Compute color (based on raw signed weight)
                color = cmap(norm_color(w_raw))
                # Get node position 
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                # Add arrow
                arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                        arrowstyle=arrow_style,
                                        color=color,
                                        linewidth=width,
                                        connectionstyle="arc3,rad=0.2",
                                        shrinkA=10, shrinkB=10,
                                        zorder=1)
                ax.add_patch(arrow)
            
            # Show edge color bar (showing signed weights)
            if show_edge_colorbar:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_color)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, shrink=0.7, aspect=20, pad=0.05)
                cbar.set_label('The change of partial correlations (negative: blue, positive: red)', fontsize=10)

        # Set node and label
        nx.draw_networkx_nodes(G, pos,
                            node_color=node_colors,
                            node_size=node_size,
                            edgecolors='grey',
                            linewidths=1.5,
                            ax=ax)
        # Add labels
        labels = {}
        for node in G.nodes():
            if cell_labels is not None:
                orig_node = original_nodes[node]
                labels[node] = cell_labels[orig_node]
            else:
                labels[node] = str(node)
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size, ax=ax)

        # Create a outer circle connecting all the nodes together 
        radii = [np.sqrt(x**2 + y**2) for x, y in pos.values()]
        avg_radius = np.mean(radii) if radii else 0.9
        circle = MPLCircle((0, 0), avg_radius + 0.1, fill=False, edgecolor='gray', linewidth=1, linestyle='--', zorder=0)
        ax.add_patch(circle)

        ax.set_title(title, fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()

    def temporal_deviation_ratio(self, thetas_seq: list[list[np.ndarray]] | list[np.ndarray] | np.ndarray, ts: list, title: str="Temporal deviation ratio"):
        """
        Plot the TD Ratio. 
        The input precision matrices are either from a bootstrap sample set
        or the original sample set.

        Parameters
        ----------
        thetas_seq : list[list[np.ndarray]] or list[np.ndarray]
            precision matrices from bootstrapped result with "shape" (B,T,p,p) or
            prevision matrices from original observations with "shape" (T,p,p)
        ts : list
            The list of time labels
        title : str
            The title of the plot
        """
        if isinstance(thetas_seq[0], list):
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
            ages = np.array([str(t) for t in ts])
            age_shifts = np.char.add(np.char.add(ages[:-1], "->"), ages[1:])
            plt.figure(figsize=(10, 5))
            plt.plot(age_shifts, TD_ratios_mean, label="Mean")
            # plt.axhline(y=1, linestyle='--', linewidth=1, color='red')
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
            plt.xlabel("Time stamp shifts")
            plt.ylabel("TD Ratios")
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.show()
        elif isinstance(thetas_seq[0], np.ndarray):
            total_TD_ratio = 0
            cur_TD_ratios = []
            for i in range(len(thetas_seq)-1):
                cur_TD_ratio = np.linalg.norm(thetas_seq[i+1] - thetas_seq[i], 'fro')
                total_TD_ratio += cur_TD_ratio
                cur_TD_ratios.append(cur_TD_ratio)
            TD_ratio = [_ / total_TD_ratio for _ in cur_TD_ratios]
            TD_ratio_mean = np.mean(TD_ratio)

            ages = np.array([str(t) for t in ts])
            age_shifts = np.char.add(np.char.add(ages[:-1], "->"), ages[1:])
            plt.figure(figsize=(10, 5))
            plt.plot(age_shifts, TD_ratio, label="Mean")
            plt.axhline(y=TD_ratio_mean, linestyle='--', linewidth=1, color='red')
            plt.xlabel("Time stamp shifts")
            plt.ylabel("TD Ratios")
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.show()
        
if __name__ == "__main__":
    import gseapy as gp

    precision_matrices = np.load("precision_matrices_CD4T_Glycolysis_all_samples_0.2_5.npy", allow_pickle=True)
    # gname = pd.read_csv("PBMC_metabolic_gene.csv", sep=",").columns.tolist()
    gname = ["ACSS1","ACSS2","ADH1A","ADH1B","ADH1C","ADH4","ADH5","ADH6","ADH7","ADPGK","AKR1A1",
                "ALDH1B1","ALDH2","ALDH3A1","ALDH3A2","ALDH3B1","ALDH3B2","ALDH7A1","ALDH9A1","ALDOA",
                "ALDOB","ALDOC","BPGM","DLAT","DLD","ENO1","ENO2","ENO3","ENO4","FBP1",
                "FBP2","G6PC1","G6PC2","G6PC3","GALM","GAPDH","GAPDHS","GCK","GPI","HK1","HK2","HK3",
                "HKDC1","LDHA","LDHAL6A","LDHAL6B","LDHB","LDHC","MINPP1","PCK1","PCK2","PDHA1","PDHA2",
                "PDHB","PFKL","PFKM","PFKP","PGAM1","PGAM2","PGAM4","PGK1","PGK2","PGM1","PGM2","PKLR","PKM","TPI1"]

    model = Visualizer(precision_matrices, gname, k=len(gname))
    pcorrs = model.pcorrs
    """Sankey plot"""
    model.riverplot(min_times=10, source="sankey")

    # pcorr = pcorrs[0]

    # """Dendrogram clustering and enrichment analysis"""
    # model.heatmap(pcorr, style='Dendrogram')
    # gmlist = model.subject_by_module()
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
    
    # """Leiden network community identification and enrichment analysis"""
    # model.heatmap(pcorr, style='Leiden')
    # gmlist = model.subject_by_module()
    # for i, _ in enumerate(gmlist):
    #     print(i, len(_))

    # gene_list = gmlist[1]
    # enr = gp.enrichr(
    #     gene_list=gene_list,
    #     gene_sets=["KEGG_2021_Human", "GO_Biological_Process_2021", "Reactome_2022", "MSigDB_Hallmark_2020"],
    #     organism="Human",
    #     background=gname,
    #     cutoff=0.05,
    #     outdir=None
    # )
    # # print(enr.results.head())
    # gp.dotplot(enr.results, 
    #            x='Gene_set',
    #            cutoff=0.1,                # cutoff filters enriched terms by adjusted p-value (FDR)
    #            column="Adjusted P-value", 
    #            xticklabels_rot=45, # rotate xtick labels
    #            show_ring=True, # set to False to revmove outer ring
    #            marker='o',
    #            ofname="dot_CD4T_age0_leiden")
    
    # gp.barplot(enr.results, 
    #            cutoff=0.1,
    #            column="Adjusted P-value",
    #            ofname="bar_CD4T_age0_leiden")