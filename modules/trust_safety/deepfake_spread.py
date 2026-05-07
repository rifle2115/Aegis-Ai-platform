import streamlit as st
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph
import json
import matplotlib.pyplot as plt
import numpy as np

# --- Utility: safe rerun across Streamlit versions ---
def safe_rerun():
    """
    Re-run the Streamlit script in a way compatible with multiple Streamlit versions.
    """
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        # last resort: stop current run (UI will update on next interaction)
        st.stop()


# --- Page Configuration ---


# --- Helper Functions ---

# 1. Load the Pre-Calculated Analysis (Fast)
@st.cache_data
def load_analysis_data():
    # Load Centralities
    df_centralities = pd.read_csv("data/trust_safety/deepfake_spread/centralities.csv")
    df_centralities['node'] = df_centralities['node'].astype(str)  # Ensure String IDs

    # Load Full Graph
    with open("data/trust_safety/deepfake_spread/graph_with_communities.json", 'r') as f:
        graph_data = json.load(f)

    G_raw = json_graph.node_link_graph(graph_data)
    # Force all nodes to be strings for consistency and preserve attributes
    G = nx.DiGraph()
    for n, attr in G_raw.nodes(data=True):
        G.add_node(str(n), **{k: v for k, v in attr.items()})
    for u, v, attr in G_raw.edges(data=True):
        G.add_edge(str(u), str(v), **{k: v for k, v in attr.items()})

    # Re-attach community attributes from the loaded graph data (robustness)
    node_attrs = {str(n['id']): n for n in graph_data.get('nodes', [])}
    for node, data in node_attrs.items():
        if 'community' in data:
            G.nodes[node]['community'] = data['community']

    return df_centralities, G

# 2. Load the Raw Time-Series Data (For the "Live Map")
@st.cache_data
def load_spread_data():
    df_spread = pd.read_csv("data/trust_safety/deepfake_spread/spread_dataset_with_time.csv")
    df_spread['source'] = df_spread['source'].astype(str)
    df_spread['target'] = df_spread['target'].astype(str)
    # ensure 'time' exists and is numeric
    if 'time' in df_spread.columns:
        df_spread['time'] = pd.to_numeric(df_spread['time'], errors='coerce').fillna(0).astype(int)
    return df_spread

# 3. The Plotting Function (Reused across pages)
def plot_network(G, title, node_colors=None, highlighted_nodes=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Layout calculation (seed ensures it looks the same every time)
    pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42)

    # Default colors if none provided
    if node_colors is None:
        node_colors = '#1f78b4'  # Default blue

    # Draw the graph
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, cmap=plt.cm.viridis, alpha=0.8, ax=ax)

    # Highlight specific nodes (e.g., "Patient Zero" or "Banned User")
    if highlighted_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=highlighted_nodes, node_size=150, node_color='red', ax=ax)

    ax.set_title(title, fontsize=15)
    plt.axis("off")
    return fig

# -------- Future Predictor helper: compute metrics (cached) ----------
@st.cache_data(show_spinner=False)
def compute_basic_metrics(_G):
    """
    Compute degree, approximate betweenness, reachability, and bridge score.
    _G is intentionally prefixed with _ so Streamlit won't try to hash it.
    Returns: deg_dict, betw, reach, bridge
    """
    G = _G  # local alias

    # degree (undirected degree for reach)
    deg_dict = dict(G.degree())
    n = G.number_of_nodes()

    # approximate betweenness (sample if graph large)
    G_und = G.to_undirected()
    k_sample = None
    if n > 400:
        k_sample = min(200, max(50, n // 6))
    try:
        betw = nx.betweenness_centrality(G_und, k=k_sample, seed=42, normalized=True)
    except Exception:
        betw = nx.betweenness_centrality(G_und, normalized=True)

    # descendants / reachability: directed -> descendants, undirected -> 2-hop neighbors
    reach = {}
    if G.is_directed():
        for node in G.nodes():
            try:
                reach[node] = len(nx.descendants(G, node))
            except Exception:
                reach[node] = len(list(nx.single_source_shortest_path_length(G.to_undirected(), node, cutoff=2))) - 1
    else:
        for node in G.nodes():
            reach[node] = len(list(nx.single_source_shortest_path_length(G, node, cutoff=2))) - 1

    # bridge score: neighbors in other communities (if community attr exists)
    bridge = {}
    has_community = any('community' in G.nodes[n] for n in G.nodes())
    if has_community:
        for node in G.nodes():
            comm = G.nodes[node].get('community', None)
            other_count = 0
            for nbr in G.neighbors(node):
                if G.nodes[nbr].get('community', None) != comm:
                    other_count += 1
            bridge[node] = other_count
    else:
        # fallback proxy: unique second-hop neighbors minus direct neighbors
        for node in G.nodes():
            second_hop = set()
            for nbr in G.neighbors(node):
                second_hop.update(set(G.neighbors(nbr)))
            bridge[node] = max(0, len(second_hop) - len(list(G.neighbors(node))))

    return deg_dict, betw, reach, bridge

# --- MAIN APP LOGIC ---

# Load Data
try:
    df_centralities, G_full = load_analysis_data()
    df_spread = load_spread_data()
except Exception as e:
    st.error(f"Data Loading Error: {e}. Please run 'analyze_network.py' first.")
    st.stop()

# --- Initialize Session State ---
if 'page' not in st.session_state:
    st.session_state['page'] = "1. Global Overview"

# --- Sidebar Navigation (buttons) ---
st.sidebar.title("🔍 Navigation")

page_options = [
    "1. Global Overview",
    "2. Spread Simulator (Live Map)",
    "3. Community Deep-Dive",
    "4. Resilience Lab",
    "5. Future Predictor"
]

# Use buttons for navigation; when clicked, set session state and rerun safely
for option in page_options:
    if st.sidebar.button(option, key=f"nav_{option}", use_container_width=True):
        st.session_state['page'] = option
        # Clear any auto_predict flag if present
        if 'auto_predict' in st.session_state:
            del st.session_state['auto_predict']
        safe_rerun()

page = st.session_state['page']

st.sidebar.divider()
st.sidebar.info("Project: Deepfake Proliferation Analysis\nTool: NetworkX + Streamlit")

# ==========================================
# PAGE 1: GLOBAL OVERVIEW
# ==========================================
if page == "1. Global Overview":
    st.title("🌍 Global Network Overview")
    st.markdown("### Analysis of the full deepfake propagation network.")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users Infected", G_full.number_of_nodes())
    col2.metric("Total Shares", G_full.number_of_edges())
    # safe access to community column
    if 'community' in df_centralities.columns:
        col3.metric("Communities Detected", int(df_centralities['community'].nunique()))
    else:
        col3.metric("Communities Detected", "N/A")
    col4.metric("Avg. Clustering Coeff", f"{nx.average_clustering(G_full.to_undirected()):.2f}")

    st.divider()

    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Network Map")
        # Color by community
        communities = [G_full.nodes[n].get('community', 0) for n in G_full.nodes()]
        fig = plot_network(G_full, "Full Network Structure (Colored by Community)", node_colors=communities)
        st.pyplot(fig)

    with c2:
        st.subheader("🚨 Top 10 Superspreaders")
        st.caption("Sorted by Betweenness Centrality (Bridge Score)")
        if 'betweenness_centrality' in df_centralities.columns:
            top_10 = df_centralities.head(10)[['node', 'betweenness_centrality', 'community']]
        else:
            top_10 = df_centralities.head(10)[['node', 'community']] if 'community' in df_centralities.columns else df_centralities.head(10)[['node']]
        st.dataframe(top_10, hide_index=True, use_container_width=True)

# ==========================================
# PAGE 2: SPREAD SIMULATOR (LIVE MAP)
# ==========================================
elif page == "2. Spread Simulator (Live Map)":
    st.title("⏳ Spread Simulator")
    st.markdown("### Watch how the deepfake spreads over time.")

    # Slider
    if 'time' in df_spread.columns:
        max_time = int(df_spread['time'].max())
        time_step = st.slider("Time (Hours since Patient Zero)", 0, max_time, 10)
    else:
        st.warning("Spread dataset has no 'time' column.")
        time_step = 0

    # Filter Data based on slider
    current_edges = df_spread[df_spread['time'] <= time_step] if 'time' in df_spread.columns else df_spread

    # Build the "Current" Graph
    G_now = nx.DiGraph()
    G_now.add_edges_from(zip(current_edges['source'], current_edges['target']))

    # Metrics
    st.markdown(f"**Status at Hour {time_step}:**")
    m1, m2 = st.columns(2)
    m1.metric("Infected Users", G_now.number_of_nodes())
    m2.metric("New Infections (Last Hour)", len(df_spread[(df_spread['time'] > time_step-1) & (df_spread['time'] <= time_step)]) if 'time' in df_spread.columns else 0)

    # Plot
    if G_now.number_of_nodes() > 0:
        # We highlight the "Patient Zero" (Source of first edge)
        patient_zero = [df_spread.iloc[0]['source']] if not df_spread.empty else []
        fig = plot_network(G_now, f"Network State at Hour {time_step}", highlighted_nodes=patient_zero)
        st.pyplot(fig)
    else:
        st.warning("No spreads happened yet. Move the slider!")

# ==========================================
# PAGE 3: COMMUNITY DEEP-DIVE
# ==========================================
elif page == "3. Community Deep-Dive":
    st.title("🔎 Community Analysis")
    st.markdown("### Inspect specific 'Echo Chambers' within the network.")

    # Dropdown to choose community
    if 'community' in df_centralities.columns:
        comm_list = sorted(df_centralities['community'].unique())
    else:
        comm_list = [0]
    selected_comm = st.selectbox("Select a Community to Analyze:", comm_list)

    # Filter Graph for this community
    selected_nodes = [n for n, attr in G_full.nodes(data=True) if attr.get('community') == selected_comm]
    G_comm = G_full.subgraph(selected_nodes)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Visualizing Community {selected_comm}")
        fig = plot_network(G_comm, f"Structure of Community {selected_comm}")
        st.pyplot(fig)

    with col2:
        st.subheader("Community Stats")
        st.write(f"**Size:** {len(selected_nodes)} Users")
        st.write(f"**Internal Connections:** {G_comm.number_of_edges()}")

        # Local Influencers
        st.write("**Top Local Influencer:**")
        # Recalculate degree just for this subgraph
        if len(selected_nodes) > 0:
            local_degrees = dict(G_comm.degree())
            top_local = max(local_degrees, key=local_degrees.get)
            st.metric("Local Hub Node", top_local, f"{local_degrees[top_local]} connections")
        else:
            st.write("No nodes in this community.")

# ==========================================
# PAGE 4: RESILIENCE LAB
# ==========================================
elif page == "4. Resilience Lab":
    st.title("🧪 Resilience Lab")
    st.markdown(
        "Simulate the counterfactual: remove a node **and all nodes it infected (descendants)**, "
        "then compare the original and resulting graphs side-by-side."
    )

    # --- Selection UI ---
    st.subheader("Choose node(s)")
    choice_mode = st.radio(
        "Select removal mode:",
        ["Top 1 spreader (by betweenness)", "Top 5 spreaders", "Custom node (pick one)", "Custom top-k"],
        horizontal=True
    )

    def top_nodes(n):
        return [str(x) for x in df_centralities.head(n)['node'].tolist()]

    if choice_mode == "Top 1 spreader (by betweenness)":
        selected_nodes = top_nodes(1)
    elif choice_mode == "Top 5 spreaders":
        selected_nodes = top_nodes(5)
    elif choice_mode == "Custom top-k":
        max_select = min(50, len(df_centralities))
        k = st.slider("k (top-k to remove)", 1, max_select, 5)
        selected_nodes = top_nodes(k)
    else:  # Custom node
        # allow selecting any node from centralities (string)
        node_list = df_centralities['node'].astype(str).tolist()
        selected_single = st.selectbox("Select node to remove:", node_list, index=0)
        selected_nodes = [str(selected_single)]

    with st.expander("Preview selected nodes"):
        preview_df = df_centralities[df_centralities['node'].astype(str).isin(selected_nodes)][['node', 'betweenness_centrality', 'degree_centrality']] if 'betweenness_centrality' in df_centralities.columns else df_centralities[df_centralities['node'].astype(str).isin(selected_nodes)][['node']]
        st.dataframe(preview_df, hide_index=True, use_container_width=True)

    # --- Run button ---
    run_btn = st.button("🔍 Show Before / After")

    # --- If not run, show instruction and snapshot of original ---
    if not run_btn:
        st.info("Click the button to compute and visualize the effect of removing the selected node(s) and all their downstream descendants.")
        st.subheader("Original (snapshot)")
        fig_s, ax_s = plt.subplots(figsize=(8, 6))
        pos_full = nx.spring_layout(G_full, k=0.15, iterations=20, seed=42)
        if any('community' in G_full.nodes[n] for n in G_full.nodes()):
            colors = [G_full.nodes[n].get('community', 0) for n in G_full.nodes()]
            nx.draw_networkx_nodes(G_full, pos_full, node_size=40, node_color=colors, cmap=plt.cm.tab20, alpha=0.8, ax=ax_s)
        else:
            nx.draw_networkx_nodes(G_full, pos_full, node_size=40, node_color='lightblue', alpha=0.8, ax=ax_s)
        nx.draw_networkx_edges(G_full, pos_full, alpha=0.12, ax=ax_s, arrows=True, arrowsize=6)
        ax_s.set_title("Original Network", fontsize=12)
        ax_s.axis('off')
        st.pyplot(fig_s)
        plt.close(fig_s)

    else:
        # --- Build after graph by removing selected nodes + all their descendants ---
        G_before = G_full
        G_after = G_full.copy()

        # compute union of nodes to remove
        nodes_to_remove = set()
        for n in selected_nodes:
            if n in G_before:
                nodes_to_remove.add(n)
                try:
                    desc = nx.descendants(G_before, n)
                    nodes_to_remove.update(desc)
                except Exception:
                    pass
            else:
                st.warning(f"Selected node {n} not found in graph.")

        # Remove nodes
        nodes_present = [n for n in nodes_to_remove if n in G_after.nodes()]
        if nodes_present:
            G_after.remove_nodes_from(nodes_present)
        else:
            st.warning("No nodes were removed.")

        # --- Compute metrics ---
        before_nodes = G_before.number_of_nodes()
        before_edges = G_before.number_of_edges()
        after_nodes = G_after.number_of_nodes()
        after_edges = G_after.number_of_edges()

        reduction_nodes = before_nodes - after_nodes
        reduction_edges = before_edges - after_edges
        reduction_pct = (reduction_nodes / before_nodes * 100) if before_nodes > 0 else 0

        # --- Display metrics ---
        st.subheader("Quick Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nodes (before)", before_nodes)
        c2.metric("Nodes (after)", after_nodes, delta=f"-{reduction_nodes} ({reduction_pct:.1f}%)")
        c3.metric("Edges (before)", before_edges)
        c4.metric("Edges (after)", after_edges, delta=f"-{reduction_edges}")

        # --- Visual comparison ---
        st.subheader("Before vs After")
        col1, col2 = st.columns(2)

        # Compute layout once
        if 'pos_full' not in st.session_state:
            st.session_state['pos_full'] = nx.spring_layout(G_before, k=0.15, iterations=20, seed=42)
        pos = st.session_state['pos_full']

        with col1:
            st.markdown("**🔵 Before**")
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            if any('community' in G_before.nodes[n] for n in G_before.nodes()):
                colors_b = [G_before.nodes[n].get('community', 0) for n in G_before.nodes()]
                nx.draw_networkx_nodes(G_before, pos, node_size=40, node_color=colors_b, cmap=plt.cm.tab20, alpha=0.85, ax=ax1)
            else:
                nx.draw_networkx_nodes(G_before, pos, node_size=40, node_color='lightblue', alpha=0.85, ax=ax1)
            nx.draw_networkx_edges(G_before, pos, alpha=0.12, ax=ax1, arrows=True, arrowsize=6)
            highlight = [n for n in nodes_present if n in G_before.nodes()]
            if highlight:
                nx.draw_networkx_nodes(G_before, pos, nodelist=highlight, node_size=140, node_color='red', alpha=0.9, ax=ax1)
            ax1.set_title(f"Original — {before_nodes} nodes", fontsize=11)
            ax1.axis('off')
            st.pyplot(fig1)
            plt.close(fig1)

        with col2:
            st.markdown("**🔴 After**")
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            pos_after = {n: pos[n] for n in G_after.nodes() if n in pos}
            if any('community' in G_after.nodes[n] for n in G_after.nodes()):
                colors_a = [G_after.nodes[n].get('community', 0) for n in G_after.nodes()]
                nx.draw_networkx_nodes(G_after, pos_after, node_size=40, node_color=colors_a, cmap=plt.cm.tab20, alpha=0.85, ax=ax2)
            else:
                nx.draw_networkx_nodes(G_after, pos_after, node_size=40, node_color='lightblue', alpha=0.85, ax=ax2)
            nx.draw_networkx_edges(G_after, pos_after, alpha=0.12, ax=ax2, arrows=True, arrowsize=6)
            ax2.set_title(f"After removal — {after_nodes} nodes", fontsize=11)
            ax2.axis('off')
            st.pyplot(fig2)
            plt.close(fig2)

# ==========================================
# PAGE 5: FUTURE PREDICTOR (adjusted scoring)
# ==========================================
elif page == "5. Future Predictor":
    st.title("🔮 Future Predictor")
    st.markdown("Predict the next superspreaders using the original graph (G_full). Use the slider to trade off bridge vs adjusted reach.")

    # Use the original graph
    G_used = G_full
    st.caption("Using original graph (G_full)")

    # Inputs
    max_k = max(1, min(200, G_used.number_of_nodes()))
    k = st.number_input("Number of top nodes (k):", min_value=1, max_value=max_k, value=min(10, max_k), step=1, key='ui_k_simple')
    w_betw = st.slider("Betweenness weight (remaining weight goes to adjusted Reach):", 0.0, 1.0, 0.6, key='ui_w_simple')

    # Prediction button
    if st.button("🔮 Run Prediction (original graph)"):
        def run_prediction_and_display(G_used, k, w_betw):
            with st.spinner("Computing metrics and prediction..."):
                deg_dict, betw, reach, bridge = compute_basic_metrics(G_used)

            nodes = list(G_used.nodes())
            df_tmp = pd.DataFrame({
                "node": [str(n) for n in nodes],
                "betweenness": [betw.get(n, 0.0) for n in nodes],
                "reach": [reach.get(n, 0) for n in nodes],
                "degree": [deg_dict.get(n, 0) for n in nodes],
            })

            # Adjusted reach: amplify reach by node degree (log scale) to break ties
            df_tmp['reach_adj'] = df_tmp['reach'] * np.log1p(df_tmp['degree'])

            # Normalization helper (min-max)
            def norm(s):
                lo, hi = s.min(), s.max()
                if hi == lo:
                    return s.apply(lambda _: 0.0)
                return (s - lo) / (hi - lo)

            df_tmp['betw_n'] = norm(df_tmp['betweenness'])
            df_tmp['reach_adj_n'] = norm(df_tmp['reach_adj'])

            # Final score (weighted)
            df_tmp['score'] = w_betw * df_tmp['betw_n'] + (1 - w_betw) * df_tmp['reach_adj_n']
            df_tmp = df_tmp.sort_values('score', ascending=False).reset_index(drop=True)
            topk = df_tmp.head(int(k))

            # Estimated reach (simple sum of raw reach of top-k)
            estimated_reach = int(topk['reach'].sum())

            # Show results
            st.subheader("🎯 Prediction Results")
            st.metric("Estimated reachable nodes (top-k combined)", f"{estimated_reach:,}")

            st.subheader("Top Predicted Superspreaders")
            display_cols = ['node', 'score', 'betweenness', 'reach', 'degree', 'reach_adj']
            st.dataframe(topk[display_cols].round(4), hide_index=True, use_container_width=True)

            # Visualization: highlight top-k on the network
            st.subheader("Network Visualization (Top-k highlighted in red)")
            fig, ax = plt.subplots(figsize=(10, 8))
            with st.spinner("Computing network layout..."):
                pos = nx.spring_layout(G_used, k=0.15, iterations=20, seed=42)

            all_nodes = list(G_used.nodes())
            top_nodes_set = set(topk['node'].tolist())

            top_nodes_list = [n for n in all_nodes if str(n) in top_nodes_set]
            default_nodes = [n for n in all_nodes if str(n) not in top_nodes_set]

            if default_nodes:
                nx.draw_networkx_nodes(G_used, pos, nodelist=default_nodes,
                                       node_size=40, node_color='lightgray', alpha=0.6, ax=ax)
            if top_nodes_list:
                nx.draw_networkx_nodes(G_used, pos, nodelist=top_nodes_list,
                                       node_size=200, node_color='red', alpha=0.9,
                                       ax=ax, edgecolors='darkred', linewidths=2)
            nx.draw_networkx_edges(G_used, pos, alpha=0.15, ax=ax,
                                   arrows=G_used.is_directed(), arrowsize=8)

            ax.set_title(f"Top {k} Predicted Superspreaders (Highlighted)", fontsize=14, fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)

        run_prediction_and_display(G_used, k, w_betw)

