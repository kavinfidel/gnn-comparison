#
# Copyright (C)  2020  University of Pisa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import argparse
import os
import logging
import torch
from datasets import *
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils.convert import to_networkx
import pickle


def rewire_Graph(data):
    """Original bridge-based rewiring function"""
    import networkx as nx
    from torch_geometric.utils import to_networkx
    
    logging.info("Starting bridge-based rewiring...")
    g6 = to_networkx(data, to_undirected=True)
    
    bridges = list(nx.bridges(g6))
    if not bridges:
        logging.info("No bridges found in the graph.")
        edge_index = torch.tensor(list(g6.edges), dtype=torch.long).t().contiguous()
        return edge_index

    filtered_bridges = []
    for u, v in bridges:
        if len(list(nx.neighbors(g6, u))) > 1 and len(list(nx.neighbors(g6, v))) > 1:
            filtered_bridges.append((u, v))

    if not filtered_bridges:
        logging.info("No suitable bridges found for rewiring.")
        edge_index = torch.tensor(list(g6.edges), dtype=torch.long).t().contiguous()
        return edge_index

    adj_node_dict = {}
    for u, v in filtered_bridges:
        for node in (u, v):
            neighbors = list(nx.neighbors(g6, node))
            if len(neighbors) > 1:
                adj_node_dict[node] = [n for n in neighbors if n not in (u, v)]

    for u, v in filtered_bridges:
        for nu in adj_node_dict.get(u, []):
            if nu != v and not g6.has_edge(nu, v):
                g6.add_edge(nu, v)
        for nv in adj_node_dict.get(v, []):
            if nv != u and not g6.has_edge(nv, u):
                g6.add_edge(nv, u)

    edge_index = torch.tensor(list(g6.edges), dtype=torch.long).t().contiguous()
    logging.info("Bridge rewiring complete. New edge count: %d", edge_index.size(1))
    return edge_index

logging.basicConfig(level = logging.INFO, format ='%(asctime)s - %(levelname)s - %(message)s' )
DATASETS = {
    'REDDIT-BINARY': RedditBinary,
    'REDDIT-MULTI-5K': Reddit5K,
    'COLLAB': Collab,
    'IMDB-BINARY': IMDBBinary,
    'IMDB-MULTI': IMDBMulti,
    'NCI1': NCI1,
    'ENZYMES': Enzymes,
    'PROTEINS': Proteins,
    'DD': DD
}

def find_local_bridges(G):
    """
    Find local bridges - edges whose removal increases shortest path length.
    Improved version for unweighted graphs.
    """
    import networkx as nx
    
    def is_local_bridge(u, v):
        """Check if edge (u,v) is a local bridge."""
        # Remove edge temporarily
        G.remove_edge(u, v)
        
        # Check connectivity and path length
        try:
            new_shortest_path = nx.shortest_path_length(G, u, v)
            # For unweighted graphs, any increase from 1 is significant
            is_bridge = new_shortest_path > 1
        except nx.NetworkXNoPath:
            # If no path exists after removal, it's definitely a bridge
            is_bridge = True
        
        # Restore edge
        G.add_edge(u, v)
        
        return is_bridge
    
    local_bridges = []
    
    # Check each edge
    for u, v in list(G.edges()):
        if is_local_bridge(u, v):
            local_bridges.append((u, v))
    
    return local_bridges
    
def rewire_Graph_local_bridges(data, top_n=None):
    """
    Rewire graph by connecting neighbors through local bridges.
    
    Args:
        data: PyTorch Geometric data object
        top_n: Number of local bridges to rewire (None for all)
    
    Returns:
        torch.Tensor: New edge index with rewired connections
    """
    import networkx as nx
    from torch_geometric.utils import to_networkx
    
    logging.info("Starting local bridges-based rewiring...")
    
    # Convert to NetworkX for analysis
    g = to_networkx(data, to_undirected=True)
    original_edge_count = g.number_of_edges()
    logging.info("Original edge count: %d", original_edge_count)
    
    # Find local bridges
    local_bridges = find_local_bridges(g)
    logging.info("Local bridges found: %d", len(local_bridges))
    
    # Select subset if top_n is specified
    if top_n is not None and len(local_bridges) > top_n:
        # You could add scoring here (e.g., by edge betweenness)
        # For now, just take the first top_n
        local_bridges = local_bridges[:top_n]
        logging.info("Using top %d local bridges for rewiring", top_n)
    
    if not local_bridges:
        logging.warning("No local bridges found. Returning original graph.")
        edge_index = torch.tensor(list(g.edges), dtype=torch.long).t().contiguous()
        return edge_index
    
    logging.info("Local bridges to rewire: %s", local_bridges)
    
    # Create adjacency dictionary for neighbors (excluding the local bridge endpoints)
    adj_node_dict = {}
    for u, v in local_bridges:
        for node in (u, v):
            neighbors = list(nx.neighbors(g, node))
            if len(neighbors) > 1:
                adj_node_dict[node] = [n for n in neighbors if n not in (u, v)]
    
    # Add rewiring edges: connect neighbors of local bridge endpoints
    new_edges_added = 0
    for u, v in local_bridges:
        # Connect neighbors of u to v
        for nu in adj_node_dict.get(u, []):
            if nu != v and not g.has_edge(nu, v):
                g.add_edge(nu, v)
                new_edges_added += 1
        
        # Connect neighbors of v to u  
        for nv in adj_node_dict.get(v, []):
            if nv != u and not g.has_edge(nv, u):
                g.add_edge(nv, u)
                new_edges_added += 1
    
    # Convert back to edge index format
    edge_index = torch.tensor(list(g.edges), dtype=torch.long).t().contiguous()
    logging.info("Local bridges rewiring complete. New edge count: %d, Added: %d", 
                edge_index.size(1), new_edges_added)
    
    return edge_index

def rewire_Graph_betweenness(data, top_n=2):
    """
    Rewire graph by connecting neighbors through high betweenness centrality edges.
    
    Args:
        data: PyTorch Geometric data object
        top_n: Number of highest betweenness edges to rewire
    
    Returns:
        torch.Tensor: New edge index with rewired connections
    """
    import networkx as nx
    from torch_geometric.utils import to_networkx
    
    logging.info("Starting betweenness-based rewiring...")
    
    # Convert to NetworkX for analysis
    g = to_networkx(data, to_undirected=True)
    original_edge_count = g.number_of_edges()
    logging.info("Original edge count: %d", original_edge_count)
    
    # Calculate edge betweenness centrality
    edge_betweenness = nx.edge_betweenness_centrality(g)
    
    # Get top_n edges with highest betweenness centrality
    sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
    high_betweenness_edges = [edge for edge, _ in sorted_edges[:top_n]]
    
    logging.info("High betweenness edges identified: %s", high_betweenness_edges)
    
    # Create adjacency dictionary for neighbors (excluding the high betweenness edge endpoints)
    adj_node_dict = {}
    for u, v in high_betweenness_edges:
        for node in (u, v):
            neighbors = list(nx.neighbors(g, node))
            if len(neighbors) > 1:
                adj_node_dict[node] = [n for n in neighbors if n not in (u, v)]
    
    # Add rewiring edges: connect neighbors of high betweenness edge endpoints
    for u, v in high_betweenness_edges:
        # Connect neighbors of u to v
        for nu in adj_node_dict.get(u, []):
            if nu != v and not g.has_edge(nu, v):
                g.add_edge(nu, v)
        
        # Connect neighbors of v to u  
        for nv in adj_node_dict.get(v, []):
            if nv != u and not g.has_edge(nv, u):
                g.add_edge(nv, u)
    
    # Convert back to edge index format
    edge_index = torch.tensor(list(g.edges), dtype=torch.long).t().contiguous()
    logging.info("Betweenness rewiring complete. New edge count: %d", edge_index.size(1))
    
    return edge_index

# Update the argument parser to include new rewiring strategies

def apply_rewiring_strategy(data, strategy='bridges', top_n=2):
    """Apply different rewiring strategies"""
    if strategy == 'bridges':
        return rewire_Graph(data)
    elif strategy == 'betweenness':
        return rewire_Graph_betweenness(data, top_n=top_n)
    elif strategy == 'local_bridges':
        return rewire_Graph_local_bridges(data, top_n=top_n)
    else:
        logging.warning("Unknown rewiring strategy: %s. Using original graph.", strategy)
        return data.edge_index

def get_args_dict():
    parser = argparse.ArgumentParser()

    parser.add_argument('DATA_DIR',
                        help='where to save the datasets')
    parser.add_argument('--dataset-name', dest='dataset_name',
                        default='all', help='dataset name [Default: \'all\']')
    parser.add_argument('--outer-k', dest='outer_k', type=int,
                        default=10, help='evaluation folds [Default: 10]')
    parser.add_argument('--inner-k', dest='inner_k', type=int,
                        default=None, help='model selection folds [Default: None]')
    parser.add_argument('--use-one', action='store_true',
                        default=False, help='use 1 as feature')
    parser.add_argument('--use-degree', dest='use_node_degree', action='store_true',
                        default=False, help='use degree as feature')
    parser.add_argument('--no-kron', dest='precompute_kron_indices', action='store_false',
                        default=True, help='don\'t precompute kron reductions')
    parser.add_argument('--use-rewired', action = 'store_true', 
                        default = False, help = 'Add rewired edges to the dataset.')
    parser.add_argument('--rewiring-strategy', type=str, default='bridges', 
                    choices=['bridges', 'betweenness', 'local_bridges'],
                    help='Rewiring strategy to use: bridges (default), betweenness, or local_bridges')
    parser.add_argument('--top-n-edges', type=int, default=2,
                    help='Number of top edges to rewire (for betweenness and local_bridges strategies)')


    return vars(parser.parse_args())

def preprocess_dataset(dataset_path, dataset_name, use_rewired=False, rewiring_strategy='bridges', top_n=2):
    """
    Preprocess the dataset and optionally add rewired edges.
    Makes sure:
     The data object has both the edge_index(unaltered) and rewired_edge_index
     To change the rewiring(eg: rewire1 --> betweenness) only make changes to graph.py
    """
    logging.info(f"Preprocessing started for {dataset_name}")
    
    # Use TUDataset directly like the original working code
    dataset = TUDataset(root=dataset_path, name=dataset_name)

    for i, data in enumerate(dataset):
        logging.info(f"Processing graph {i + 1}/{len(dataset)} in dataset {dataset_name}")
        
        if use_rewired:
            # Convert data to nx graph
            graph = to_networkx(data, to_undirected=True)
            
            # Apply the selected rewiring strategy
            if rewiring_strategy == 'bridges':
                data.edge_index = rewire_Graph(data)
            elif rewiring_strategy == 'betweenness':
                data.edge_index = rewire_Graph_betweenness(data, top_n=top_n)
            elif rewiring_strategy == 'local_bridges':
                data.edge_index = rewire_Graph_local_bridges(data, top_n=top_n)
            else:
                logging.warning(f"Unknown rewiring strategy: {rewiring_strategy}. Using bridges.")
                data.edge_index = rewire_Graph(data)
            
            logging.info(f"Original edges: {data.edge_index.size(1)} | Strategy: {rewiring_strategy}")

    # Save the dataset with a different name if rewired
    save_name = f"{dataset_name}_rewired&original_preprocessed.pt" if use_rewired else f"{dataset_name}_processed.pt"
    torch.save(dataset, os.path.join(dataset_path, save_name))
    print(f"Dataset {dataset_name} processed & saved as {save_name} in {dataset_path}.")

if __name__ == "__main__":
    
    args_dict = get_args_dict()
    print(args_dict)

    dataset_name = args_dict['dataset_name']
    dataset_path = args_dict['DATA_DIR']
    use_rewired = args_dict['use_rewired']
    rewiring_strategy = args_dict.get('rewiring_strategy', 'bridges')  # Use get() for safety
    top_n = args_dict.get('top_n_edges', 2)  # Use get() for safety

    if dataset_name == 'all':
        for name in DATASETS:
            preprocess_dataset(dataset_path, name, use_rewired=use_rewired, 
                             rewiring_strategy=rewiring_strategy, top_n=top_n)
    else:
        preprocess_dataset(dataset_path, dataset_name, use_rewired=use_rewired,
                         rewiring_strategy=rewiring_strategy, top_n=top_n)

    # parser = argparse.ArgumentParser(description="Preprocess datasets.")
    # parser.add_argument("dataset_path", type=str, help="Path to the dataset folder.")
    # parser.add_argument("--dataset-name", type=str, required=True, help="Name of the dataset.")
    # parser.add_argument("--use-rewired", action="store_true", help="Add rewired edges to the dataset.")
    # args = parser.parse_args()
    
    # preprocess_dataset(args.dataset_path, args.dataset_name, use_rewired=args.use_rewired)

# use it like: python PrepareDatasets.py DATA/CHEMICAL --dataset-name NCI1 --use-rewired


# ~ python PrepareDatasets.py DATA/CHEMICAL --dataset-name PROTEINS --outer-k 10 --use-rewired

