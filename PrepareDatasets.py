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


def rewire_graph(g: nx.Graph) -> torch.Tensor:   # REWIRE 1
    logging.info("Edge Rewiring starting for graph with %d nodes and %d edges", g.number_of_nodes(), g.number_of_edges())

    g6 = g.copy()
    bridges = list(nx.bridges(g6))
    adj_node_dict = {}

    filtered_bridges = [bridge for bridge in bridges
                        if g6.degree[bridge[0]] > 1 and g6.degree[bridge[1]] > 1]

    logging.info("Filtered down to %d bridges after degree check", len(filtered_bridges))

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
    logging.info("Rewiring complete. New edge count: %d", edge_index.size(1))
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

    return vars(parser.parse_args())


def preprocess_dataset(dataset_path, dataset_name, use_rewired=False):
    """
    Preprocess the dataset and optionally add rewired edges.
    Makes sure:
     The data object has both the edge_index(unaltered) an rewired_edge_index
     To change the rewiring(eg: rewire1 --> betweenes) only make changes to graph.py

    """

   # dataset_dir = os.path.join(dataset_path, dataset_name)
    logging.info(f"Preprocessing started for {dataset_name}")
    # if dataset_name in DATASETS:
    #     dataset_class = DATASETS[dataset_name]

    #     if dataset_name == 'ENZYMES':
    #         dataset = dataset_class(root=dataset_path, use_node_attr=True)
    #         logging.info(f"Additional 18 node attributes used for ENZYMES")
    #     else:
    #         dataset = dataset_class(root=dataset_path)
    #else:
    dataset = TUDataset(root=dataset_path, name=dataset_name)

    for i, data in enumerate(dataset):
        logging.info(f"Processing graph {i + 1}/{len(dataset)} in dataset {dataset_name}")
        # convert data to nx graph
        # pass into rewire function
        if use_rewired:
            graph = to_networkx(data, to_undirected=True)
        # Use the method from graph.py to get both original and rewired edge indices
            data.rewired_edge_index = rewire_graph(graph).long()
        # data.edge_index IS THE ORIGNAL (UNALTERED) Edge index
            logging.info(f"Original edges: {data.edge_index.size(1)} | Rewired edges: {data.rewired_edge_index.size(1)}")

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

    if dataset_name == 'all':
        for name in DATASETS:
            preprocess_dataset(dataset_path, name, use_rewired=use_rewired)
    else:
        preprocess_dataset(dataset_path, dataset_name, use_rewired=use_rewired)

    # parser = argparse.ArgumentParser(description="Preprocess datasets.")
    # parser.add_argument("dataset_path", type=str, help="Path to the dataset folder.")
    # parser.add_argument("--dataset-name", type=str, required=True, help="Name of the dataset.")
    # parser.add_argument("--use-rewired", action="store_true", help="Add rewired edges to the dataset.")
    # args = parser.parse_args()
    
    # preprocess_dataset(args.dataset_path, args.dataset_name, use_rewired=args.use_rewired)

# use it like: python PrepareDatasets.py DATA/CHEMICAL --dataset-name NCI1 --use-rewired


# ~ python PrepareDatasets.py DATA/CHEMICAL --dataset-name PROTEINS --outer-k 10 --use-rewired

