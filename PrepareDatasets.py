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
from datasets.graph import Graph
from torch_geometric.datasets import TUDataset
from torch_geometric_data import Data

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
                        choices=DATASETS.keys(), default='all', help='dataset name [Default: \'all\']')
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


# def preprocess_dataset(name, args_dict):
#     dataset_class = DATASETS[name]
#     if name == 'ENZYMES':
#         args_dict.update(use_node_attrs=True)
#     dataset_class(**args_dict)

def preprocess_dataset(dataset_path, dataset_name, use_rewired = False):

    """
    Preprocesses the dataset and rewires them
    """
    dataset_dir = os.path.join(dataset_path, dataset_name)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"{dataset_dir} does not exist")

    logging.info(f"Preprocessing started for {dataset_name}")
    if dataset_name in DATASETS:
        dataset_class = DATASETS[dataset_name]

        if dataset_name == 'ENZYMES':
            dataset = dataset_class(root = dataset_path, use_node_attr = True)
            logging.info(f"Additional 18 node attributes used for ENZYMES")
        else:
            dataset = dataset_class(root = dataset_path)
    else:
        dataset = TUDataset(root = dataset_path, name = dataset_name)

    for data in dataset:
        graph = Graph(target =  data.y.item())
        logging.info(f"Processing graph {i+1}/{len(dataset)}")
        graph.add_edges_from(data.edge_index.t().tolist())
    
        if use_rewired:
            rewired_edge_index = graph.rewire_edges()
            data.rewired_edge_index = rewired_edge_index

    torch.save(dataset, os.path.join(dataset_path, f"{dataset_name}_processed.pt"))
    print(f"Dataset {dataset_name} proccesed & saved {dataset_path}.")



if __name__ == "__main__":
    
    # args_dict = get_args_dict()

    # print(args_dict)

    # dataset_name = args_dict.pop('dataset_name')
    # if dataset_name == 'all':
    #     for name in DATASETS:
    #         preprocess_dataset(name, args_dict)
    # else:
    #     preprocess_dataset(dataset_name, args_dict)

    parser = argparse.ArgumentParser(description="Preprocess datasets.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset folder.")
    parser.add_argument("--dataset-name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--use-rewired", action="store_true", help="Add rewired edges to the dataset.")
    args = parser.parse_args()
    
    preprocess_dataset(args.dataset_path, args.dataset_name, use_rewired=args.use_rewired)

# use it like: python PrepareDatasets.py DATA/CHEMICAL --dataset-name NCI1 --use-rewired