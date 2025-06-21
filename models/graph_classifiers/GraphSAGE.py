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
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, global_max_pool


class GraphSAGE(nn.Module):
    def __init__(self, dim_features, dim_target, config):
        super().__init__()

        num_layers = config['num_layers']
        dim_embedding = config['dim_embedding']
        self.aggregation = config['aggregation']  # can be mean or max
        self.use_rewired_for_all_layers = bool(config.get('rewire_for_all_layers',0))
        

        if self.aggregation == 'max':
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_features if i == 0 else dim_embedding

            # Overwrite aggregation method (default is set to mean
            conv = SAGEConv(dim_input, dim_embedding, aggr=self.aggregation)

            self.layers.append(conv)

        # For graph classification
        self.fc1 = nn.Linear(num_layers * dim_embedding, dim_embedding)
        self.fc2 = nn.Linear(dim_embedding, dim_target)

    def forward(self, data):
        rewired_edge_index = getattr(data,'rewire_edge_index', None) 
        x, edge_index, batch, = data.x, data.edge_index, data.batch

        if self.use_rewired_for_all_layers and rewired_edge_index is not None:
            edge_index = rewired_edge_index

        x_all = []

        for i, layer in enumerate(self.layers):

            if not self.use_rewired_for_all_layers and i == len(self.layers) - 1 and rewired_edge_index is not None:
                edge_index = rewired_edge_index
                
            x = layer(x, edge_index)
            if self.aggregation == 'max':
                x = torch.relu(self.fc_max(x))
            x_all.append(x)

        x = torch.cat(x_all, dim=1)
        x = global_max_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
