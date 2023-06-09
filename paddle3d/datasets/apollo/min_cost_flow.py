# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------------------------------------------------------------------------
# Modified from BEV-LaneDet (https://github.com/gigo-team/bev_lane_det)
# ------------------------------------------------------------------------

# ==============================================================================
# Binaries and/or source for the following packages or projects are presented under one or more of the following open
# source licenses:
# MinCostFlow.py       The PersFormer Authors        Apache License, Version 2.0
#
# Contact simachonghao@pjlab.org.cn if you have any issue
#
# See:
# https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection/blob/master/tools/MinCostFlow.py
#
# Copyright (c) 2022 The PersFormer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function

import numpy as np
from ortools.graph import pywrapgraph


def SolveMinCostFlow(adj_mat, cost_mat):
    """
        Solving an Assignment Problem with MinCostFlow"
    :param adj_mat: adjacency matrix with binary values indicating possible matchings between two sets
    :param cost_mat: cost matrix recording the matching cost of every possible pair of items from two sets
    :return:
    """

    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    # Define the directed graph for the flow.

    cnt_1, cnt_2 = adj_mat.shape
    cnt_nonzero_row = int(np.sum(np.sum(adj_mat, axis=1) > 0))
    cnt_nonzero_col = int(np.sum(np.sum(adj_mat, axis=0) > 0))

    # prepare directed graph for the flow
    start_nodes = np.zeros(cnt_1, dtype=np.int).tolist() +\
                  np.repeat(np.array(range(1, cnt_1+1)), cnt_2).tolist() + \
                  [i for i in range(cnt_1+1, cnt_1 + cnt_2 + 1)]
    end_nodes = [i for i in range(1, cnt_1+1)] + \
                np.repeat(np.array([i for i in range(cnt_1+1, cnt_1 + cnt_2 + 1)]).reshape([1, -1]), cnt_1, axis=0).flatten().tolist() + \
                [cnt_1 + cnt_2 + 1 for i in range(cnt_2)]
    capacities = np.ones(
        cnt_1, dtype=np.int).tolist() + adj_mat.flatten().astype(
            np.int).tolist() + np.ones(
                cnt_2, dtype=np.int).tolist()
    costs = (np.zeros(cnt_1, dtype=np.int).tolist() + cost_mat.flatten().astype(
        np.int).tolist() + np.zeros(cnt_2, dtype=np.int).tolist())
    # Define an array of supplies at each node.
    supplies = [min(cnt_nonzero_row, cnt_nonzero_col)] + np.zeros(
        cnt_1 + cnt_2,
        dtype=np.int).tolist() + [-min(cnt_nonzero_row, cnt_nonzero_col)]

    source = 0
    sink = cnt_1 + cnt_2 + 1

    # Add each arc.
    for i in range(len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(
            start_nodes[i], end_nodes[i], capacities[i], costs[i])

    # Add node supplies.
    for i in range(len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    match_results = []
    # Find the minimum cost flow between node 0 and node 10.
    if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
        for arc in range(min_cost_flow.NumArcs()):

            # Can ignore arcs leading out of source or into sink.
            if min_cost_flow.Tail(arc) != source and min_cost_flow.Head(
                    arc) != sink:

                # Arcs in the solution have a flow value of 1. Their start and end nodes
                # give an assignment of worker to task.

                if min_cost_flow.Flow(arc) > 0:
                    match_results.append([
                        min_cost_flow.Tail(arc) - 1,
                        min_cost_flow.Head(arc) - cnt_1 - 1,
                        min_cost_flow.UnitCost(arc)
                    ])
    else:
        print('There was an issue with the min cost flow input.')

    return match_results


def main():
    """Solving an Assignment Problem with MinCostFlow"""

    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    # Define the directed graph for the flow.

    start_nodes = [0, 0, 0, 0] + [
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4
    ] + [5, 6, 7, 8]
    end_nodes = [1, 2, 3, 4] + [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8
                                ] + [9, 9, 9, 9]
    capacities = [1, 1, 1, 1] + [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ] + [1, 1, 1, 1]
    costs = ([0, 0, 0, 0] + [
        90, 76, 75, 70, 35, 85, 55, 65, 125, 95, 90, 105, 45, 110, 95, 115
    ] + [0, 0, 0, 0])
    # Define an array of supplies at each node.
    supplies = [4, 0, 0, 0, 0, 0, 0, 0, 0, -4]
    source = 0
    sink = 9
    tasks = 4

    # Add each arc.
    for i in range(len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(
            start_nodes[i], end_nodes[i], capacities[i], costs[i])

    # Add node supplies.

    for i in range(len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])
    # Find the minimum cost flow between node 0 and node 10.
    if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
        print('Total cost = ', min_cost_flow.OptimalCost())
        print()
        for arc in range(min_cost_flow.NumArcs()):

            # Can ignore arcs leading out of source or into sink.
            if min_cost_flow.Tail(arc) != source and min_cost_flow.Head(
                    arc) != sink:

                # Arcs in the solution have a flow value of 1. Their start and end nodes
                # give an assignment of worker to task.

                if min_cost_flow.Flow(arc) > 0:
                    print('Worker %d assigned to task %d.  Cost = %d' %
                          (min_cost_flow.Tail(arc), min_cost_flow.Head(arc),
                           min_cost_flow.UnitCost(arc)))
    else:
        print('There was an issue with the min cost flow input.')
