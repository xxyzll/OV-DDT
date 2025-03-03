import torch
import torch.nn as nn
import sys
# clip text encoder
from yolo_world import *
from queue import Queue
import json
import os
import torch.nn.functional as F
import numpy as np


class DecisionNode(torch.nn.Module):
    def __init__(self, text: str, text_embedding, left: bool, right: bool, category_id: int = None):
        super(DecisionNode, self).__init__()
        self.text = text
        if text != 'leaf':
            self.text_embedding = torch.nn.Parameter(
                torch.tensor(text_embedding, dtype=torch.float32)
            )
        else:
            self.category_id = category_id
            self.text_embedding = torch.nn.Linear(text_embedding.shape[0], category_id)
        self.left = left
        self.right = right
        
    def forward(self, x, args):
        if self.text != 'leaf':
            text_embeddings = F.normalize(self.text_embedding, p=2, dim=-1)
            sim = torch.einsum('bc,c->b', x, text_embeddings)
            # sim = (sim*args[0] + args[1])
            ret = torch.sigmoid(sim)[:, None]
        else:
            ret = (self.text_embedding(x))
            # text_embeddings = F.normalize(self.text_embedding, p=2, dim=-1)
            # sim = torch.einsum('bc,kc->bk', x, text_embeddings)
            # ret = (sim*args[0] + args[1])
        return ret
    
    def is_leaf(self):
        return self.text == 'leaf'
    
class TrainableTree(torch.nn.Module):
    def __init__(self, connections, num_class, embedding_path, dtype='decision'):
        super(TrainableTree, self).__init__()
        self.num_class = num_class
        self.embedding_generator = HuggingCLIPLanguageBackbone(
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=['all']
        )
        with open(connections, 'r') as f:
            structure = json.load(f)
        self.connections = structure['connection']
        self.root = structure['root']
        self.dtype = dtype
        self.embedding_path = embedding_path
        if dtype == 'random':
            self.build_rand_tree(self.connections, structure['root'])
            self.run = self.random_forward
        elif dtype == 'decision':
            self.build_tree_squeeze(self.connections, structure['root'])
            # all forward 
            self.run = self.tree_squeeze_forward
            # single path forward
            # self.build_single_path_tree(self.connections, structure['root'])
            # self.run = self.single_path_forward

    def build_tree_squeeze(self, connections, root):
        # build tree with squeeze
        num_nodes = len(connections)
        res_path = []
        node_embeddings = []
        self.nodes = nn.ModuleList()
        num_leaf = 0
        q = Queue()
        q.put((root, []))
        while not q.empty():
            for _ in range(q.qsize()):
                top = q.get()
                node_text, path = top
                if isinstance(node_text, int):
                    res_path.append(path)
                    num_leaf += 1
                    continue        
                embedding = self.embedding_generator([[node_text]])[0][0]
                node_embeddings.append(embedding)
                if connections[node_text]['left'] is not None:
                    q.put((connections[node_text]['left'], path + [1]))

                if connections[node_text]['right'] is not None:
                    q.put((connections[node_text]['right'], path + [-1]))
        
        self.leaf_decisions = torch.nn.Linear(512, self.num_class*num_leaf)
        for path_id, path in enumerate(res_path):
            if len(path) < num_nodes:
                path += [0] * (num_nodes - len(path))
                res_path[path_id] = path
        # num_leaf * num_nodes
        self.res_path = torch.tensor(res_path, dtype=torch.float32).to('cuda')
        self.num_leaf = num_leaf
        # set all trainable 
        self.node_embeddings = torch.nn.Parameter(torch.stack(node_embeddings, dim=0))
        
        
    def tree_squeeze_forward(self, x, args):
        node_embeddings = F.normalize(self.node_embeddings, p=2, dim=-1)
        node_sim = torch.einsum('bc,kc->bk', x, node_embeddings).sigmoid()
        leaf_decisions = self.leaf_decisions(x).view(x.size(0), self.num_leaf, self.num_class)

        # Using path matrix to calculate the product
        left_mask = (self.res_path == 1).unsqueeze(0)
        right_mask = (self.res_path == -1).unsqueeze(0)

        # Calculate the cumulative product for left and right paths
        prob_left = torch.prod(torch.where(left_mask, node_sim.unsqueeze(1), torch.ones_like(node_sim.unsqueeze(1))), dim=-1)
        prob_right = torch.prod(torch.where(right_mask, 1 - node_sim.unsqueeze(1), torch.ones_like(node_sim.unsqueeze(1))), dim=-1)

        # Combine probabilities
        probs = prob_left.unsqueeze(-1) * prob_right.unsqueeze(-1) * leaf_decisions

        return probs.sum(dim=1), node_sim
            
    def build_single_path_tree(self, connections, root):
        leaf_id = 0
        self.nodes = nn.ModuleDict()
        class_embeddings = np.load(self.embedding_path)
        self.root = root
        self.connections = {}
        for key, value in connections.items():
            self.connections[key] = value
            if isinstance(value['left'], int):
                self.connections[key]['left'] = f'leaf_{leaf_id}'
                leaf_id += 1
            if isinstance(value['right'], int):
                self.connections[key]['right'] = f'leaf_{leaf_id}'
                leaf_id += 1
        
        q = Queue()
        q.put(root)

        while not q.empty():
            node_text = q.get()
            if node_text in self.nodes:
                continue
            if 'leaf' in node_text:
                node = DecisionNode('leaf', class_embeddings[0], left = False, right = False, 
                                    category_id=self.num_class)
                self.nodes[node_text] = node
                continue

            embedding = self.embedding_generator([[node_text]])[0][0]
            node = DecisionNode(text=node_text, text_embedding=embedding, 
                                left=connections[node_text]['left'] is not None,
                                right=connections[node_text]['right'] is not None,
                                category_id=self.num_class)
            self.nodes[node_text] = node

            left_text = connections[node_text]['left']
            right_text = connections[node_text]['right']

            if left_text is not None:
                q.put(left_text)
            if right_text is not None:
                q.put(right_text)

    def single_path_forward(self, x):
        batch_size = x.size(0)
        prob = torch.zeros(batch_size, self.num_class, device=x.device)
        q = Queue()
        initial_path = torch.ones(batch_size, 1, device=x.device)
        q.put((self.root, initial_path, torch.arange(batch_size)))

        while not q.empty():
            node_text, path, indices = q.get()
            node = self.nodes[node_text]
            if len(x[indices].shape) == 3:
                print(x)
            decision = node(x[indices])

            if 'leaf' in node_text:
                prob[indices] += path * decision
            else:
                left_mask = (decision < 0.5).squeeze(-1)
                right_mask = (decision >= 0.5).squeeze(-1)

                if left_mask.any():
                    left_indices = indices[left_mask]
                    left_path = path[left_mask] * decision[left_mask]
                    q.put((self.connections[node_text]['left'], left_path, left_indices))
                
                if right_mask.any():
                    right_indices = indices[right_mask]
                    right_path = path[right_mask] * (1 - decision[right_mask])
                    q.put((self.connections[node_text]['right'], right_path, right_indices))

        return prob

    def build_tree(self, connections, root):
        class_embeddings = np.load(self.embedding_path)
        self.nodes = nn.ModuleList()
        q = Queue()
        q.put(root)
        while not q.empty():
            level_nodes = nn.ModuleList()
            for _ in range(q.qsize()):
                node_text = q.get()
                if isinstance(node_text, int):
                    level_nodes.append(DecisionNode('leaf', class_embeddings[node_text], None, None, self.num_class))
                    # level_nodes.append(DecisionNode('leaf', class_embeddings, None, None, self.num_class))
                    continue
                embedding = self.embedding_generator([[node_text]])[0][0]
                node = DecisionNode(node_text, embedding, connections[node_text]['left'] is not None, connections[node_text]['right'] is not None)
                if node.left:
                    q.put(connections[node_text]['left'])
                if node.right:
                    q.put(connections[node_text]['right'])
                level_nodes.append(node)    
            if level_nodes:
                self.nodes.append(level_nodes)

    def tree_forward(self, x, args):
        batch_size = x.size(0)
        path = torch.ones(batch_size, 1, device=x.device)
        outputs = []
        decisions = []

        for level_nodes in self.nodes:
            next_paths = []
            for node_id, node in enumerate(level_nodes):
                decision = node(x, args)
                if node.is_leaf():
                    outputs.append(path[:, node_id:node_id+1] * decision)
                else:
                    decisions.append(decision)
                    if node.left:
                        next_paths.append(path[:, node_id:node_id+1] * decision)
                    if node.right:
                        next_paths.append(path[:, node_id:node_id+1] * (1 - decision))
            if next_paths:
                path = torch.cat(next_paths, dim=1)
            
        prob = torch.zeros(batch_size, self.num_class, device=x.device)
        for category_id, output in enumerate(outputs):
            prob += output
        
        return prob, torch.cat(decisions, dim=1)
    
    def forward(self, x, args):
        return self.run(x, args)



