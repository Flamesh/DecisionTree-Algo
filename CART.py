from __future__ import print_function 
import numpy as np 
import pandas as pd 

class TreeNode(object):
    def __init__(self, ids = None, children = [], depth = 0):
        self.ids = ids          
        self.depth = depth       
        self.split_attribute = None 
        self.children = children 
        self.order = None       
        self.label = None       

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label


def gini(freq):
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0/float(freq_0.sum())
    return np.sum(np.power(prob_0,2))
    

class DecisionTreeCART(object):

    def __init__(self, max_depth= 50, min_samples_split = 2, min_gini = 1e-4):
        self.root = None
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split 
        self.Ntrain = 0
        self.min_gini = min_gini
    
    def fit(self, data, target):
        self.Ntrain = data.count()[0]
        self.data = data 
        self.attributes = list(data)
        self.target = target 
        self.labels = target.unique()
        
        ids = range(self.Ntrain)
        self.root = TreeNode(ids = ids, depth = 0)
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.gini < self.min_gini:
                node.children = self._split(node)
                if not node.children: #leaf node
                    self._set_label(node)
                queue += node.children
            else:
                self._set_label(node)
        
    def _gini(self, ids):
        if len(ids) == 0: return 0
        ids = [i+1 for i in ids] 
        freq = np.array(self.target[ids].value_counts())     
        return gini(freq)


    def _set_label(self, node):
        target_ids = [i + 1 for i in node.ids]  
        node.set_label(self.target[target_ids].mode()[0]) 
    
    def _split(self, node):
        ids = node.ids 
        best_gini = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]
        for i, att in enumerate(self.attributes):
            values = self.data.iloc[ids, i].unique().tolist()
            
            if len(values) == 1: continue 
            splits = []
            for val in values: 
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append([sub_id-1 for sub_id in sub_ids])
            
            if min(map(len, splits)) < self.min_samples_split: continue
            
            Gini = 0.0
            
            for split in splits:
                Gini += len(split)*self._gini(split)/len(ids)
                
           
            if Gini < self.min_gini: continue 
            if Gini > best_gini:
                best_gini = Gini
                best_splits = splits
                best_attribute = att
                order = values
        node.set_properties(best_attribute, order)
        child_nodes = [TreeNode(ids = split,
                       depth = node.depth + 1) for split in best_splits]
        return child_nodes

    def predict(self, new_data):
        
        npoints = new_data.count()[0]
        
        labels = [None]*npoints
        for n in range(npoints):
            x = new_data.iloc[n, :]
            node = self.root
            while node.children: 
                node = node.children[node.order.index(x[node.split_attribute])]
            labels[n] = node.label
        
        return labels
