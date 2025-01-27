from __future__ import print_function 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import pydotplus

class TreeNode(object):
    def __init__(self, ids = None, children = [], entropy = 0, depth = 0):
        self.ids = ids           # index of data in this node
        self.entropy = entropy   # entropy, will fill later
        self.depth = depth       # distance to root node
        self.split_attribute = None # which attribute is chosen, it non-leaf
        self.children = children # list of its child nodes
        self.order = None       # order of values of split_attribute in children
        self.label = None       # label of node if it is a leaf

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label


def entropy(freq):
    # remove prob 0 
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0/float(freq_0.sum())
    #print(freq_0)
    return -np.sum(prob_0*np.log(prob_0))

class DecisionTreeC45(object):

    def __init__(self, max_depth= 50, min_samples_split = 2, min_gain = 1e-4):
        self.root = None
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split 
        self.Ntrain = 0
        self.min_gain = min_gain
    
    def fit(self, data, target):
        self.Ntrain = data.count()[0]
        self.data = data 
        self.attributes = list(data)
        self.target = target 
        self.labels = target.unique()
        

        ids = range(self.Ntrain)
        self.root = TreeNode(ids = ids, entropy = self._entropy(ids), depth = 0)
        
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.children = self._split(node)
                if not node.children: #leaf node
                    self._set_label(node)
                queue += node.children
            else:
                self._set_label(node)
        
    def _entropy(self, ids):
        # calculate entropy of a node with index ids
        if len(ids) == 0: return 0
        ids = [i+1 for i in ids] # panda series index starts from 1
        freq = np.array(self.target[ids].value_counts())
        
        
        return entropy(freq)

    def _set_label(self, node):
        
        target_ids = [i + 1 for i in node.ids] 
        node.set_label(self.target[target_ids].mode()[0]) 
    
    def _split(self, node):
        ids = node.ids 
        
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]
        lent_ids = len(ids)
        for i, att in enumerate(self.attributes):
            values = self.data.iloc[ids, i].unique().tolist()
            
            if len(values) == 1: continue 
            splits = []
            
            for val in values: 
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append([sub_id-1 for sub_id in sub_ids])
         
            if min(map(len, splits)) < self.min_samples_split: continue
           
            HxS= 0
            IV = 0
            for split in splits:
                p = len(split)/lent_ids
                HxS += p*self._entropy(split)    
                IV = IV + p*np.log(len(split))
                
            gain = node.entropy - HxS
                        
            if(IV==0): continue
            inforGain_ration = gain / IV
            if inforGain_ration < self.min_gain: continue 
            if inforGain_ration > best_gain:
                best_gain = inforGain_ration
                best_splits = splits
                best_attribute = att
                order = values               
        node.set_properties(best_attribute, order)
        child_nodes = [TreeNode(ids = split,
                     entropy = self._entropy(split), depth = node.depth + 1) for split in best_splits]
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
    
