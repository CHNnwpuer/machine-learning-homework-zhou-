import math
import pandas as pd
import numpy as np

from treePlotter import createPlot
def entropy(data):
    label_values = data[data.columns[-1]]
    #Returns object containing counts of unique values.
    counts =  label_values.value_counts()
    s = 0
    for c in label_values.unique():
        freq = float(counts[c])/len(label_values)
        s -= freq*math.log(freq,2)
    return s

def is_continuous(data,attr):
    """Check if attr is a continuous attribute"""
    return data[attr].dtype == 'float64'

def split_points(data,attr):
    """Returns Ta,Equation(4.7),p.84"""
    values = np.sort(data[attr].values)
    return [(x+y)/2 for x,y in zip(values[:-1],values[1:])]

def discrete_gain(data,attr):
    V = data[attr].unique()
    s = 0
    for v in V:
        data_v = data[data[attr]== v]
        s += float(len(data_v))/len(data)*entropy(data_v)
    return (entropy(data) - s,None)

def continuous_gain(data,attr,points):
    """Equation(4.8),p.84,returns the max gain along with its splitting point"""
    entD = entropy(data)
    #gains is a list of pairs of the form (gain,t)
    gains = []
    for t in points:
        d_plus = data[data[attr] > t]
        d_minus = data[data[attr] <= t]
        gain = entD - (float(len(d_plus))/len(data)*entropy(d_plus)+float(len(d_minus))/len(data)*entropy(d_minus))
        gains.append((gain,t))
    return max(gains)

def gain(data,attr):
    if is_continuous(data,attr):
        points = split_points(data,attr)
        return continuous_gain(data,attr,points)
    else:
        return discrete_gain(data,attr)

def majority(label_values):
    counts = label_values.value_counts()
    return counts.index[0]

def id3(data):
    attrs = data.columns[:-1]
    #attrWithGain is of the form [(attr,(gain,t))], t is None if attr is discrete
    attrWithGain = [(a,gain(data,a)) for a in attrs]
    attrWithGain.sort(key = lambda tup:tup[1][0],reverse = True)
    return attrWithGain[0]

def createTree(data,split_function):
    label = data.columns[-1]
    label_values = data[label]
    #Stop when all classes are equal
    if len(label_values.unique()) == 1:
        return label_values.values[0]
    #When no more features, or only one feature with same values, return majority
    if data.shape[1] == 1 or (data.shape[1]==2 and len(data.T.ix[0].unique())==1):
        return majority(label_values)
    bestAttr,(g,t) = split_function(data)
    #If bestAttr is discrete
    if t is None:
        #In this tree,a key is a node, the value is a list of trees,also a dictionary
        myTree = {bestAttr:{}}
        values = data[bestAttr].unique()
        for v in values:
            data_v = data[data[bestAttr]== v]
            attrsAndLabel = data.columns.tolist()
            attrsAndLabel.remove(bestAttr)
            data_v = data_v[attrsAndLabel]
            myTree[bestAttr][v] = createTree(data_v,split_function)
        return myTree
    #If bestAttr is continuous
    else:
        t = round(t,3)
        node = bestAttr+'<='+str(t)
        myTree = {node:{}}
        values = ['yes','no']
        for v in values:
            data_v = data[data[bestAttr] <= t] if v == 'yes' else data[data[bestAttr] > t]
            myTree[node][v] = createTree(data_v,split_function)
        return myTree

if __name__ == "__main__":
    f = pd.read_csv(filepath_or_buffer = 'w3.csv', sep = ',')
    data = f[f.columns[1:]]
    tree = createTree(data,id3)
    print(tree)
    createPlot(tree)