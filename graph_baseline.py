import csv
import networkx as nx
import numpy as np
from random import randint
from sklearn.linear_model import LogisticRegression

# Create a directed graph
G = nx.read_weighted_edgelist('edgelist.txt', delimiter=' ', create_using=nx.DiGraph(), nodetype=int)
nodes = list(G.nodes())
n = G.number_of_nodes()
m = G.number_of_edges()
print('Number of nodes:', n)
print('Number of edges:', m)

# Create the training matrix. Each row corresponds to a pair of nodes and
# its class label is 1 if it corresponds to an edge and 0, otherwise.
# Use the following 4 features for each pair of nodes:
# (1) in-degree of source node
# (2) out-degree of source node
# (3) in-degree of target node
# (4) out-degree of target node
X_train = np.zeros((2*m, 4))
y_train = np.zeros(2*m)
for i,edge in enumerate(G.edges()):
    # an edge
    X_train[2*i,0] = G.in_degree(edge[0])
    X_train[2*i,1] = G.out_degree(edge[0])
    X_train[2*i,2] = G.in_degree(edge[1])
    X_train[2*i,3] = G.out_degree(edge[1])
    y_train[2*i] = 1

    # a randomly generated pair of nodes
    n1 = nodes[randint(0, n-1)]
    n2 = nodes[randint(0, n-1)]
    X_train[2*i+1,0] = G.in_degree(n1)
    X_train[2*i+1,1] = G.out_degree(n1)
    X_train[2*i+1,2] = G.in_degree(n2)
    X_train[2*i+1,3] = G.out_degree(n2)
    y_train[2*i+1] = 0

# Read test data. Each sample is a pair of nodes
node_pairs = list()
with open('test.txt', 'r') as f:
    for line in f:
        t = line.split(',')
        node_pairs.append((int(t[0]), int(t[1])))

# Create the test matrix. Use the same 4 features as above
X_test = np.zeros((len(node_pairs), 4))
for i,node_pair in enumerate(node_pairs):
    X_test[i,0] = G.in_degree(node_pair[0])
    X_test[i,1] = G.out_degree(node_pair[0])
    X_test[i,2] = G.in_degree(node_pair[1])
    X_test[i,3] = G.out_degree(node_pair[1])

# Use logistic regression to predict if two nodes are linked by an edge
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

# Write predictions to a file
y_pred = y_pred[:,1]
predictions = zip(range(len(y_pred)), y_pred)
with open("submission.csv","w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','predicted'])
    for row in predictions:
        csv_out.writerow(row) 