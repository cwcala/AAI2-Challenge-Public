import networkx as nx
import csv
import numpy as np
from random import randint
from sklearn.linear_model import LogisticRegression

# Create a directed graph
G = nx.read_weighted_edgelist('edgelist.txt', delimiter=' ', create_using=nx.DiGraph(), nodetype=int)
m = G.number_of_edges()

# Read the textual content of each domain name
text = list()
for i in range(33226):
    with open('node_information/text/'+str(i)+'.txt', 'r', errors='ignore') as f:
        text.append(f.read())

# Map text to set of terms
text = [set(text[i].split()) for i in range(len(text))]

# Create the training matrix. Each row corresponds to a pair of nodes and
# its class label is 1 if it corresponds to an edge and 0, otherwise.
# Use the following 3 features for each pair of nodes:
# (1) number of unique terms of the source node's textual content
# (2) number of unique terms of the target node's textual content
# (3) number of common terms between the textual content of the two nodes
X_train = np.zeros((2*m, 3))
y_train = np.zeros(2*m)
n = G.number_of_nodes()
for i,edge in enumerate(G.edges()):
    # an edge
    X_train[2*i,0] = len(text[edge[0]])
    X_train[2*i,1] = len(text[edge[1]])
    X_train[2*i,2] = len(text[edge[0]].intersection(text[edge[1]]))
    y_train[2*i] = 1

    # a randomly generated pair of nodes
    n1 = randint(0, n-1)
    n2 = randint(0, n-1)
    X_train[2*i+1,0] = len(text[n1])
    X_train[2*i+1,1] = len(text[n2])
    X_train[2*i+1,2] = len(text[n1].intersection(text[n2]))
    y_train[2*i+1] = 0

# Read test data. Each sample is a pair of nodes
node_pairs = list()
with open('test.txt', 'r') as f:
    for line in f:
        t = line.split(',')
        node_pairs.append((int(t[0]), int(t[1])))

# Create the test matrix. Use the same 4 features as above
X_test = np.zeros((len(node_pairs), 3))
for i,node_pair in enumerate(node_pairs):
    X_test[i,0] = len(text[node_pair[0]])
    X_test[i,1] = len(text[node_pair[1]])
    X_test[i,2] = len(text[node_pair[0]].intersection(text[node_pair[1]]))

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
