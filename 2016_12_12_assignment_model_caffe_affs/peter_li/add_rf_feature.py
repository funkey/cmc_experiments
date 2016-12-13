#!/usr/bin/python

# adds a RF "feature" to the given dataset (which is assumed to not have the 
# feature yet)

from pycmc import *
import math


def probToEnergy(prob):

    # Get the probability of being switched on (i.e., label 1). In a simple 
    # probabilistic model, this is
    #
    #   p(y_n==1) = 1/Z exp(-E_n(1))
    #
    # where Z = exp(-E_n(0)) + exp(-E_n(1)) and E_n is an energy.
    #
    #   exp(-E_n(1)) = p(y_n==1)*( exp(-E_n(0)) + exp(-E_n(1)) )
    #   exp(-E_n(1)) = p(y_n==1)*exp(-E_n(0)) + p(y_n==1)*exp(-E_n(1))
    #   exp(-E_n(1))*(1 - p(y_n==1)) = p(y_n==1)*exp(-E_n(0))
    #   exp(-E_n(1) + E_n(0))(1 - p(y_n==1)) = p(y_n==1)
    #   exp(-E_n(1) + E_n(0)) = p(y_n==1)/p(y_n==0)
    #   -E_n(1) + E_n(0) = log(p(y_n==1)/p(y_n==0))
    #   -E_n(1) + E_n(0) = log(p(y_n==1)) - log(p(y_n==0))
    #
    # without loss of generality, we can set E_n(0) = 0
    #
    #   E_n(1) = log(p(y_n==0)) - log(p(y_n==1))
    #
    # this energy is negative, if p(y_n==1) > 0.5

    # ensure numerical stability
    prob = max(0.001, min(0.999, prob))

    return math.log(1.0-prob) - math.log(prob);

def add_rf_feature(rf_filename, project_filename, higher_node_bias = 0, higher_edge_bias = 0):

    crag = Crag()
    nodeFeatures = NodeFeatures(crag)
    edgeFeatures = EdgeFeatures(crag)

    cragStore = Hdf5CragStore(project_filename)
    cragStore.retrieveCrag(crag)
    cragStore.retrieveNodeFeatures(crag, nodeFeatures)
    cragStore.retrieveEdgeFeatures(crag, edgeFeatures)

    print "Read slice node features of dim " + str(nodeFeatures.dims(CragNodeType.SliceNode))
    print "Read assignment node features of dim " + str(nodeFeatures.dims(CragNodeType.AssignmentNode))
    print "Read edge features of dim " + str(edgeFeatures.dims(CragEdgeType.NoAssignmentEdge))

    sliceNodeRandomForest = RandomForest()
    assNodeRandomForest = RandomForest()
    edgeRandomForest = RandomForest()

    sliceNodeRandomForest.read(rf_filename, "classifiers/slice_node_rf");
    assNodeRandomForest.read(rf_filename, "classifiers/assignment_node_rf");
    edgeRandomForest.read(rf_filename, "classifiers/no_assignment_edge_rf");

    print "Adding RF \"feature\"..."

    print "...to node features..."
    for n in crag.nodes():
        prob = 0
        if crag.type(n) == CragNodeType.SliceNode:
            prob = sliceNodeRandomForest.getProbabilities(nodeFeatures[n])[1]
        elif crag.type(n) == CragNodeType.AssignmentNode:
            prob = assNodeRandomForest.getProbabilities(nodeFeatures[n])[1]
        else:
            continue
        nodeFeatures.append(n, probToEnergy(prob))

    print "...to edge features..."
    for e in crag.edges():
        if crag.type(e) == CragEdgeType.NoAssignmentEdge:
            prob = edgeRandomForest.getProbabilities(edgeFeatures[e])[1]
            edgeFeatures.append(e, probToEnergy(prob))

    cragStore.saveNodeFeatures(crag, nodeFeatures);
    cragStore.saveEdgeFeatures(crag, edgeFeatures);

    cragStore = None
