import networkx as nx
import csv
import numpy as np
import math
import time
import matplotlib


def read(file):  # turns file into two arrays: a list of the variables of each node and a list of rows of data points
    data = []
    with open(file, newline='\n') as f:
        reader = csv.reader(f, delimiter=',')
        nodes = list(next(reader))
        for row in reader:
            line = [int(x) for x in row]
            data.append(line)
        return nodes, data


def get_num_instantiations(node, nodes, data):  # return the number of instantiations that a variable can take on
    ind = nodes.index(node)
    col = []
    for i in range(len(data)):
        col.append(data[i][ind])
    return max(col)


def get_parents(node, graph):  # return the parents of a node in list form
    if not graph.predecessors(node):
        return []
    else:
        return list(graph.predecessors(node))


def p_instantiations(parents, nodes, data): # return list of lists of parental instantiation permutations in order
    num_parents = len(parents)
    instants, matrix = [], []
    for i in range(num_parents):
        instants.append(get_num_instantiations(parents[i], nodes, data))
    # base case: only one parent; return instantiations
    if num_parents == 1:
        for i in range(1, instants[0] + 1):
            row = [i]
            matrix.append(row)
        return matrix
    else:
        remaining_p_instants = p_instantiations(parents[1:], nodes, data)  # get instants for remaining parents
        for i in range(1, instants[0] + 1):  # iterate through instants of first parent
            for j in remaining_p_instants:  # iterate through instants of remaining parents
                matrix.append([i] + j)  # combine val of first parent with list of instants for remaining parents
        return matrix


def fill_matrix(node, nodes, graph, data):  # return the n x m counts matrix for a single node
    instants = get_num_instantiations(node, nodes, data)
    parents = get_parents(node, graph)
    rows, j = 1, 0  # if the node has no parents, we want a 1 x number_instantiations matrix and j = 0 because we only have one row
    if len(parents) > 0:
        p_instants = p_instantiations(parents, nodes, data)
        rows = len(p_instants)
    count = np.zeros((rows, instants), dtype='i')
    for i in range(len(data)):
        if len(parents) > 0:
            p_vals = []
            for p in range(len(parents)):  # check if parents have correct instantiations
                p_vals.append(data[i][nodes.index(parents[p])])
            j = p_instants.index(p_vals)
        data_point = data[i][nodes.index(node)]
        count[j][data_point - 1] += 1
    return count


# return a prior (matrix of all 1s) with the same shape as the n x m matrix of counts for that node
def prior(node, nodes, graph, data):
    instants = get_num_instantiations(node, nodes, data)
    parents = get_parents(node, graph)
    rows = 1
    if len(parents) > 0:
        p_instants = p_instantiations(parents, nodes, data)
        rows = len(p_instants)
    p = np.ones((rows, instants), dtype='i') # creates an all-ones matrix with the correct dimensions
    return p


def final_m(nodes, data, graph):  # returns list of count matrices, one for each node
    final = []
    for i in range(len(nodes)):
        final.append(fill_matrix(nodes[i], nodes, graph, data))
    return final


def final_a(nodes, data, graph):  # returns list of prior matrices, one for each node
    final = []
    for i in range(len(nodes)):
        final.append(prior(nodes[i], nodes, graph, data))
    return final


def bayesian_score_component_r(j, M, a):  # computes the rightmost component of the bayes score algorithm (sum ri)
    sum_r = 0
    for k in range(a.shape[1]):  # COLS
        sum_r += (math.lgamma(a[j][k] + M[j][k]) - math.lgamma(a[j][k]))
    return sum_r


def bayesian_score_component_q(M, a):  # computes the rest of the bayes score algorithm (sum qi)
    sum_q = 0
    for j in range(a.shape[0]):  # ROWS
        sum_q += ((math.lgamma(sum(a[j])) - math.lgamma(sum(a[j]) + sum(M[j]))) + bayesian_score_component_r(j, M, a))
    return sum_q


def bayes_score(nodes, data, graph):  # computes final bayes score, adding up the score for each node (sum n)
    M, a = final_m(nodes, data, graph), final_a(nodes, data, graph)
    n = len(nodes)
    score = 0
    for i in range(n):
        score += bayesian_score_component_q(M[i], a[i])
    return score


def valid_edge(child, graph, max_parents):  # ensures that adding an edge between two nodes does not create a cyclic graph or exceed a maximum number of parents
    if nx.is_directed_acyclic_graph(graph) and len(get_parents(child, graph)) <= max_parents:
        return True
    return False


def k2_search(nodes, data, max_parents):  # standard K2 search algorithm
    G = nx.DiGraph()  # with n number of nodes
    G.add_nodes_from(nodes)
    for k in range(1, len(nodes)):
        score = bayes_score(nodes, data, G)
        while True:
            best_score = -10e10
            best_node = 0
            for j in range(0, k):
                if not G.has_edge(nodes[j], nodes[k]):
                    G.add_edge(nodes[j], nodes[k])
                    if valid_edge(nodes[k], G, max_parents):
                        score_temp = bayes_score(nodes, data, G)
                        if score_temp > best_score:
                            best_score = score_temp
                            best_node = nodes[j]
                        G.remove_edge(nodes[j], nodes[k])
                    else:
                        G.remove_edge(nodes[j], nodes[k])
            if best_score > score:
                score = best_score
                G.add_edge(best_node, nodes[k])
            else:
                break
    return G


def write_gph(dag, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(edge[0], edge[1]))


def main():
    start_time = time.time()
    nodes, data = read("data/large.csv")[0], read("data/large.csv")[1]
    G = k2_search(nodes, data, 4)
    print(bayes_score(nodes, data, G))
    #nx.draw_networkx(G, with_labels=True)
    #matplotlib.pyplot.show()
    end_time = time.time()
    print(end_time-start_time)
    return write_gph(G, "large.gph")


if __name__ == '__main__':
    main()
