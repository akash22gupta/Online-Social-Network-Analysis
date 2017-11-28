"""
cluster.py
"""
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import re
from collections import defaultdict, deque, Counter

def bfs(graph, root, max_depth):
    """
    Perform breadth-first search to compute the shortest paths from a root node to all
    other nodes in the graph. To reduce running time, the max_depth parameter ends
    the search after the specified depth.
    E.g., if max_depth=2, only paths of length 2 or less will be considered.
    This means that nodes greather than max_depth distance from the root will not
    appear in the result.

    You may use these two classes to help with this implementation:
      https://docs.python.org/3.5/library/collections.html#collections.defaultdict
      https://docs.python.org/3.5/library/collections.html#collections.deque

    Params:
      graph.......A networkx Graph
      root........The root node in the search graph (a string). We are computing
                  shortest paths from this node to all others.
      max_depth...An integer representing the maximum depth to search.

    Returns:
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree

    In the doctests below, we first try with max_depth=5, then max_depth=2.

    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> sorted(node2distances.items())
    [('A', 3), ('B', 2), ('C', 3), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('A', 1), ('B', 1), ('C', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('A', ['B']), ('B', ['D']), ('C', ['B']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 2)
    >>> sorted(node2distances.items())
    [('B', 2), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('B', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('B', ['D']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    """
    ###TODO
    pass

    flag = 0
    flag1 = 0
    q = deque()
    q1 = deque()
    q.append(root)
    seen = set()
    node2distances = defaultdict(lambda:-1)
    node2num_paths = defaultdict(int)
    node2parents = defaultdict(list)
    node2distances[root] = 0
    node2num_paths[root] = 1
    #count += 1
    while len(q) > 0:
        n = q.popleft()
        if n not in seen:
            seen.add(n)
        neighbor = graph.neighbors(n)
        for nn in neighbor:
            #if nn not in seen and nn not in q:
             if node2distances[n] < max_depth:
                    if node2distances[nn] < 0:
                        q.append(nn)
                        node2distances[nn] = node2distances[n] + 1


    q1.append(root)
    while len(q1) > 0:
        n1 = q1.popleft()
        if n1 not in seen:
            seen.add(n1)
        neighbors = graph.neighbors(n1)
        for nn in neighbors:
            if node2distances[nn] > node2distances[n1]:
                q1.append(nn)
                if n1 not in node2parents[nn]:
                    node2parents[nn].append(n1)
                    distance = node2distances[nn] - 1
                    if node2distances[n1] == distance:
                        node2num_paths[nn] += node2num_paths[n1]


    node2distances = { k:v for k, v in node2distances.items() if v != -1 }
    #print(sorted(node2distances.items()))
    #print(sorted(node2num_paths.items()))
    #print(sorted(node2parents.items()))

    return node2distances, node2num_paths, node2parents


def bottom_up(root, node2distances, node2num_paths, node2parents):
    """
    Compute the final step of the Girvan-Newman algorithm.
    See p 352 From your text:
    https://github.com/iit-cs579/main/blob/master/read/lru-10.pdf
        The third and final step is to calculate for each edge e the sum
        over all nodes Y of the fraction of shortest paths from the root
        X to Y that go through e. This calculation involves computing this
        sum for both nodes and edges, from the bottom. Each node other
        than the root is given a credit of 1, representing the shortest
        path to that node. This credit may be divided among nodes and
        edges above, since there could be several different shortest paths
        to the node. The rules for the calculation are as follows: ...

    Params:
      root.............The root node in the search graph (a string). We are computing
                       shortest paths from this node to all others.
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    Returns:
      A dict mapping edges to credit value. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).

      Any edges excluded from the results in bfs should also be exluded here.

    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> result = bottom_up('E', node2distances, node2num_paths, node2parents)
    >>> sorted(result.items())
    [(('A', 'B'), 1.0), (('B', 'C'), 1.0), (('B', 'D'), 3.0), (('D', 'E'), 4.5), (('D', 'G'), 0.5), (('E', 'F'), 1.5), (('F', 'G'), 0.5)]
    """
    ###TODO
    pass
    credit = {}
    edge_betweenness = {}
    sorted_node2distances = sorted(node2distances.items(), key=lambda x:x[1], reverse = True)
    #print(sorted_node2distances)
    for k,v in sorted_node2distances:
        credit[k] = 1
    credit[root] = 0
    #print(credit)

    for k,v in sorted_node2distances:
        if node2num_paths[k] == 1 and k!=root:
            #print(node2num_paths[k],k, node2parents[k])
            #print(credit[k], credit[node2parents[k][0]])
            credit[node2parents[k][0]] = credit[node2parents[k][0]] + credit[k]
            if k > node2parents[k][0]:
                edges = [node2parents[k][0],k]
            else:
                edges = [k,node2parents[k][0]]
            edge_betweenness[(edges[0],edges[1])] = 1.*credit[k]
        elif node2num_paths[k] > 1 and k!=root:
            credit[k] = credit[k]/node2num_paths[k]
            for parent in node2parents[k]:
                credit[parent] = credit[parent] + (credit[k] * node2num_paths[parent])
                if k > parent:
                    edges = [parent,k]
                else:
                    edges = [k,parent]
                edge_betweenness[(edges[0],edges[1])] = 1.*credit[k]* node2num_paths[parent]

    #print(credit)
    #print(sorted(edge_betweenness.items()))
    #print(edge_betweenness)
    return edge_betweenness

def approximate_betweenness(graph, max_depth):
    """
    Compute the approximate betweenness of each edge, using max_depth to reduce
    computation time in breadth-first search.

    You should call the bfs and bottom_up functions defined above for each node
    in the graph, and sum together the results. Be sure to divide by 2 at the
    end to get the final betweenness.

    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.

    Returns:
      A dict mapping edges to betweenness. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).

    >>> sorted(approximate_betweenness(example_graph(), 2).items())
    [(('A', 'B'), 2.0), (('A', 'C'), 1.0), (('B', 'C'), 2.0), (('B', 'D'), 6.0), (('D', 'E'), 2.5), (('D', 'F'), 2.0), (('D', 'G'), 2.5), (('E', 'F'), 1.5), (('F', 'G'), 1.5)]
    """
    ###TODO
    pass

    betweenness = defaultdict(int)
    nodes = graph.nodes()
    for node in nodes:
        node2distances,node2num_paths,node2parents=bfs(graph,node,max_depth)
        result = bottom_up(node, node2distances, node2num_paths, node2parents)
        for k in result:
            betweenness[k] += result[k]
    for k in betweenness:
        betweenness[k] = betweenness[k]/2
    #print(sorted(betweenness.items()))
    return betweenness

def partition_girvan_newman(graph, max_depth):
    """
    Use your approximate_betweenness implementation to partition a graph.
    Unlike in class, here you will not implement this recursively. Instead,
    just remove edges until more than one component is created, then return
    those components.
    That is, compute the approximate betweenness of all edges, and remove
    them until multiple comonents are created.

    You only need to compute the betweenness once.
    If there are ties in edge betweenness, break by edge name (e.g.,
    (('A', 'B'), 1.0) comes before (('B', 'C'), 1.0)).

    Note: the original graph variable should not be modified. Instead,
    make a copy of the original graph prior to removing edges.
    See the Graph.copy method https://networkx.github.io/documentation/development/reference/generated/networkx.Graph.copy.html
    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.

    Returns:
      A list of networkx Graph objects, one per partition.

    >>> components = partition_girvan_newman(example_graph(), 5)
    >>> components = sorted(components, key=lambda x: sorted(x.nodes())[0])
    >>> sorted(components[0].nodes())
    ['A', 'B', 'C']
    >>> sorted(components[1].nodes())
    ['D', 'E', 'F', 'G']
    """
    ###TODO
    pass

    new_graph = graph.copy()

    if new_graph.order() == 1:
        return [new_graph.nodes()]

    approx_betweenness = approximate_betweenness(new_graph, max_depth)
    components = [c for c in nx.connected_component_subgraphs(new_graph)]
    betweenness = sorted(sorted(approximate_betweenness(graph, max_depth).items(), key=lambda x: x[0]), key = lambda x:x[1], reverse= True)

    while len(components) == 1:
        edge_to_remove = sorted(approx_betweenness.items(), key=lambda x: (-x[1],x[0][0],x[0][1]))[0][0]
        #new_graph.remove_edge(*edge_to_remove)
        del approx_betweenness[edge_to_remove]
        components = [c for c in nx.connected_component_subgraphs(new_graph)]
    clusters=[]
    i = 0
    while True:
        if len(clusters) >= 2:
            break
        else:
            new_graph.remove_edge(betweenness[i][0][0], betweenness[i][0][1])
            clusters = list(nx.connected_component_subgraphs(new_graph))
            i += 1
    result = [components[1],components[0]]
    return result






def girvan_newman(graph):


    clusters = partition_girvan_newman(graph, 1)
    # print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
    #       (clusters[0].order(), clusters[1].order()))
    #orders=[clusters[i].order() for i in range(len(clusters))]
    return (clusters[0].order()+clusters[1].order()+10)


def main():
    tweets = pickle.load(open("tweets.pkl", "rb"))
    graph = nx.Graph()
    cluster_file = open('cluster_output.txt', 'w', encoding='utf-8')
    for tweet in tweets:
        if '@' in tweet['text']:
            mentions = re.findall(r'[@]\S+', tweet['text'])
            for mention in mentions:
                graph.add_node(tweet['user']['screen_name'])
                graph.add_node(mention[1:])
                graph.add_edge(tweet['user']['screen_name'], mention[1:])

    # drawing graph
    remove=[node for node,degree in graph.degree().items() if degree < 2]
    graph.remove_nodes_from(remove)
    nx.draw_networkx(graph, pos=None, with_labels=False, node_color='b', node_size=10, alpha=0.5, )
    plt.savefig('unclustered.png')
    num_clusters=girvan_newman(graph)

    cluster_file.write(str(num_clusters))
    cluster_file.write('\n'+str(graph.order()))

if __name__ == '__main__':
    main()
