import networkx as nx

def jaccard_wt(graph, node):

    """
      The weighted jaccard score, defined above.
      Args:
        graph....a networkx graph
        node.....a node to score potential new edges for.
      Returns:
        A list of ((node, ni), score) tuples, representing the
                  score assigned to edge (node, ni)
                  (note the edge order)
    """
    pass
    scores =[]
    suma=0
    neighbors = set(graph.neighbors(node))
#     print(neighbors)
    for n in neighbors:
        suma += graph.degree(n)
    suma = 1/suma
    for n in sorted(graph.nodes()):
        if n not in neighbors and n !=node:
            sumb=0
            denominator =0
            numerator=0
            neighbors2 = set(graph.neighbors(n))
            for neighbor2 in neighbors2:
                sumb += graph.degree(neighbor2)
            sumb = 1/sumb
            aandb = neighbors2 & neighbors
            for neigh in sorted(aandb):
                numerator += 1/(graph.degree(neigh))
            denominator = suma + sumb
            final = numerator / denominator
            scores.append(((node,n),final))
    sorted_score = sorted(scores, key=lambda x: (-x[1],x[0][1]))
    return(sorted_score)
