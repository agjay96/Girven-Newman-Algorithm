import pyspark
import itertools
import sys
import time

input_file=str(sys.argv[1])
output_file=str(sys.argv[2])
sc = pyspark.SparkContext()

def iterate_vertexes(root, matrix):
    visited = []
    dict_visited = dict()
    future_visit = [root]
    current_node = {}
    current_edge = {}

    while (future_visit):
        vertex = future_visit.pop(0)
        if vertex == root:
            dict_visited[root] = [[], 0]
            visited.append(root)
        else:
            visited.append(vertex)
        to_update = matrix[vertex] - set(visited)
        for nodes in to_update - set(future_visit):
            future_visit.append(nodes)

        for x in to_update:
            if dict_visited.get(x):
                if dict_visited[x][1] == dict_visited[vertex][1] + 1:
                    dict_visited[x][0].append(vertex)
            else:
                dict_visited[x] = [[vertex], dict_visited[vertex][1] + 1]

    for node in visited[::-1]:
        current_node[node] = 1 if not current_node.get(node) else current_node.get(node) + 1
        parent_node = dict_visited[node][0]
        if len(parent_node):
            edge_credit = float(current_node[node]) / len(parent_node)

        for p in parent_node:
            current_edge[(node, p)] = edge_credit
            current_node[p] = current_node.get(p) + edge_credit if current_node.get(p) else edge_credit
    result = []
    for k, v in current_edge.items():
        result.append((k, v))
    return result

data = sc.textFile(input_file)
header = data.first()
df = data.filter(lambda x: x != header).map(lambda x: x.split(',')).map(lambda x: (x[0], x[1])).groupByKey().filter(lambda x: len(x[1]) >= 7).cache()
pairs = []
df2 = df.collectAsMap()
for i in itertools.combinations(df.map(lambda x: x[0]).collect(), 2):
    intersection_pairs = set(df2[i[0]]).intersection(set(df2[i[1]]))
    if len(intersection_pairs) >= 7:
        pairs.append(i)

matrix = sc.parallelize(pairs).flatMap(lambda x: [(x[0], x[1]), (x[1], x[0])]).groupByKey().mapValues(
    set).collectAsMap()
ans = sc.parallelize(matrix.keys()).flatMap(lambda x: iterate_vertexes(x, matrix)).map(
    lambda x: (tuple(sorted(x[0])), x[1])).reduceByKey(lambda x, y: x + y).map(lambda x: (x[0], x[1] / 2)).cache()
ans=ans.map(lambda x : (tuple(sorted(x[0])),x[1])).collect()
def myfunc(e):
	return (-e[1], e[0])
ans=sorted(ans, key=myfunc)

file=open(output_file,"w+")
for i in ans:
    file.write(str(i[0])+", "+str(i[1])+"\n")