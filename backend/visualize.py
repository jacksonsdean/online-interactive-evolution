"""Visualize a CPPN network, primarily for debugging"""
import copy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from backend.nextGeneration.graph_util import required_for_output


def draw_nodes(graph, pos, node_labels, node_size):
    """Draw nodes on the graph"""

    shapes = set((node[1]["shape"] for node in graph.nodes(data=True)))
    for shape in shapes:
        nodes = [sNode[0] for sNode in filter(
            lambda x: x[1]["shape"] == shape, graph.nodes(data=True))]
        colors = [nx.get_node_attributes(graph, 'color')[
            cNode] for cNode in nodes]
        nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=colors,
                               label=node_labels, node_shape=shape, nodelist=nodes)


def add_edges_to_graph(individual, visualize_disabled, graph, pos, required):
    """Add edges to the graph
    Args:
        individual (CPPN): The CPPN to visualize
        visualize_disabled (bool): Whether to visualize disabled nodes
        graph (Graph): The graph to add the edges to
        pos (dict): The positions of the nodes

    Returns:
        edge_labels (dict): labels of edges
    """
    connections = individual.connection_genome
    max_weight = individual.config.max_weight
    edge_labels = {}
 
    for cx in connections:
        if(not visualize_disabled and (not cx.enabled or np.isclose(cx.weight, 0))):
            continue
        style = ('-', 'k',  .5+abs(cx.weight)/max_weight) if cx.enabled\
            else ('--', 'grey', .5 + abs(cx.weight)/max_weight)

        if cx.enabled and cx.weight < 0:
            style = ('-', 'r', .5+abs(cx.weight)/max_weight)


        if cx.from_node in required and cx.to_node in required:
            graph.add_edge(cx.from_node, cx.to_node,
                           weight=f"{cx.weight:.4f}", pos=pos, style=style)
            edge_labels[(cx.from_node, cx.to_node)] = f"{cx.weight:.3f}"

    return edge_labels


def draw_edges(graph, pos, show_weights, node_size, edge_labels):
    """Draw edges on the graph"""

    edge_styles = set((s[2] for s in graph.edges(data='style')))
    for style in edge_styles:
        edges = [e for e in filter(
            lambda x: x[2] == style, graph.edges(data='style'))]
        nx.draw_networkx_edges(graph, pos,
                               edgelist=edges,
                               arrowsize=25, arrows=True,
                               node_size=[node_size]*1000,
                               style=style[0],
                               edge_color=[style[1]]*1000,
                               width=style[2],
                               connectionstyle="arc3"
                               )
    if show_weights:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, label_pos=.75)


def add_input_nodes(individual, node_labels, graph):
    """add input nodes to the graph

    Args:
        individual (CPPN): CPPN to visualize
        node_labels (dictionary): labels of nodes
        graph (Graph): graph to add nodes to
    """
    for i, node in enumerate(individual.input_nodes()):
        graph.add_node(node, color='lightsteelblue',
                       shape='d', layer=(node.layer), subset=node.layer)
        if len(individual.input_nodes()) == 4:
            # includes bias and distance node
            input_labels = ['y', 'x', 'd', 'b']
        else:
            # includes bias node or distance node
            input_labels = ['y', 'x', 'b/d']

        label = f"{node.id}({node.layer})\n{input_labels[i]}:"
        label += f"\n{node.activation.__name__.replace('_activation', '')}"
        if node.outputs is not None:
            label += f"\n{node.outputs:.3f}"

        node_labels[node] = label


def add_hidden_nodes(individual, node_labels, graph, visualize_disabled=False):
    """add input nodes to the graph

    Args:
        individual (CPPN): CPPN to visualize
        node_labels (dictionary): labels of nodes
        graph (Graph): graph to add nodes to
    """
    required = required_for_output(individual.input_nodes(), individual.output_nodes(),
                                   [(cx.from_node, cx.to_node) for cx in individual.enabled_connections()])

    print([n in required for n in individual.hidden_nodes()])
    for node in individual.hidden_nodes():
        if node in required or visualize_disabled:
            graph.add_node(node, color='lightsteelblue',
                        shape='o', layer=(node.layer), subset=node.layer)
            label = f"{node.id}({node.layer})"
            label += f"\n{node.activation.__name__.replace('_activation', '')}"
            if node.outputs is not None:
                label += f"\n{node.outputs:.3f}"
            node_labels[node] = label


def add_output_nodes(individual, node_labels, graph):
    """add input nodes to the graph
    Args:
        individual (CPPN): CPPN to visualize
        node_labels (dictionary): labels of nodes
        graph (Graph): graph to add nodes to
    """
    color_mode = individual.config.color_mode

    for i, node in enumerate(individual.output_nodes()):
        title = color_mode[i] if i < len(color_mode) else 'XXX'
        graph.add_node(node, color='lightsteelblue',
                       shape='s', layer=(node.layer), subset=node.layer)
        label = f"{node.id}({node.layer})\n{title}:"
        label += f"\n{node.activation.__name__.replace('_activation', '')}"
        if node.outputs is not None:
            label += f"\n{node.outputs:.3f}"
        node_labels[node] = label


def add_nodes_to_graph(individual, node_labels, graph, visualize_disabled=False):
    """Add nodes to the graph"""
    add_input_nodes(individual, node_labels, graph)
    add_hidden_nodes(individual, node_labels, graph, visualize_disabled)
    add_output_nodes(individual, node_labels, graph)


def visualize_network(individual, visualize_disabled=False, show_weights=False):
    """Visualize a CPPN network"""
    node_labels = {}
    node_size = 2000
    graph = nx.DiGraph()
    individual = copy.deepcopy(individual)
    individual.update_node_layers()
    # configure plot
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(left=0, bottom=0, right=1.25,
                        top=1.25, wspace=0, hspace=0)



    required = required_for_output(individual.input_nodes(), individual.output_nodes(),
                [(cx.from_node, cx.to_node) for cx in individual.enabled_connections()])
    required = required.union(individual.input_nodes())

    # nodes:
    add_nodes_to_graph(individual, node_labels, graph, visualize_disabled)

    # create the positions
    pos = nx.layout.multipartite_layout(graph)

    # force layers to be sorted on X axis
    x_pos = list(set([pos[n][0] for n in graph]))
    x_pos.sort()
    for k, v in pos.items():
        if k.layer <= len(x_pos):
            pos[k] = [x_pos[k.layer],
                      v[1]]

    # draw
    draw_nodes(graph, pos, node_labels, node_size)

    # edges:
    edge_labels = add_edges_to_graph(
        individual, visualize_disabled, graph, pos, required)
    draw_edges( graph, pos, show_weights, node_size, edge_labels)

    nx.draw_networkx_labels(graph, pos, labels=node_labels)
    plt.tight_layout()
    plt.show()
