#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-22


from growing_string_inside import InsideString
import time
import matplotlib.pyplot as plt


current_time = time.strftime("%y%m%d_%H%M%S")
L = 60
frames = 200
beta = 0.
basedir_img = "results/img/inside/"
basedir_video = "results/video/inside/"
basefn = "beta=%2.2f_" % beta + current_time


class Parameters(dict):
    def save_image(self):
        self.update({
            'plot': False,
            'save_image': True,
            'filename_image': basedir_img + basefn + ".png",
        })

    def save_video(self):
        self.update({
            'plot': False,
            'save_video': True,
            'filename_video': basedir_video + basefn + ".mp4",
        })

def view_network(plot=True, save_image=False):
    import networkx as nx
    import pygraphviz

    # g = nx.nx_agraph.to_agraph(main.G)
    # g.draw(basedir_img + 'tree_' + basefn  + ".png", prog='neato')
    # print '[saved] ' + basedir_img + 'tree_' + basefn  + ".png"

    pos = nx.drawing.nx_agraph.graphviz_layout(main.G, prog='neato')
    nx.draw(main.G, pos, node_size=20, alpha=0.5, node_color='blue',
            with_labels=False)
    plt.axis('equal')
    if save_image:
        plt.savefig(basedir_img + 'tree_' + basefn  + ".png")
        print '[saved] ' + basedir_img + 'tree_' + basefn  + ".png"
    if plot:
        plt.show()
    else:
        plt.close()

def save_to_json():
    from networkx.readwrite import json_graph
    import json
    from pprint import pprint

    # save to json file
    g_json = json_graph.node_link_data(main.G)
    g_json['links'] = [
        {
            'source': g_json['nodes'][link['source']]['id'],
            'target': g_json['nodes'][link['target']]['id']
        }
        for link in g_json['links']
    ]
    with open(basedir_img + 'tree_data/' + 'tree_' + basefn  + ".json", 'w') as outfile:
        json.dump(g_json, outfile)
    print '[saved] ' + basedir_img + 'tree_data/' + 'tree_' + basefn  + ".json"

params = Parameters({
    'Lx': L,
    'Ly': L,
    'frames': frames,
    'beta': beta,
    'boundary': {'h': 'periodic', 'v': 'periodic'},
    # 'boundary': {'h': 'reflective', 'v': 'reflective'},
    'plot': True,
    'plot_surface': True,
    'interval': 1,
})

# params.save_image()
# params.save_video()

main = InsideString(initial_state=[(L / 2, L / 2 - 1)], **params)

# view_network(plot=True, save_image=False)
save_to_json()

