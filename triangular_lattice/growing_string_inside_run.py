#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto
# 2016-10-22


from growing_string_inside import InsideString
import time
import networkx as nx
import pygraphviz
import matplotlib.pyplot as plt

current_time = time.strftime("%y%m%d_%H%M%S")
L = 60
frames = 400
beta = 4.

params = {
    'Lx': L,
    'Ly': L,
    'frames': frames,
    'beta': beta,
    'boundary': {'h': 'periodic', 'v': 'periodic'},
    # 'boundary': {'h': 'reflective', 'v': 'reflective'},
    'plot': True,
    'plot_surface': True,
    'interval': 1,
}

# save img
# params.update({
#     'plot': False,
#     'plot_surface': False,
#     'save_image': True,
#     'filename_image': "results/img/inside/" + "beta=%2.2f_" % beta + current_time + ".png",
# })

# save video
params.update({
    'plot': False,
    'plot_surface': True,
    'save_image': False,
    'save_video': True,
    'filename_video': "results/video/inside/" + "beta=%2.2f_" % beta + current_time + ".mp4",
})

# save img and video
# params.update({
#     'plot': False,
#     'save_image': True,
#     'save_video': True,
#     'filename_image': "results/img/inside/" + "beta=%2.2f_" % beta + current_time + ".png",
#     'filename_video': "results/video/inside/" + "beta=%2.2f_" % beta + current_time + ".mp4",
# })

main = InsideString(initial_state=[(L / 2, L / 2 - 1)], **params)

# g = nx.nx_agraph.to_agraph(main.G)
# g.draw("results/img/inside/tree_" + "beta=%2.2f_" % beta + current_time + ".png",
#        prog='neato')

pos = nx.drawing.nx_agraph.graphviz_layout(main.G, prog='neato')
nx.draw(main.G, pos, node_size=20, alpha=0.5, node_color='blue', with_labels=False)
plt.axis('equal')
plt.savefig("results/img/inside/tree_" + "beta=%2.2f_" % beta + current_time + ".png")
plt.show()




