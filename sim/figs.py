import random
import time
import sys

import numpy as np
from termcolor import colored
import seaborn as sns
import pandas as pd

import plotly.subplots as sp
import plotly.graph_objects as go
import plotly.colors as cm 

sns.set()

def go_parcoords(sim, nodes, cscale, params, rnum):
    dimensions = []
    dimensions.append(
        dict(
            range = [1, max(nodes[rnum]['name'])],
            label = 'Nodes', 
            tickvals = [v.name for v in sim.auctioneer.node_list()],
            ticktext = [v.name for v in sim.auctioneer.node_list()],
            values = nodes[rnum]['name']
        )
    )
    dimensions.append(
        dict(
            range = [1, max(nodes[rnum]['name'])],
            label = 'Buyers', 
            tickvals = [v.name for v in sim.auctioneer.node_list('buyer')],
            ticktext = [v.name for v in sim.auctioneer.node_list('buyer')],
            values = nodes[rnum]['name']
        )
    )
    dimensions.append(
        dict(
            range = [1, max(nodes[rnum]['name'])],
            label = 'Sellers', 
            tickvals = [v.name for v in sim.auctioneer.node_list('seller')],
            ticktext = [v.name for v in sim.auctioneer.node_list('seller')],
            values = nodes[rnum]['name']
        )
    )
    for k in range(max(0,rnum-10),rnum):
        dimensions.append(
            dict(
                label = 'Round' + str(k) + 'price', 
                range = [0, max(nodes[k]['price'])+10],
                values = nodes[k]['price']
            )
        )
        dimensions.append(
            dict(
                range = [1, max(nodes[rnum]['name'])],
                label = 'Winners', 
                tickvals = [v.name for v in sim.auctioneer.node_list('winner')],
                ticktext = [v.name for v in sim.auctioneer.node_list('winner')],
                values = nodes[rnum]['name']
            )
        )
        dimensions.append(
            dict(
                label = 'Round' + str(k) + 'value', 
                range = [0, max(nodes[k]['value'])+10],
                values = nodes[k]['value']
            )
        )
 
    return go.Parcoords(
            line = dict(
                    color = nodes[rnum]['color'],
                    colorscale = cscale,
                ),
            dimensions = dimensions
        )

def go_scatter(sim, nodes, cscale, msize, rnum):
    x=[]
    y=[]
    color=[]
    ntype=[]
    ids=[]
    for k in range(max(0,rnum-5), rnum):
        print("NODESK",nodes[k])
        for j in nodes[k]['type']:
            if j == 'seller':
                x.append(k+0.5)
            else:
                x.append(k)
        for j in nodes[k]['name']:
            ids.append(j)
            y.append(sim.auctioneer.layout[int(j)][1]) 
        for j in nodes[k]['color']:
            color.append(j)
        for j in nodes[k]['type']:
            ntype.append(j)
    print("============================HERE============================")
    return go.Scatter(
                x=x,
                y=y,
                hovertext=ntype,
                hovertemplate='<b>%{hovertext}</b><extra></extra>',
                ids=ids,
                text=ids,
                showlegend=True,
                mode = 'markers',
                opacity = 0.9,
                marker = {  
                            'color': color,
                            'colorscale': cscale,
                            'size' : msize
                        },
                )

def go_line(sim, nodes, edges, cscale, rnum):
    x=[]
    y=[]
    color=[]
    for k in range(max(0,rnum-5), rnum):
        u = list(edges[k]['source'])
        v = list(edges[k]['target'])
        print(u,v)
        print("****************************HERE****************************")
        print("EDGES", [(u[n],v[n]) for n in range(len(v))])
        for n in range(len(edges[k]['source'])):
            maxu = max(edges[k]['source'])
            u = edges[k]['source'][n]
            v = edges[k]['target'][n]
            x.append(k)
            if edges[k]['type'][n] == 'bid':
                x.append(k+0.5)
            else:
                x.append(k+1)
            y.append(sim.auctioneer.layout[u][1])
            y.append(sim.auctioneer.layout[v][1])
            color.append(u/maxu)
            x.append(None)
            y.append(None)
    return go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    line = {  
                            'width' : 0.5,
                            'color' : 'lightgrey'
                            },
                    )

def go_sankey(sim, nodes, edges, cscale, rnum):
    k=rnum
    return go.Sankey(
        valueformat = ".0f",
        valuesuffix = "TWh",
        # Define nodes
        node = dict(
          pad = 15,
          thickness = 15,
          line = dict(color = "black", width = 0),
          label =  nodes[k]['name'],
          color =  cm.sample_colorscale(cscale, nodes[k]['color'], high=max(nodes[k]['color']))
        ),
        # Add links
        link = dict(
          source = edges[k]['source'],
          target = edges[k]['target'],
          value =  edges[k]['capacity'],
          label = edges[k]['source'],
          color =  cm.sample_colorscale(cscale, nodes[k]['color'], high=max(nodes[k]['color']))
        )
    )

def go_auctioncoords(sim, df, cscale, params, rnum):
    sim.ntraces+=1
    auctions=[]
    sellers = []
    buyers = []
    print("AUCTIONS", df['data']['auctions'])
    print("NEIGHBORS", df['data']['neighbors'])
    for auction in df['data']['auctions']:
        auctions.append(auction[0])
        sellers.append(auction[0])
        for v in auction[1]:
            auctions.append(v)
            buyers.append(v)
            sellers.append(auction[0])
    print("ALL", auctions)
    return go.Parcoords(
            line = dict(
                    color = [v.color for v in auctions],
                    colorscale = cscale,
                ),
            dimensions = list([
            dict(
                range = [1, len(df['nodes'])],
                label = 'Buyers', 
                tickvals = [v.name for v in buyers],
                ticktext = [v.name for v in buyers],
                values = [v.name for v in auctions],
            ),
            dict(
                range = [1, len(df['nodes'])],
                label = 'Sellers', 
                multiselect=True,
                tickvals = [v.name for v in sellers],
                ticktext = [v.name for v in sellers],
                values = [v.name for v in auctions],
            ),
            dict(
                label = 'Price', 
                range = [0, max(df['nodes']['price'])],
                values = [v.price for v in auctions],
            )
            ])
        )

def go_scatter3d(sim, df, cscale, surfcolor, msize):
    sim.ntraces+=1
    return go.Scatter3d(
                x=df['pos_x'],
                y=df['pos_y'],
                z=df['pos_z'],
                hovertext=df['type'],
                ids=df['name'],
                hovertemplate='<b>%{hovertext}</b><extra></extra>',
                showlegend=False,
                mode = 'markers',
                surfacecolor = surfcolor,
                surfaceaxis = 2,
                opacity = 0.5,
                marker = {  
                            'color': df['color'],
                            'colorscale': cscale,
                            'size' : msize
                        },
                )
'''
We consider that all nodes are equipped with IEEE 802.11a 11 Mbps network interface card, whose electric currents are 280 mA and
330 mA in reception mode and transmission mode, respectively, and the electric potential is 5 V
'''
def calc_circle(r, c):
  theta = np.linspace(0, 2*np.pi, 50)
  return c + r*np.exp(1.0j*theta)

def go_scattersmith(sim, nodes, edges, cscale):
    sim.ntraces+=1
    edge_list = edges.set_index(['source', 'target'])
    for u, v in edge_list.index:
        u/=len(edge_list.index)
        v/=len(edge_list.index)
        values=calc_circle(u,v)
    return go.Scattersmith(
                        real=np.real(values),
                        imag=np.imag(values)
                        )

def go_volume(sim, df, cscale):
    sim.ntraces+=1
    X, Y, Z = np.mgrid[0:150:5j, 0:150:5j, 0:150:5j]
    print(X)
    print(X.flatten())
    print(len(X.flatten()))
    print(df['price'].values)
    print(len(df['price'].values))

    return go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=df['price'],
                hovertext=df['type'],
                ids=df['name'],
                hovertemplate='<b>%{hovertext}</b><extra></extra>',
                opacity = 0.5,
                surface_count=40,
                colorscale=cscale
                )



def plot_clock(sim, clock, row, col):
    sim.read_clock(clock,1)
    return
    global mk
    params = mk.make_params()
    buyers = mk.buyer_list()
    sellers = mk.seller_list()
    inf = np.array([
                    [-1 for v in mk.seller_list(buyer)
                    ] for buyer in buyers], dtype=object
                ).flatten()
    values = [v.id for v in clock.inf_nodes()]
    rand = np.random.randint(0,len(inf), size=len(values))
    for i in range(len(values)):
        inf[rand[i]] = values[i]
    nbrs = np.array([
                [-1 for v in sim.seller_list(buyer)
                ] for buyer in buyers], dtype=object
            ).flatten()
    values = np.array([
                    [v.name for v in n.neighbors
                    ] for n in clock.inf_nodes()], dtype=object
                ).flatten(),
    rand = np.random.randint(0,len(nbrs), size=len(values))
    for i in range(len(values)):
        nbrs[rand[i]] = values[i]
    sim.fig.add_trace(go.Parcoords(
        line = dict(
                    color = np.array([
                                [buyer.color for v in sim.seller_list(buyer)
                                ] for buyer in buyers]
                            ).flatten(),
                    colorscale = cscale1,
                ),
        dimensions = list([
            dict(
                range = [1, sim.nbuyers()],
                label = 'Auction', 
                values = np.array([
                            [buyer.name for v in sim.seller_list(buyer)
                            ] for buyer in buyers], dtype=object
                        ).flatten()
            ),
            dict(
                range = [1, params['seller']['max_price']],
                label = 'Price', 
                values = np.array([
                                [v.price for v in sim.seller_list(buyer)
                                ] for buyer in buyers], dtype=object
                        ).flatten()
            ),
            dict(
                range = [1, params['buyer']['max_price']],
                label = 'Bid', 
                values = np.array([
                                [buyer.price for v in sim.seller_list(buyer)
                                ] for buyer in buyers], dtype=object
                                ).flatten()
            ),
           dict(
                range = [-1, max([v.name for v in buyers])],
                tickvals = [v.name for v in clock.inf_nodes()],
                label = 'Winners', 
                values = nbrs
             ),
           dict(
                range = [-1, max([v.name for v in buyers])],
                tickvals = np.array([
                                [v.name for v in n.neighbors
                                ] for n in clock.inf_nodes()], dtype=object
                            ).flatten(),
                label = 'Influence', 
                values = inf
            )
           ]) 
           ), row=row, col=col)
    sim.ntraces+=1


def add_scatter_plots(sim):
    sim.add_scatter3d(nodes, edges, ids, cscale1, row=1, col=1)
    corr = mat.corr()
    links = corr.stack().reset_index()
    links.columns = ['b', 's', 'v']
    corr_edges = links.loc[ (links['v'] > 0) & (links['b'] != links['s']) ]
    G = nx.from_pandas_edgelist(corr_edges, 'b', 's')
    pos = nx.spectral_layout(G, dim=3)
    for v in nodes:
        v.pos = pos[v]
    sim.add_scatter3d(nodes, corr_edges, ids, cscale1, row=1, col=2)
    
def go_cone(sim, source, target, cscale):
    print(source)
    return go.Cone(
                    x=[source['pos_x']],
                    y=[source['pos_y']],
                    z=[source['pos_z']],
                    u=[source['pos_x']*target['pos_x'], 0, 0],
                    v=[0, source['pos_y']*target['pos_y'], 0],
                    w=[0, 0, source['pos_z']*target['pos_z']],
                    colorscale=cscale,
                    sizeref=0.1,
                )

def add_contour(sim, mat, ids, cscale, row, col):
    sim.fig.add_trace(go.Contour(
                        x=ids,
                        y=ids,
                        z=mat,
                        hovertemplate='<b>%{z}</b><extra></extra>',
                        ids=ids,
                        showlegend=False,
                        contours = {'coloring': 'heatmap'},
                        colorscale = cscale2,
                        line = {'width': 0},
                        showscale=False
                        ), row=row, col=col)
    sim.ntraces+=1

def update_scatter_plots(sim):
    corr = dict([(k,mat[k].corr()) for k in keys])
    links = [(k,corr[k].stack().reset_index()) for k in keys]
    for k,link in links:
        link.columns = ['b', 's', 'v']
    corr_edges = dict([(k,link.loc[(link['v'] > 0) & (link['b'] != link['s'])]) for k,link in links])
    
    corr_nodes = dict([(k,corr_edges[k].drop(columns='v').stack()) for k in keys])


