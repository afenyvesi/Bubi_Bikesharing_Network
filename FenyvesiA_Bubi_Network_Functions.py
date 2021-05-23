import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def preprocess(stations, trips):
    '''
    - Drops rows with missing values from both tables
    - Drop trips that have a start or end station id that is not in the stations table
    - Dros trips that ended at the same station where they started under 5 minutes
    - Sets start_time and end_time columns in trips to datetime type
    - Create the length column from the two columns mentioned above and calculate it to minutes
    - Creates separate date and time columns
    - Drop trips where length is more than 5 hours, it is most likely an error
    '''
    print('Shape of stations: ', stations.shape)
    print('Shape of trips: ', trips.shape)
    stations = stations.dropna(subset = ['station_id'])
    trips = trips.dropna()
    trips = trips[trips.start_station_id.isin(list(set(stations.station_id))) & trips.end_station_id.isin(list(set(stations.station_id)))]
    trips = trips.drop(index = trips[(trips.start_station_id == trips.end_station_id) & (trips.length < 5)].index)
    trips.start_time = pd.to_datetime(trips.start_time)
    trips.end_time = pd.to_datetime(trips.end_time)
    trips.rename(columns = {'start_time' : 'start_datetime', 'end_time' : 'end_datetime'}, inplace = True)
    trips.insert(6, 'start_date', trips.start_datetime.apply(lambda x : x.date()), True)
    trips.insert(7, 'start_time', trips.start_datetime.apply(lambda x : x.time()), True)
    trips.insert(8, 'end_date',   trips.end_datetime.apply(lambda x : x.date()), True)
    trips.insert(9, 'end_time',   trips.end_datetime.apply(lambda x : x.time()), True)
    trips.insert(10, 'length',  (trips.end_datetime - trips.start_datetime).apply(lambda x: x.seconds / 60), True)
    trips = trips[trips.length <= 300]
    print('Shape of stations: ', stations.shape)
    print('Shape of trips: ', trips.shape)
    
    return stations, trips
    
def filter_datetime(df, filter_start_date = '2018-01-01', filter_end_date = '2020-06-13', filter_start_time = '00:00:00', filter_end_time = '23:59:59'):
    return df[(df.start_date >= pd.Timestamp(filter_start_date)) & (df.end_date <= pd.Timestamp(filter_end_date)) & (df.start_time >= pd.Timestamp(filter_start_time).time()) & (df.end_time <= pd.Timestamp(filter_end_time).time())]
    

def groupbyer(df, cut = 0):
    '''
    This function creates the groupby table needed for edgelist and also filters out the rows that have less trips than cut
    This is for later graphing purposes
    '''
    df = df.groupby(['start_station_id', 'end_station_id']).agg({'trip_id' : 'count', 'length' : 'median'}).reset_index()
    return df[df.trip_id >= cut]
    
def create_edgelist(df):
    return [(row.start_station_id, row.end_station_id, {'weight' : row.trip_id, 'median_length' : row.length}) for i, row in df.iterrows()]
    

def create_graph(df, stations, cut = 0):
    '''
    This functions creates a graph from a time and date filtered trips df
    Edges will have weight according to route freqency
    Edges also have the mean trip length attribute
    Nodes get the name, longitude an latitude attributes from stations df
    '''
    G = nx.DiGraph()
    G.add_edges_from(create_edgelist(groupbyer(df, cut = cut)))
    nx.set_node_attributes(G, {node : stations[stations.station_id == node].name.values[0] for node in list(G.nodes)}, 'name')
    nx.set_node_attributes(G, {node : stations[stations.station_id == node].latitude.values[0] for node in list(G.nodes)}, 'latitude')
    nx.set_node_attributes(G, {node : stations[stations.station_id == node].longitude.values[0] for node in list(G.nodes)}, 'longitude')
    nx.set_node_attributes(G, {node : sum([edge[2]['weight'] for edge in G.edges.data() if edge[0] == node]) for node in G.nodes}, 'traffic_out')
    nx.set_node_attributes(G, {node : sum([edge[2]['weight'] for edge in G.edges.data() if edge[1] == node]) for node in G.nodes}, 'traffic_in')
    nx.set_node_attributes(G, {node : G.nodes[node]['traffic_in'] + G.nodes[node]['traffic_out'] for node in G.nodes}, 'total_traffic')
    nx.set_node_attributes(G, {node : G.nodes[node]['traffic_in'] / G.nodes[node]['total_traffic'] for node in G.nodes}, 'in_ratio')
    
    return G
    
def anal_graph(G):
    print('Nodes in the graph: ', len(G.nodes))
    print('Edges in the graph: ', len(G.edges))
    print('Clustering coefficient: ', nx.average_clustering(G))
    print('Is the graph strongly connected? ', nx.is_strongly_connected(G))
    print('Number of strongly connected components: ', nx.number_strongly_connected_components(G))
    print('Number of weakly connected components: ', nx.number_weakly_connected_components(G))
    
def plot_degree_dist(G, bins = 20):
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    
    axes[0].hist([d for i, d in G.in_degree], bins = bins)
    axes[0].set_title('A hálózat BEMENŐ fokszámeloszlása', font = 'garamond', size = 16, fontweight = 'bold')
    axes[0].set_xlabel('Fokszám', font = 'garamond', size = 16)
    axes[0].set_ylabel('Frekvencia', font = 'garamond', size = 16)

    axes[1].hist([d for i, d in G.out_degree], bins = bins)
    axes[1].set_title('A hálózat KIMENŐ fokszámeloszlása', font = 'garamond', size = 16, fontweight = 'bold')
    axes[1].set_xlabel('Fokszám', font = 'garamond', size = 16)
    axes[1].set_ylabel('Frekvencia', font = 'garamond', size = 16)
    
    plt.show()
    
def plot_weight_dist(G, bins = 20):
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    
    axes[0].hist([edge[2]['weight'] for edge in G.edges.data()], bins = bins)
    axes[0].set_title('A hálózat élsúlyainak eloszlása', font = 'garamond', size = 16, fontweight = 'bold')
    axes[0].set_xlabel('Élsúly', font = 'garamond', size = 16)
    axes[0].set_ylabel('Frekvencia', font = 'garamond', size = 16)

    axes[1].hist([np.log(edge[2]['weight']) for edge in G.edges.data()], bins = bins)
    axes[1].set_title('A hálózat élsúlyainak logaritmikus eloszlása', font = 'garamond', size = 16, fontweight = 'bold')
    axes[1].set_xlabel('Élsúly logaritmusa', font = 'garamond', size = 16)
    axes[1].set_ylabel('Frekvencia', font = 'garamond', size = 16)
    
    plt.show()

def plot_traffic_dist(G, bins = 20):
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    
    axes[0].hist([G.nodes[node]['traffic_in'] for node in G.nodes], bins = bins)
    axes[0].set_title('Az állomások beáramló forgalmának eloszlása', font = 'garamond', size = 16, fontweight = 'bold')
    axes[0].set_xlabel('Forgalom', font = 'garamond', size = 16)
    axes[0].set_ylabel('Frekvencia', font = 'garamond', size = 16)

    axes[1].hist([G.nodes[node]['traffic_out'] for node in G.nodes], bins = bins)
    axes[1].set_title('Az állomások kiáramló forgalmának eloszlása', font = 'garamond', size = 16, fontweight = 'bold')
    axes[1].set_xlabel('Forgalom', font = 'garamond', size = 16)
    axes[1].set_ylabel('Frekvencia', font = 'garamond', size = 16)
    
    plt.show()

def plot_inout_dist(G, cut = 10, bins = 20):

    plt.hist([G.nodes[node]['in_ratio'] for node in G.nodes if G.nodes[node]['total_traffic'] > cut], bins = bins)
    plt.title('A ki-beáramló forgalom arányának eloszlása', font = 'garamond', size = 16, fontweight = 'bold')
    plt.xlabel('Beáramló/összes forgalom', font = 'garamond', size = 16)
    plt.ylabel('Frekvencia', font = 'garamond', size = 16)
    plt.xticks(font = 'garamond')
    plt.yticks(font = 'garamond')
    
    plt.show()

def plot_trip_median(G, cut = 100, bins = 20):
    plt.hist([edge[2]['median_length'] if edge[2]['median_length'] < cut else cut for edge in G.edges.data()], bins = bins)
    plt.title('Az állomások közötti medián úthossz eloszlása (a legmagasabb kategória összevont)', font = 'garamond', size = 16, fontweight = 'bold')
    plt.xlabel('Medián úthossz', font = 'garamond', size = 16)
    plt.ylabel('Frekvencia', font = 'garamond', size = 16)
    plt.xticks(font = 'garamond')
    plt.yticks(font = 'garamond')
    
    plt.show()

def plot_traffic_length(G):
    plt.scatter([edge[2]['median_length'] for edge in G.edges.data()], [edge[2]['weight'] for edge in G.edges.data()])
    plt.title('A medián úthossz és az élek forgalma', font = 'garamond', size = 16, fontweight = 'bold')
    plt.xlabel('Medián úthossz', font = 'garamond', size = 16)
    plt.ylabel('Forgalom', font = 'garamond', size = 16)
    plt.xticks(font = 'garamond')
    plt.yticks(font = 'garamond')
    plt.show()

def plot_traffic_cent(G):
    plt.scatter(list(nx.degree_centrality(G).values()), [node[1]['total_traffic'] for node in G.nodes.data()])
    plt.title('Centrality és a csúcsok forgalma', font = 'garamond', size = 16, fontweight = 'bold')
    plt.xlabel('Centrality', font = 'garamond', size = 16)
    plt.ylabel('Forgalom', font = 'garamond', size = 16)
    plt.xticks(font = 'garamond')
    plt.yticks(font = 'garamond')
    plt.show()

def plot_traffic_bcent(G):
    plt.scatter(list(nx.betweenness_centrality(G).values()), [node[1]['total_traffic'] for node in G.nodes.data()])
    plt.title('Betweenness centrality és a csúcsok forgalma', font = 'garamond', size = 16, fontweight = 'bold')
    plt.xlabel('Betweenness centrality', font = 'garamond', size = 16)
    plt.ylabel('Forgalom', font = 'garamond', size = 16)
    plt.xticks(font = 'garamond')
    plt.yticks(font = 'garamond')
    plt.show()
    
def plot_traffic_degree(G):
    plt.scatter([d for n, d in list(nx.degree(G))], [node[1]['total_traffic'] for node in G.nodes.data()])
    plt.title('Fokszám és a csúcsok forgalma', font = 'garamond', size = 16, fontweight = 'bold')
    plt.xlabel('Fokszám', font = 'garamond', size = 16)
    plt.ylabel('Forgalom', font = 'garamond', size = 16)
    plt.xticks(font = 'garamond')
    plt.yticks(font = 'garamond')
    plt.show()
    
def plot_bcent_length(G):
    plt.scatter(list(nx.edge_betweenness_centrality(G).values()), [edge[2]['median_length'] for edge in G.edges.data()])
    plt.title('Az edge betweenness centrality és a medián úthossz', font = 'garamond', size = 16, fontweight = 'bold')
    plt.xlabel('Edge betweenness centrality', font = 'garamond', size = 16)
    plt.ylabel('Medián úthossz', font = 'garamond', size = 16)
    plt.xticks(font = 'garamond')
    plt.yticks(font = 'garamond')
    plt.show()
    
    
def top5_subgraph(G):
    nagy = [top5 for top5 in [sorted([(edge[0], edge[1], edge[2]['weight']) for edge in G.edges.data() if edge[0] == node], key = lambda tup : tup[2], reverse = True)[0:5] for node in G.nodes] if len(top5) == 5]
    keep_edges = sum([[(a,b) for (a,b,c) in kicsi] for kicsi in nagy], [])
    return G.edge_subgraph(keep_edges)
    

def recalc_node_att(G):
    nx.set_node_attributes(G, {node : sum([edge[2]['weight'] for edge in G.edges.data() if edge[0] == node]) for node in G.nodes}, 'traffic_out')
    nx.set_node_attributes(G, {node : sum([edge[2]['weight'] for edge in G.edges.data() if edge[1] == node]) for node in G.nodes}, 'traffic_in')
    nx.set_node_attributes(G, {node : G.nodes[node]['traffic_in'] + G.nodes[node]['traffic_out'] for node in G.nodes}, 'total_traffic')
    nx.set_node_attributes(G, {node : G.nodes[node]['traffic_in'] / G.nodes[node]['total_traffic'] for node in G.nodes}, 'in_ratio')

    

    