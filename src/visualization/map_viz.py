import plotly.graph_objects as go
import networkx as nx
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import random
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
from src.models.network import TransportNetwork
from src.models.nodes import TransportNode
from src.models.connections import TransportConnection
from src.visualization.dash_app import app, network


def create_sample_network():
    network = TransportNetwork()
    
    # Add nodes (stations and places in Stockholm)
    nodes = [
        # Major stations
        TransportNode('T-CEN', 'T-Centralen', 'train_station', 59.3308, 18.0583),
        TransportNode('STH', 'Stockholm City', 'train_station', 59.3303, 18.0560),
        TransportNode('SLU', 'Stockholm South', 'train_station', 59.3176, 18.0680),
        
        # Metro stations
        TransportNode('ODN', 'Odenplan', 'metro_station', 59.3424, 18.0480),
        TransportNode('FRI', 'Fridhemsplan', 'metro_station', 59.3324, 18.0328),
        TransportNode('GAM', 'Gamla Stan', 'metro_station', 59.3247, 18.0716),
        TransportNode('MED', 'Medborgarplatsen', 'metro_station', 59.3135, 18.0730),
        
        # Bus terminals
        TransportNode('TBG', 'Tegelbacken', 'bus_stop', 59.3293, 18.0651),
        TransportNode('HBG', 'Hornstull', 'bus_stop', 59.3167, 18.0325),
        TransportNode('NKS', 'Norrtull', 'bus_stop', 59.3480, 18.0442),
        TransportNode('VBG', 'Värtahamnen', 'bus_stop', 59.3442, 18.1028),
        
        # Local stops
        TransportNode('ZIN', 'Zinkensdamm', 'bus_stop', 59.3171, 18.0488),
        TransportNode('KTH', 'KTH Royal Institute', 'bus_stop', 59.3474, 18.0721),
        TransportNode('STU', 'Stureplan', 'bus_stop', 59.3346, 18.0727),
        TransportNode('FJS', 'Fjärilshuset', 'bus_stop', 59.3547, 18.0566)
    ]
    
    for node in nodes:
        network.add_node(node)
    
    # Add connections with approximate durations and frequencies
    connections = [
        # Train connections
        TransportConnection('T-CEN', 'STH', 'train', 2, 5),
        TransportConnection('STH', 'SLU', 'train', 5, 10),
        
        # Metro connections
        TransportConnection('T-CEN', 'ODN', 'metro', 4, 4),
        TransportConnection('T-CEN', 'FRI', 'metro', 3, 4),
        TransportConnection('T-CEN', 'GAM', 'metro', 2, 4),
        TransportConnection('GAM', 'MED', 'metro', 3, 4),
        TransportConnection('ODN', 'FRI', 'metro', 4, 4),
        
        # Cross metro connections
        TransportConnection('ODN', 'MED', 'metro', 12, 4),
        TransportConnection('FRI', 'GAM', 'metro', 7, 4),
        
        # Bus connections
        TransportConnection('TBG', 'ZIN', 'bus', 8, 6),
        TransportConnection('TBG', 'KTH', 'bus', 12, 6),
        TransportConnection('HBG', 'STU', 'bus', 10, 6),
        TransportConnection('NKS', 'FJS', 'bus', 15, 6),
        
        # Terminal-to-terminal bus
        TransportConnection('TBG', 'HBG', 'bus', 15, 8),
        TransportConnection('TBG', 'NKS', 'bus', 12, 8),
        TransportConnection('VBG', 'FJS', 'bus', 18, 8),
        
        # Metro to Bus Terminal
        TransportConnection('ODN', 'TBG', 'bus', 5, 6),
        TransportConnection('MED', 'HBG', 'bus', 6, 6),
        TransportConnection('FRI', 'VBG', 'bus', 10, 6)
    ]
    
    # Add reverse connections for all routes
    reverse_connections = []
    for conn in connections:
        reverse_connections.append(
            TransportConnection(
                conn.to_node, 
                conn.from_node, 
                conn.mode, 
                conn.duration, 
                conn.frequency
            )
        )
    
    # Add all connections to the network
    for conn in connections + reverse_connections:
        network.add_connection(conn)
    
    return network

def create_route_map(route):
    # Create a base map centered on Stockholm
    fig = go.Figure()
    
    # Add map background using Mapbox
    fig.update_layout(
        mapbox=dict(
            style='carto-positron',  # Light map style
            center=dict(lat=59.3293, lon=18.0686),  # Stockholm center
            zoom=12
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=700
    )
    
    # Add nodes (stations)
    node_colors = {
        'train_station': 'red',
        'metro_station': 'blue',
        'bus_stop': 'green'
    }
    
    # Create traces for each station type
    for station_type in node_colors:
        stations = [node for node in network.nodes.values() if node.type == station_type]
        if stations:
            fig.add_trace(go.Scattermapbox(
                lat=[s.lat for s in stations],
                lon=[s.lon for s in stations],
                mode='markers+text',
                marker=dict(size=12, color=node_colors[station_type]),
                text=[s.name for s in stations],
                textposition="top center",
                name=station_type.replace('_', ' ').title(),
                hoverinfo='text'
            ))
    
    # Add route path if provided
    if route:
        path = route['path']
        mode_colors = {'train': 'red', 'metro': 'blue', 'bus': 'green'}
        
        for i in range(len(path) - 1):
            from_node = network.nodes[path[i]]
            to_node = network.nodes[path[i + 1]]
            mode = network.graph[path[i]][path[i + 1]]['mode']
            
            fig.add_trace(go.Scattermapbox(
                lat=[from_node.lat, to_node.lat],
                lon=[from_node.lon, to_node.lon],
                mode='lines',
                line=dict(width=4, color=mode_colors[mode]),
                name=f'{mode} connection',
                hoverinfo='none'
            ))
    
    return fig

def create_route_info(routes):
    if not routes:
        return html.Div("No route found")
    
    route_elements = []
    for i, route in enumerate(routes[:3]):  # Show top 3 routes
        connections = html.Ul([
            html.Li([
                f"{conn['from']} → {conn['to']} ({conn['mode']})",
                html.Ul([
                    html.Li(f"Duration: {conn['duration']} min"),
                    html.Li(f"Predicted delay: {conn['predicted_delay']:.1f} min"),
                    html.Li(f"Average wait: {conn['wait_time']} min")
                ])
            ]) for conn in route['connections']
        ])
        
        route_elements.extend([
            html.H3(f"Route {i+1}:"),
            html.P([
                f"Total duration: {route['total_duration']:.1f} minutes",
                html.Br(),
                f"Reliability: {route['reliability']:.2f}",
                html.Br(),
                f"Expected arrival: {route['arrival_time'].strftime('%H:%M:%S')}"
            ]),
            html.H4("Connections:"),
            connections,
            html.Hr() if i < 2 else None
        ])
    
    return html.Div(route_elements)

@app.callback(
    [Output('route-map', 'figure'),
     Output('route-info', 'children')],
    [Input('origin-dropdown', 'value'),
     Input('destination-dropdown', 'value'),
     Input('weather-dropdown', 'value')]
)
def update_route(origin, destination, weather):
    if origin and destination:
        routes = network.find_optimal_route(origin, destination, datetime.now(), weather)
        if routes:
            return create_route_map(routes[0]), create_route_info(routes)
    
    return create_route_map(None), "Please select origin and destination stations"

if __name__ == '__main__':
    app.run_server(debug=True)