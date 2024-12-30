import networkx as nx
import pandas as pd
import numpy as np
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

class TransportNode:
    def __init__(self, id, name, type, lat, lon):
        self.id = id
        self.name = name
        self.type = type  # 'bus_stop', 'train_station', 'metro_station'
        self.lat = lat
        self.lon = lon

class TransportConnection:
    def __init__(self, from_node, to_node, mode, duration, frequency):
        self.from_node = from_node
        self.to_node = to_node
        self.mode = mode
        self.duration = duration  # in minutes
        self.frequency = frequency  # in minutes
        self.reliability = np.random.uniform(0.8, 1.0)  # historical reliability score

class TransportNetwork:
    def __init__(self):
        self.nodes = {}
        self.connections = []
        self.graph = nx.DiGraph()
        self.delay_model = None

    def add_node(self, node):
        self.nodes[node.id] = node
        self.graph.add_node(node.id, 
                           name=node.name, 
                           type=node.type,
                           pos=(node.lon, node.lat))

    def add_connection(self, connection):
        self.connections.append(connection)
        # Add edge with duration as weight
        self.graph.add_edge(connection.from_node, 
                           connection.to_node,
                           weight=connection.duration,
                           mode=connection.mode,
                           frequency=connection.frequency,
                           reliability=connection.reliability)

    def generate_dummy_data(self):
        # Generate dummy historical delay data for ML training
        delay_data = []
        for _ in range(1000):  # 1000 historical records
            hour = random.randint(0, 23)
            day = random.randint(0, 6)
            weather = random.choice(['clear', 'rain', 'snow'])
            connection = random.choice(self.connections)
            
            # Generate realistic delay based on factors
            base_delay = np.random.exponential(5)  # base delay in minutes
            weather_factor = 1.5 if weather != 'clear' else 1.0
            rush_hour_factor = 1.3 if hour in [8, 9, 17, 18] else 1.0
            
            delay = base_delay * weather_factor * rush_hour_factor
            
            delay_data.append({
                'hour': hour,
                'day': day,
                'weather': weather,
                'mode': connection.mode,
                'duration': connection.duration,
                'frequency': connection.frequency,
                'delay': delay
            })
        
        return pd.DataFrame(delay_data)

    def train_delay_model(self):
        df = self.generate_dummy_data()
        df_encoded = pd.get_dummies(df, columns=['weather', 'mode'])
        
        self.feature_columns = df_encoded.columns.drop('delay')  # Save the feature names
        
        X = df_encoded.drop('delay', axis=1)
        y = df_encoded['delay']
        
        self.delay_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.delay_model.fit(X, y)

    def predict_delay(self, hour, day, weather, mode, duration, frequency):
        if self.delay_model is None:
            raise ValueError("Model not trained yet!")
        
        input_data = pd.DataFrame({
            'hour': [hour],
            'day': [day],
            'duration': [duration],
            'frequency': [frequency]
        })
        
        weather_cols = ['weather_clear', 'weather_rain', 'weather_snow']
        mode_cols = ['mode_bus', 'mode_train', 'mode_metro']
        
        for col in weather_cols:
            input_data[col] = 1 if f'weather_{weather}' == col else 0
        
        for col in mode_cols:
            input_data[col] = 1 if f'mode_{mode}' == col else 0
        
        # Ensure input matches training features
        for feature in self.feature_columns:
            if feature not in input_data.columns:
                input_data[feature] = 0  # Add missing columns with default value 0
        
        input_data = input_data[self.feature_columns]  # Align columns to training order
        
        return self.delay_model.predict(input_data)[0]

    def find_optimal_route(self, start_node, end_node, current_time, weather='clear'):
        if not self.delay_model:
            self.train_delay_model()

        paths = list(nx.shortest_simple_paths(self.graph, start_node, end_node))
        best_routes = []

        for path in paths[:3]:  # Consider top 3 shortest paths
            total_duration = 0
            total_reliability = 1
            connections_info = []
            current_path_time = current_time

            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i+1]]
                
                # Predict delay
                hour = current_path_time.hour
                day = current_path_time.weekday()
                predicted_delay = self.predict_delay(
                    hour, day, weather,
                    edge_data['mode'],
                    edge_data['weight'],
                    edge_data['frequency']
                )

                # Calculate waiting time based on frequency
                avg_wait = edge_data['frequency'] / 2
                
                # Update total duration and time
                connection_duration = edge_data['weight'] + predicted_delay + avg_wait
                total_duration += connection_duration
                current_path_time += timedelta(minutes=connection_duration)
                
                # Update reliability
                total_reliability *= edge_data['reliability']

                connections_info.append({
                    'from': self.nodes[path[i]].name,
                    'to': self.nodes[path[i+1]].name,
                    'mode': edge_data['mode'],
                    'duration': edge_data['weight'],
                    'predicted_delay': predicted_delay,
                    'wait_time': avg_wait
                })

            best_routes.append({
                'path': path,
                'total_duration': total_duration,
                'reliability': total_reliability,
                'connections': connections_info,
                'arrival_time': current_time + timedelta(minutes=total_duration)
            })

        return sorted(best_routes, key=lambda x: x['total_duration'])

    def visualize_route(self, route):
        # Color mapping for different transport types
        node_colors = {
            'train_station': 'red',
            'metro_station': 'blue',
            'bus_stop': 'green'
        }
        
        mode_colors = {
            'train': 'red',
            'metro': 'blue',
            'bus': 'green'
        }

        # Create node trace with different colors based on type
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node_id in self.graph.nodes():
            node = self.nodes[node_id]
            node_x.append(node.lon)
            node_y.append(node.lat)
            node_text.append(f"{node.name}\n({node.type})")
            node_color.append(node_colors[node.type])

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=25,
                color=node_color,
                line=dict(width=2, color='black')
            ),
            text=node_text,
            textposition="top center",
            name='Stations'
        )

        # Create separate edge traces for each transport mode
        edge_traces = []
        path = route['path']
        
        for i in range(len(path) - 1):
            from_node = self.nodes[path[i]]
            to_node = self.nodes[path[i + 1]]
            mode = self.graph[path[i]][path[i + 1]]['mode']
            
            edge_trace = go.Scatter(
                x=[from_node.lon, to_node.lon],
                y=[from_node.lat, to_node.lat],
                mode='lines',
                line=dict(
                    width=3,
                    color=mode_colors[mode],
                    dash='solid'
                ),
                name=f'{mode} connection',
                showlegend=True
            )
            edge_traces.append(edge_trace)

        # Create figure with improved layout
        fig = go.Figure(
            data=[*edge_traces, node_trace],
            layout=go.Layout(
                title="Transport Network Route Visualization",
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(
                    showgrid=True,
                    zeroline=False,
                    title="Longitude"
                ),
                yaxis=dict(
                    showgrid=True,
                    zeroline=False,
                    title="Latitude"
                ),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
        )

        return fig

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

def test_planner():
    # Create and set up network
    network = create_sample_network()
    
    # Test a complex route
    start_time = datetime.now()
    routes = network.find_optimal_route('T-CEN', 'FJS', start_time, weather='clear')
    
    # Print results
    print("\nTop routes found:")
    for i, route in enumerate(routes):
        print(f"\nRoute {i+1}:")
        print(f"Total duration: {route['total_duration']:.1f} minutes")
        print(f"Reliability: {route['reliability']:.2f}")
        print(f"Expected arrival: {route['arrival_time']}")
        print("\nConnections:")
        for conn in route['connections']:
            print(f"  {conn['from']} -> {conn['to']} ({conn['mode']})")
            print(f"    Duration: {conn['duration']} min")
            print(f"    Predicted delay: {conn['predicted_delay']:.1f} min")
            print(f"    Average wait: {conn['wait_time']} min")
    
    # Visualize the best route
    fig = network.visualize_route(routes[0])
    fig.show()

# if __name__ == "__main__":
#     test_planner()

# Create the Dash app
app = dash.Dash(__name__)

# Initialize the network
network = create_sample_network()

# Get all station names and IDs for dropdowns
station_options = [{'label': node.name, 'value': node.id} 
                  for node_id, node in network.nodes.items()]

app.layout = html.Div([
    html.H1('Stockholm Transport Route Planner',
            style={'textAlign': 'center', 'marginBottom': 20}),
    
    html.Div([
        html.Div([
            html.Label('From Station:'),
            dcc.Dropdown(
                id='origin-dropdown',
                options=station_options,
                value='T-CEN'
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
        
        html.Div([
            html.Label('To Station:'),
            dcc.Dropdown(
                id='destination-dropdown',
                options=station_options,
                value='MED'
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label('Weather:'),
            dcc.Dropdown(
                id='weather-dropdown',
                options=[
                    {'label': 'Clear', 'value': 'clear'},
                    {'label': 'Rain', 'value': 'rain'},
                    {'label': 'Snow', 'value': 'snow'}
                ],
                value='clear'
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'})
    ], style={'marginBottom': 20}),
    
    # Route information
    html.Div(id='route-info', style={'marginBottom': 20}),
    
    # Map with routes
    dcc.Graph(id='route-map'),
])

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
