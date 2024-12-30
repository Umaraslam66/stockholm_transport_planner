import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from datetime import datetime
from .map_viz import create_route_map, create_route_info
from ..data.stockholm_network import create_sample_network



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
