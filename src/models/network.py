import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from .nodes import TransportNode
from .connections import TransportConnection
import random

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