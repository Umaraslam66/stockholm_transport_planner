from ..models.nodes import TransportNode
from ..models.connections import TransportConnection
from ..models.network import TransportNetwork

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