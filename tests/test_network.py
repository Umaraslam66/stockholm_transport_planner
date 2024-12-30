import unittest
from src.models.network import TransportNetwork
from src.models.nodes import TransportNode
from src.models.connections import TransportConnection

class TestTransportNetwork(unittest.TestCase):
    def setUp(self):
        self.network = TransportNetwork()
        
    def test_add_node(self):
        node = TransportNode('T1', 'Test Station', 'train_station', 59.3293, 18.0686)
        self.network.add_node(node)
        self.assertIn('T1', self.network.nodes)
        
    def test_add_connection(self):
        node1 = TransportNode('T1', 'Test Station 1', 'train_station', 59.3293, 18.0686)
        node2 = TransportNode('T2', 'Test Station 2', 'train_station', 59.3294, 18.0687)
        self.network.add_node(node1)
        self.network.add_node(node2)
        
        connection = TransportConnection('T1', 'T2', 'train', 10, 5)
        self.network.add_connection(connection)
        self.assertEqual(len(self.network.connections), 1)

if __name__ == '__main__':
    unittest.main()