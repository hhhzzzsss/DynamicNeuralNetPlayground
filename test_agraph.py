import unittest
from dnn_libs.agraph import AcyclicGraph

class TestAcyclicGraph(unittest.TestCase):
    def setUp(self):
        self.graph = AcyclicGraph()

    def test_create_node(self):
        """Test node creation with hunger and utility values."""
        node_id = self.graph.add_node(0.5, 0.7)
        self.assertIsInstance(node_id, int)
        self.assertEqual(self.graph.get_node_hunger(node_id), 0.5)
        self.assertEqual(self.graph.get_node_utility(node_id), 0.7)
    
    def test_modify_node_properties(self):
        """Test modifying node hunger and utility values."""
        node_id = self.graph.add_node(0.1, 0.2)
        
        self.graph.set_node_hunger(node_id, 0.3)
        self.assertEqual(self.graph.get_node_hunger(node_id), 0.3)
        
        self.graph.set_node_utility(node_id, 0.4)
        self.assertEqual(self.graph.get_node_utility(node_id), 0.4)
    
    def test_remove_node(self):
        """Test removing a node."""
        node_id = self.graph.add_node()
        self.assertTrue(self.graph.remove_node(node_id))
        
        # Node shouldn't appear in nodes list anymore
        self.assertNotIn(node_id, self.graph.get_nodes())
    
    def test_add_edge(self):
        """Test adding edges with weights."""
        node1 = self.graph.add_node()
        node2 = self.graph.add_node()
        
        self.assertTrue(self.graph.add_edge(node1, node2, 0.5))
        self.assertEqual(self.graph.get_edge_weight(node1, node2), 0.5)
        
        # Edge should appear in outgoing edges for node1
        out_edges = self.graph.get_out_edges(node1)
        self.assertEqual(len(out_edges), 1)
        self.assertEqual(out_edges[0][0], node2)
        self.assertEqual(out_edges[0][1], 0.5)
        
        # Edge should appear in incoming edges for node2
        in_edges = self.graph.get_in_edges(node2)
        self.assertEqual(len(in_edges), 1)
        self.assertEqual(in_edges[0][0], node1)
        self.assertEqual(in_edges[0][1], 0.5)
    
    def test_remove_edge(self):
        """Test removing an edge."""
        node1 = self.graph.add_node()
        node2 = self.graph.add_node()
        
        self.graph.add_edge(node1, node2)
        self.assertTrue(self.graph.remove_edge(node1, node2))
        
        # Edge shouldn't exist anymore
        with self.assertRaises(RuntimeError):
            self.graph.get_edge_weight(node1, node2)
    
    def test_set_edge_weight(self):
        """Test setting edge weight."""
        node1 = self.graph.add_node()
        node2 = self.graph.add_node()
        
        self.graph.add_edge(node1, node2, 1.0)
        self.assertTrue(self.graph.set_edge_weight(node1, node2, 2.0))
        self.assertEqual(self.graph.get_edge_weight(node1, node2), 2.0)
    
    def test_has_path(self):
        """Test path detection between nodes."""
        node1 = self.graph.add_node()
        node2 = self.graph.add_node()
        node3 = self.graph.add_node()
        
        self.graph.add_edge(node1, node2)
        self.graph.add_edge(node2, node3)
        
        self.assertTrue(self.graph.has_path(node1, node3))
        self.assertFalse(self.graph.has_path(node3, node1))
    
    def test_cycle_detection(self):
        """Test that cycles are rejected."""
        node1 = self.graph.add_node()
        node2 = self.graph.add_node()
        node3 = self.graph.add_node()
        
        # Create path node1 -> node2 -> node3
        self.graph.add_edge(node1, node2)
        self.graph.add_edge(node2, node3)
        
        # Attempt to create cycle by connecting node3 -> node1
        self.assertFalse(self.graph.add_edge(node3, node1))
    
    def test_topological_order(self):
        """Test that topological ordering is maintained."""
        node1 = self.graph.add_node()
        node2 = self.graph.add_node()
        node3 = self.graph.add_node()
        
        self.graph.add_edge(node1, node2)
        self.graph.add_edge(node1, node3)
        
        topo_order = self.graph.get_topological_order()
        
        # Check node1 comes before node2 and node3
        idx1 = topo_order.index(node1)
        idx2 = topo_order.index(node2)
        idx3 = topo_order.index(node3)
        
        self.assertLess(idx1, idx2)
        self.assertLess(idx1, idx3)

if __name__ == "__main__":
    unittest.main()

