#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <algorithm>

// Fixed NodeData structure with hunger and utility
struct NodeData {
    float hunger;
    float utility;
    
    NodeData(float h = 0.0f, float u = 0.0f) : hunger(h), utility(u) {}
};

template <typename WeightType = float>
class AcyclicGraph {
public:
    using NodeId = int;
    struct Edge {
        NodeId nodeId;
        WeightType weight;
        
        Edge(NodeId id, WeightType w) : nodeId(id), weight(w) {}
        
        bool operator==(const NodeId& other) const {
            return nodeId == other;
        }
    };

    // Add a new node with hunger and utility values
    NodeId addNode(float hunger = 0.0f, float utility = 0.0f) {
        NodeId id = nextNodeId++;
        nodes[id] = NodeData(hunger, utility);
        outEdges[id] = {};
        inEdges[id] = {};
        
        // Add to topological order
        topoOrder.push_back(id);
        return id;
    }
    
    // Get node hunger value
    float getNodeHunger(NodeId id) const {
        return nodes.at(id).hunger;
    }
    
    // Set node hunger value
    void setNodeHunger(NodeId id, float hunger) {
        nodes.at(id).hunger = hunger;
    }
    
    // Get node utility value
    float getNodeUtility(NodeId id) const {
        return nodes.at(id).utility;
    }
    
    // Set node utility value
    void setNodeUtility(NodeId id, float utility) {
        nodes.at(id).utility = utility;
    }
    
    // Remove a node and all its connected edges
    bool removeNode(NodeId id) {
        if (nodes.find(id) == nodes.end()) return false;
        
        // Remove all associated edges
        for (const Edge& edge : outEdges[id]) {
            NodeId to = edge.nodeId;
            auto& ins = inEdges[to];
            ins.erase(std::remove(ins.begin(), ins.end(), id), ins.end());
        }
        
        for (const Edge& edge : inEdges[id]) {
            NodeId from = edge.nodeId;
            auto& outs = outEdges[from];
            outs.erase(std::remove(outs.begin(), outs.end(), id), outs.end());
        }
        
        // Remove node from all structures
        nodes.erase(id);
        outEdges.erase(id);
        inEdges.erase(id);
        topoOrder.erase(std::remove(topoOrder.begin(), topoOrder.end(), id), topoOrder.end());
        
        return true;
    }
    
    // Add edge with cycle detection and weight
    bool addEdge(NodeId from, NodeId to, WeightType weight = 1.0) {
        // Validate nodes
        if (nodes.find(from) == nodes.end() || nodes.find(to) == nodes.end())
            return false;
            
        // Check for self-loop
        if (from == to) return false;
        
        // Check if edge already exists
        auto it = std::find_if(outEdges[from].begin(), outEdges[from].end(),
                             [to](const Edge& edge) { return edge.nodeId == to; });
                             
        if (it != outEdges[from].end()) {
            // Update weight if edge exists
            it->weight = weight;
            
            // Also update the corresponding in-edge weight
            auto inIt = std::find_if(inEdges[to].begin(), inEdges[to].end(),
                                  [from](const Edge& edge) { return edge.nodeId == from; });
            if (inIt != inEdges[to].end()) {
                inIt->weight = weight;
            }
            return true;
        }
        
        // Check if adding this edge would create a cycle
        if (wouldCreateCycle(from, to))
            return false;
        
        // Add edge with weight
        outEdges[from].push_back(Edge(to, weight));
        inEdges[to].push_back(Edge(from, weight));
        
        // Update topological ordering
        updateTopologicalOrder();
        return true;
    }
    
    // Remove edge
    bool removeEdge(NodeId from, NodeId to) {
        if (outEdges.find(from) == outEdges.end())
            return false;
            
        auto& outs = outEdges[from];
        auto it = std::find_if(outs.begin(), outs.end(),
                            [to](const Edge& edge) { return edge.nodeId == to; });
                            
        if (it == outs.end())
            return false;
            
        // Remove the edge
        outs.erase(it);
        
        // Remove the corresponding in-edge
        auto& ins = inEdges[to];
        ins.erase(std::find_if(ins.begin(), ins.end(),
                            [from](const Edge& edge) { return edge.nodeId == from; }));
        
        return true;
    }
    
    // Get edge weight
    WeightType getEdgeWeight(NodeId from, NodeId to) const {
        auto outIt = outEdges.find(from);
        if (outIt != outEdges.end()) {
            auto edgeIt = std::find_if(outIt->second.begin(), outIt->second.end(),
                                    [to](const Edge& edge) { return edge.nodeId == to; });
            if (edgeIt != outIt->second.end()) {
                return edgeIt->weight;
            }
        }
        throw std::runtime_error("Edge not found");
    }
    
    // Set edge weight for existing edge
    bool setEdgeWeight(NodeId from, NodeId to, WeightType weight) {
        auto outIt = outEdges.find(from);
        if (outIt != outEdges.end()) {
            auto edgeIt = std::find_if(outIt->second.begin(), outIt->second.end(),
                                    [to](const Edge& edge) { return edge.nodeId == to; });
            if (edgeIt != outIt->second.end()) {
                edgeIt->weight = weight;
                
                // Also update the corresponding in-edge
                auto& ins = inEdges[to];
                auto inEdgeIt = std::find_if(ins.begin(), ins.end(),
                                         [from](const Edge& edge) { return edge.nodeId == from; });
                if (inEdgeIt != ins.end()) {
                    inEdgeIt->weight = weight;
                }
                return true;
            }
        }
        return false;
    }
    
    // Check if path exists between nodes
    bool hasPath(NodeId source, NodeId target) const {
        if (nodes.find(source) == nodes.end() || nodes.find(target) == nodes.end())
            return false;
        
        std::unordered_set<NodeId> visited;
        return hasPathDFS(source, target, visited);
    }
    
    // Get current topological ordering
    std::vector<NodeId> getTopologicalOrder() const {
        return topoOrder;
    }
    
    // Get all nodes
    std::vector<NodeId> getNodes() const {
        std::vector<NodeId> nodeIds;
        for (const auto& pair : nodes) {
            nodeIds.push_back(pair.first);
        }
        return nodeIds;
    }
    
    // Get outgoing edges with weights for a node
    std::vector<std::pair<NodeId, WeightType>> getOutEdges(NodeId id) const {
        std::vector<std::pair<NodeId, WeightType>> result;
        auto it = outEdges.find(id);
        if (it != outEdges.end()) {
            for (const Edge& edge : it->second) {
                result.push_back({edge.nodeId, edge.weight});
            }
        }
        return result;
    }
    
    // Get incoming edges with weights for a node
    std::vector<std::pair<NodeId, WeightType>> getInEdges(NodeId id) const {
        std::vector<std::pair<NodeId, WeightType>> result;
        auto it = inEdges.find(id);
        if (it != inEdges.end()) {
            for (const Edge& edge : it->second) {
                result.push_back({edge.nodeId, edge.weight});
            }
        }
        return result;
    }
    
private:
    NodeId nextNodeId = 0;
    std::unordered_map<NodeId, NodeData> nodes;                  // Node storage
    std::unordered_map<NodeId, std::vector<Edge>> outEdges;      // Forward edges with weights
    std::unordered_map<NodeId, std::vector<Edge>> inEdges;       // Backward edges with weights
    std::vector<NodeId> topoOrder;                               // Cached topological order
    
    // Cycle detection using DFS
    bool wouldCreateCycle(NodeId from, NodeId to) const {
        // If adding edge from->to creates cycle, then there must be a path from to->from
        return hasPath(to, from);
    }
    
    // DFS for path finding
    bool hasPathDFS(NodeId current, NodeId target, std::unordered_set<NodeId>& visited) const {
        if (current == target) return true;
        
        visited.insert(current);
        
        const auto& neighbors = outEdges.find(current);
        if (neighbors != outEdges.end()) {
            for (const Edge& edge : neighbors->second) {
                NodeId next = edge.nodeId;
                if (visited.find(next) == visited.end()) {
                    if (hasPathDFS(next, target, visited))
                        return true;
                }
            }
        }
        
        return false;
    }
    
    // Kahn's algorithm for topological sort
    void updateTopologicalOrder() {
        // Calculate in-degrees
        std::unordered_map<NodeId, int> inDegree;
        for (const auto& entry : nodes) {
            inDegree[entry.first] = 0;
        }
        
        for (const auto& entry : outEdges) {
            for (const Edge& edge : entry.second) {
                inDegree[edge.nodeId]++;
            }
        }
        
        // Find nodes with no incoming edges
        std::queue<NodeId> q;
        for (const auto& entry : inDegree) {
            if (entry.second == 0) {
                q.push(entry.first);
            }
        }
        
        // Process nodes in topological order
        std::vector<NodeId> newOrder;
        while (!q.empty()) {
            NodeId current = q.front();
            q.pop();
            
            newOrder.push_back(current);
            
            for (const Edge& edge : outEdges[current]) {
                NodeId next = edge.nodeId;
                inDegree[next]--;
                if (inDegree[next] == 0) {
                    q.push(next);
                }
            }
        }
        
        topoOrder = std::move(newOrder);
    }
};

namespace nb = nanobind;

void bind_acyclic_graph(nb::module_& m) {
    // Bind NodeData struct
    nb::class_<NodeData>(m, "NodeData")
        .def(nb::init<float, float>())
        .def_rw("hunger", &NodeData::hunger)
        .def_rw("utility", &NodeData::utility);

    // Bind AcyclicGraph class with fixed node type
    nb::class_<AcyclicGraph<float>>(m, "AcyclicGraph")
        .def(nb::init<>())
        .def("add_node", &AcyclicGraph<float>::addNode, 
             nb::arg("hunger") = 0.0f, nb::arg("utility") = 0.0f)
        .def("get_node_hunger", &AcyclicGraph<float>::getNodeHunger)
        .def("set_node_hunger", &AcyclicGraph<float>::setNodeHunger)
        .def("get_node_utility", &AcyclicGraph<float>::getNodeUtility)
        .def("set_node_utility", &AcyclicGraph<float>::setNodeUtility)
        .def("remove_node", &AcyclicGraph<float>::removeNode)
        .def("add_edge", &AcyclicGraph<float>::addEdge, 
             nb::arg("from"), nb::arg("to"), nb::arg("weight") = 1.0f)
        .def("remove_edge", &AcyclicGraph<float>::removeEdge)
        .def("get_edge_weight", &AcyclicGraph<float>::getEdgeWeight)
        .def("set_edge_weight", &AcyclicGraph<float>::setEdgeWeight)
        .def("has_path", &AcyclicGraph<float>::hasPath)
        .def("get_topological_order", &AcyclicGraph<float>::getTopologicalOrder)
        .def("get_nodes", &AcyclicGraph<float>::getNodes)
        .def("get_out_edges", &AcyclicGraph<float>::getOutEdges)
        .def("get_in_edges", &AcyclicGraph<float>::getInEdges);
}

NB_MODULE(agraph, m) {
    bind_acyclic_graph(m);
}