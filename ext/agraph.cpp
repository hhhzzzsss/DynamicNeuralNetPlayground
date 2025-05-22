#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>
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

using WeightType = float;
using NodeId = uint32_t;
struct EdgeId {
    uint64_t value;
    
    EdgeId(NodeId source, NodeId target) 
        : value((static_cast<uint64_t>(source) << 32) | static_cast<uint64_t>(target)) {}

    operator uint64_t() const { return value; }
    
    NodeId source() const { return static_cast<NodeId>(value >> 32); }
    NodeId target() const { return static_cast<NodeId>(value & 0xFFFFFFFF); }
    
    bool operator==(const EdgeId& other) const { return value == other.value; }
};
template<>
struct std::hash<EdgeId> {
    std::size_t operator()(const EdgeId& id) const {
        return std::hash<uint64_t>{}(id.value);
    }
};

class AcyclicGraph {
public:
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
        topoOrderIndices[id] = topoOrder.size() - 1;
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
    void removeNode(NodeId id) {
        // Check if the node exists
        if (nodes.find(id) == nodes.end())
            throw std::runtime_error("Node not found");
            
        // Remove all incoming edges to this node
        for (const NodeId& source : inEdges[id]) {
            auto& sourceOutEdges = outEdges[source];
            sourceOutEdges.erase(std::remove(sourceOutEdges.begin(), sourceOutEdges.end(), id), sourceOutEdges.end());
            edgeWeights.erase({source, id});
        }

        // Remove all outgoing edges from this node
        for (const NodeId& target : outEdges[id]) {
            auto& targetInEdges = inEdges[target];
            targetInEdges.erase(std::remove(targetInEdges.begin(), targetInEdges.end(), id), targetInEdges.end());
            edgeWeights.erase({id, target});
        }

        // Remove node data structures
        nodes.erase(id);
        inEdges.erase(id);
        outEdges.erase(id);

        // Remove the node from the topological order
        topoOrder.erase(topoOrder.begin() + topoOrderIndices[id]);

        // Update the indices in the topological order
        updateTopologicalIndices();
    }
    
    // Add edge with cycle detection and weight
    void addEdge(NodeId from, NodeId to, WeightType weight = 1.0) {
        // Validate nodes
        if (nodes.find(from) == nodes.end() || nodes.find(to) == nodes.end())
            throw std::runtime_error("Node not found");
            
        // Check for self-loop
        if (from == to) 
            throw std::runtime_error("Self-loops are not allowed");
                             
        // Check if edge already exists
        if (edgeWeights.count({from, to})) {
            throw std::runtime_error("Edge already exists");
        }
        
        // Check if adding this edge would create a cycle
        if (wouldCreateCycle(from, to))
            return;
            // throw std::runtime_error("Adding this edge would create a cycle");
        
        // Add edge with weight
        outEdges[from].push_back(to);
        inEdges[to].push_back(from);
        edgeWeights[{from, to}] = weight;
        
        // Update topological ordering
        updateTopologicalOrder();
        updateTopologicalIndices();
    }
    
    // Remove edge
    void removeEdge(NodeId from, NodeId to) {
        // Check if nodes exist
        if (nodes.find(from) == nodes.end() || nodes.find(to) == nodes.end())
            throw std::runtime_error("Node not found");
        
        // Check if the edge exists
        EdgeId edgeId(from, to);
        if (!edgeWeights.count(edgeId))
            throw std::runtime_error("Edge not found");
        
        // Remove edge from outEdges
        auto& outs = outEdges[from];
        outs.erase(std::remove(outs.begin(), outs.end(), to), outs.end());
        
        // Remove edge from inEdges
        auto& ins = inEdges[to];
        ins.erase(std::remove(ins.begin(), ins.end(), from), ins.end());
        
        // Remove from edge weights
        edgeWeights.erase(edgeId);
        
        // No need to update topological order since removing edges
        // cannot create cycles or alter the valid ordering
    }
    
    // Get edge weight
    WeightType getEdgeWeight(NodeId from, NodeId to) const {
        auto it = edgeWeights.find({from, to});
        if (it != edgeWeights.end())
            return it->second;
        else
            throw std::runtime_error("Edge not found");
    }
    
    // Set edge weight for existing edge
    void setEdgeWeight(NodeId from, NodeId to, WeightType weight) {
        // Check if the edge exists
        EdgeId edgeId(from, to);
        if (!edgeWeights.count(edgeId))
            throw std::runtime_error("Edge not found");

        // Update the weight
        edgeWeights[edgeId] = weight;
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
    
    // Get the index of a node in the topological order
    size_t getTopologicalIndex(NodeId id) const {
        auto it = topoOrderIndices.find(id);
        if (it != topoOrderIndices.end()) {
            return it->second;
        } else {
            throw std::runtime_error("Node not found in topological ordering");
        }
    }

    // Get a vector of edges in order
    std::vector<std::tuple<NodeId, NodeId, WeightType>> getSortedEdges() const {
        std::vector<std::tuple<NodeId, NodeId, WeightType>> edges;
        for (const NodeId& node : topoOrder) {
            for (const NodeId& neighbor : inEdges.at(node)) {
                auto it = edgeWeights.find({neighbor, node});
                if (it != edgeWeights.end()) {
                    edges.push_back(std::make_tuple(neighbor, node, it->second));
                } else {
                    throw std::runtime_error("Invalid agraph edge state");
                }
            }
        }
        return edges;
    }

    // Update the weights of the edges
    void setEdgeWeights(const std::vector<std::tuple<NodeId, NodeId, WeightType>>& edges) {
        for (const auto& edge : edges) {
            NodeId from = std::get<0>(edge);
            NodeId to = std::get<1>(edge);
            WeightType weight = std::get<2>(edge);

            EdgeId edgeId(from, to);
            if (!edgeWeights.count(edgeId)) {
                throw std::runtime_error("Edge not found: from=" + std::to_string(from) + " to=" + std::to_string(to));
            }
            
            // Update the weight
            edgeWeights[edgeId] = weight;
        }
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
    std::vector<NodeId> getOutNeighbors(NodeId id) const {
        std::vector<NodeId> result = outEdges.at(id);
        return result;
    }
    
    // Get incoming edges with weights for a node
    std::vector<NodeId> getInNeighbors(NodeId id) const {
        std::vector<NodeId> result = inEdges.at(id);
        return result;
    }

    // Get out-degree of a node
    size_t getOutDegree(NodeId id) const {
        return outEdges.at(id).size();
    }

    // Get in-degree of a node
    size_t getInDegree(NodeId id) const {
        return inEdges.at(id).size();
    }

    // Get the number of nodes in the graph
    size_t getNumNodes() const {
        return nodes.size();
    }

    // Get the number of edges in the graph
    size_t getNumEdges() const {
        return edgeWeights.size();
    }
    
private:
    NodeId nextNodeId = 0;
    std::unordered_map<NodeId, NodeData> nodes;
    std::unordered_map<NodeId, std::vector<NodeId>> outEdges;
    std::unordered_map<NodeId, std::vector<NodeId>> inEdges;
    std::unordered_map<EdgeId, WeightType> edgeWeights;
    std::vector<NodeId> topoOrder;
    std::unordered_map<NodeId, size_t> topoOrderIndices;
    
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
            for (const NodeId& next : neighbors->second) {
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
            for (const NodeId& targetNode : entry.second) {
                inDegree[targetNode]++;
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
            
            for (const NodeId& next : outEdges[current]) {
                inDegree[next]--;
                if (inDegree[next] == 0) {
                    q.push(next);
                }
            }
        }
        
        topoOrder = std::move(newOrder);
    }

    // Update indices for topological order
    void updateTopologicalIndices() {
        topoOrderIndices.clear();
        for (size_t i = 0; i < topoOrder.size(); ++i) {
            topoOrderIndices[topoOrder[i]] = i;
        }
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
    nb::class_<AcyclicGraph>(m, "AcyclicGraph")
        .def(nb::init<>())
        .def("add_node", &AcyclicGraph::addNode, 
             nb::arg("hunger") = 0.0f, nb::arg("utility") = 0.0f)
        .def("get_node_hunger", &AcyclicGraph::getNodeHunger)
        .def("set_node_hunger", &AcyclicGraph::setNodeHunger)
        .def("get_node_utility", &AcyclicGraph::getNodeUtility)
        .def("set_node_utility", &AcyclicGraph::setNodeUtility)
        .def("remove_node", &AcyclicGraph::removeNode)
        .def("add_edge", &AcyclicGraph::addEdge, 
             nb::arg("from"), nb::arg("to"), nb::arg("weight") = 1.0f)
        .def("remove_edge", &AcyclicGraph::removeEdge)
        .def("get_edge_weight", &AcyclicGraph::getEdgeWeight)
        .def("set_edge_weight", &AcyclicGraph::setEdgeWeight)
        .def("has_path", &AcyclicGraph::hasPath)
        .def("get_topological_order", &AcyclicGraph::getTopologicalOrder)
        .def("get_topological_index", &AcyclicGraph::getTopologicalIndex)
        .def("get_sorted_edges", &AcyclicGraph::getSortedEdges)
        .def("set_edge_weights", &AcyclicGraph::setEdgeWeights)
        .def("get_nodes", &AcyclicGraph::getNodes)
        .def("get_out_neighbors", &AcyclicGraph::getOutNeighbors)
        .def("get_in_neighbors", &AcyclicGraph::getInNeighbors)
        .def("get_out_degree", &AcyclicGraph::getOutDegree)
        .def("get_in_degree", &AcyclicGraph::getInDegree)
        .def("get_num_nodes", &AcyclicGraph::getNumNodes)
        .def("get_num_edges", &AcyclicGraph::getNumEdges);
}

NB_MODULE(agraph, m) {
    bind_acyclic_graph(m);
}