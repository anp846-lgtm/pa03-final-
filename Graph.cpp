#include "Graph.hpp"

// ==================== NodeInfo ====================

NodeInfo::NodeInfo() {
    activationFunction = identity;
    activationDerivative = identity_prime;
    preActivationValue = 0;
    postActivationValue = 0;
    bias = 0;
    delta = 0;
}

NodeInfo::NodeInfo(std::string activationFunc, double value, double b) {
    activationFunction = getActivationFunction(activationFunc);
    activationDerivative = getActivationDerivative(activationFunc);
    preActivationValue = value;
    postActivationValue = activationFunction(value);
    bias = b;
    delta = 0;
}

bool NodeInfo::operator==(const NodeInfo& other) {
    return (activationFunction == other.activationFunction &&
            abs(preActivationValue - other.preActivationValue) < 0.00001 &&
            abs(postActivationValue - other.postActivationValue) < 0.00001 &&
            abs(bias - other.bias) < 0.00001);
}

std::ostream& operator<<(std::ostream& out, const NodeInfo& n) {
    out << getActivationIdentifier(n.activationFunction)
        << " " << n.preActivationValue
        << " " << n.bias;
    return out;
}

double NodeInfo::activate() {
    postActivationValue = activationFunction(preActivationValue);
    return postActivationValue;
}

double NodeInfo::derive() {
    return activationDerivative(preActivationValue);
}

// ==================== Connection ====================

Connection::Connection() {
    source = 0;
    dest = 0;
    weight = 0;
    delta = 0;
}

Connection::Connection(int s, int d, double w) {
    source = s;
    dest = d;
    weight = w;
    delta = 0;
}

bool Connection::operator<(const Connection& other) {
    return dest < other.dest;
}

bool Connection::operator==(const Connection& other) {
    return (source == other.source &&
            dest == other.dest &&
            abs(weight - other.weight) < 0.00001);
}

std::ostream& operator<<(std::ostream& out, const Connection& c) {
    out << c.source << " " << c.dest << " " << c.weight;
    return out;
}

// ==================== Graph ====================

Graph::Graph() {
    size = 0;
}

Graph::Graph(int sz) {
    size = 0;
    resize(sz);
}

Graph::Graph(const Graph& other) {
    size = 0;
    *this = other;
}

Graph& Graph::operator=(const Graph& other) {
    if (this != &other) {
        clear();
        size = other.size;
        nodes.assign(size, nullptr);
        adjacencyList.assign(size, std::unordered_map<int, Connection>());
        for (int i = 0; i < size; i++) {
            if (other.nodes[i] != nullptr) {
                nodes[i] = new NodeInfo(*(other.nodes[i]));
            }
        }
        adjacencyList = other.adjacencyList;
    }
    return *this;
}

Graph::~Graph() {
    clear();
}

// Allocates a NodeInfo object on the heap and stores it at index id.
void Graph::updateNode(int id, NodeInfo n) {
    if (id >= 0 && id < (int)nodes.size()) {
        if (nodes[id] != nullptr) {
            delete nodes[id];
        }
        nodes[id] = new NodeInfo(n);
    }
}

NodeInfo* Graph::getNode(int id) const {
    if (id >= 0 && id < (int)nodes.size()) {
        return nodes[id];
    }
    return nullptr;
}

// Adds/updates a directed weighted edge from v to u.
void Graph::updateConnection(int v, int u, double w) {
    if (v >= 0 && v < (int)adjacencyList.size() && u >= 0) {
        adjacencyList[v][u] = Connection(v, u, w);
    }
}

AdjList& Graph::getAdjacencyList() {
    return adjacencyList;
}

std::vector<NodeInfo*> Graph::getNodes() const {
    return nodes;
}

void Graph::clear() {
    for (int i = 0; i < (int)nodes.size(); i++) {
        if (nodes[i] != nullptr) {
            delete nodes[i];
            nodes[i] = nullptr;
        }
    }
    nodes.clear();
    adjacencyList.clear();
    size = 0;
}

// resize initializes nodes to a vector of `sz` nullptrs and adjacencyList
// to a vector of `sz` empty maps.
void Graph::resize(int sz) {
    clear();
    size = sz;
    nodes.assign(size, nullptr);
    adjacencyList.assign(size, std::unordered_map<int, Connection>());
}

std::ostream& operator<<(std::ostream& out, const Graph& g) {
    for (int i = 0; i < (int)g.nodes.size(); i++) {
        out << "Node " << i << ": ";
        if (g.nodes[i] != nullptr) {
            out << *(g.nodes[i]);
        }
        out << std::endl;
        for (auto& pair : g.adjacencyList[i]) {
            out << "\t-> " << pair.second << std::endl;
        }
    }
    return out;
}