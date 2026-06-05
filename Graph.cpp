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
    size = sz;
    nodes.resize(size);
    adjacencyList.resize(size);
    for (int i = 0; i < size; i++) {
        nodes[i] = new NodeInfo();
    }
}

Graph::Graph(const Graph& other) {
    size = other.size;
    nodes.resize(size);
    adjacencyList.resize(size);
    for (int i = 0; i < size; i++) {
        nodes[i] = new NodeInfo(*(other.nodes[i]));
    }
    adjacencyList = other.adjacencyList;
}

Graph& Graph::operator=(const Graph& other) {
    if (this != &other) {
        clear();
        size = other.size;
        nodes.resize(size);
        adjacencyList.resize(size);
        for (int i = 0; i < size; i++) {
            nodes[i] = new NodeInfo(*(other.nodes[i]));
        }
        adjacencyList = other.adjacencyList;
    }
    return *this;
}

Graph::~Graph() {
    clear();
}

void Graph::updateNode(int id, NodeInfo n) {
    if (id >= 0 && id < size) {
        *(nodes[id]) = n;
    }
}

NodeInfo* Graph::getNode(int id) const {
    if (id >= 0 && id < size) {
        return nodes[id];
    }
    return nullptr;
}

void Graph::updateConnection(int v, int u, double w) {
    if (v >= 0 && v < size && u >= 0 && u < size) {
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
        delete nodes[i];
    }
    nodes.clear();
    adjacencyList.clear();
    size = 0;
}

void Graph::resize(int sz) {
    clear();
    size = sz;
    nodes.resize(size);
    adjacencyList.resize(size);
    for (int i = 0; i < size; i++) {
        nodes[i] = new NodeInfo();
    }
}

std::ostream& operator<<(std::ostream& out, const Graph& g) {
    for (int i = 0; i < g.size; i++) {
        out << "Node " << i << ": " << *(g.nodes[i]) << std::endl;
        for (auto& pair : g.adjacencyList[i]) {
            out << "\t-> " << pair.second << std::endl;
        }
    }
    return out;
}