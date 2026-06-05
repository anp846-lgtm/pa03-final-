#include "NeuralNetwork.hpp"
#include "Trace.hpp"
#include <algorithm>
#include <iomanip>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

// ==================== Constructors ====================

NeuralNetwork::NeuralNetwork() : Graph() {
    evaluating = true;
    learningRate = 0.1;
    batchSize = 0;
}

NeuralNetwork::NeuralNetwork(int size) : Graph(size) {
    evaluating = true;
    learningRate = 0.1;
    batchSize = 0;
}

NeuralNetwork::NeuralNetwork(std::string filename) : Graph() {
    evaluating = true;
    learningRate = 0.1;
    batchSize = 0;
    std::ifstream fin(filename);
    if (fin.is_open()) {
        loadNetwork(fin);
        fin.close();
    }
}

NeuralNetwork::NeuralNetwork(std::istream& in) : Graph() {
    evaluating = true;
    learningRate = 0.1;
    batchSize = 0;
    loadNetwork(in);
}

// ==================== Getters / Setters ====================

const std::vector<std::vector<int>>& NeuralNetwork::getLayers() const {
    return layers;
}

void NeuralNetwork::eval() { evaluating = true; }
void NeuralNetwork::train() { evaluating = false; }
void NeuralNetwork::setLearningRate(double lr) { learningRate = lr; }
void NeuralNetwork::setInputNodeIds(std::vector<int> ids) { inputNodeIds = ids; }
void NeuralNetwork::setOutputNodeIds(std::vector<int> ids) { outputNodeIds = ids; }
std::vector<int> NeuralNetwork::getInputNodeIds() const { return inputNodeIds; }
std::vector<int> NeuralNetwork::getOutputNodeIds() const { return outputNodeIds; }

// ==================== Visit Helpers (Predict) ====================

void NeuralNetwork::visitPredictNode(int vId) {
    nodes[vId]->preActivationValue += nodes[vId]->bias;
    nodes[vId]->activate();
}

void NeuralNetwork::visitPredictNeighbor(Connection c) {
    nodes[c.dest]->preActivationValue += nodes[c.source]->postActivationValue * c.weight;
}

// ==================== Visit Helpers (Contribute / Backprop) ====================

void NeuralNetwork::visitContributeStart(int vId) {
    // no-op
}

void NeuralNetwork::visitContributeNode(int vId, double& outgoingContribution) {
    outgoingContribution *= nodes[vId]->derive();
    nodes[vId]->delta += outgoingContribution;
}

void NeuralNetwork::visitContributeNeighbor(Connection& c, double& incomingContribution, double& outgoingContribution) {
    c.delta += incomingContribution * nodes[c.source]->postActivationValue;
    outgoingContribution += incomingContribution * c.weight;
}

// ==================== Predict (Forward Pass) ====================

std::vector<double> NeuralNetwork::predict(DataInstance instance) {
    flush();

    // Set input node values
    for (size_t i = 0; i < inputNodeIds.size(); i++) {
        nodes[inputNodeIds[i]]->preActivationValue = instance.x[i];
    }

    // Forward pass: for file-loaded networks, node IDs are assigned sequentially
    // by layer in loadNetwork, so simple ID-order iteration is topological order.
    // For manually-built networks without layers, use BFS.
    if (!layers.empty()) {
        for (int i = 0; i < size; i++) {
            visitPredictNode(i);
            for (auto it = adjacencyList[i].begin(); it != adjacencyList[i].end(); ++it) {
                visitPredictNeighbor(it->second);
            }
        }
    } else {
        std::vector<int> inDegree(size, 0);
        for (int i = 0; i < size; i++) {
            for (auto it = adjacencyList[i].begin(); it != adjacencyList[i].end(); ++it) {
                inDegree[it->second.dest]++;
            }
        }
        std::queue<int> q;
        for (int id : inputNodeIds) {
            q.push(id);
        }
        std::vector<bool> visited(size, false);
        while (!q.empty()) {
            int nodeId = q.front();
            q.pop();
            if (visited[nodeId]) continue;
            visited[nodeId] = true;
            visitPredictNode(nodeId);
            for (auto it = adjacencyList[nodeId].begin(); it != adjacencyList[nodeId].end(); ++it) {
                visitPredictNeighbor(it->second);
                inDegree[it->second.dest]--;
                if (inDegree[it->second.dest] == 0) {
                    q.push(it->second.dest);
                }
            }
        }
    }

    // Collect output values
    std::vector<double> output;
    for (size_t i = 0; i < outputNodeIds.size(); i++) {
        output.push_back(nodes[outputNodeIds[i]]->postActivationValue);
    }

    // If training, accumulate gradients
    if (!evaluating) {
        contribute(instance.y, output[0]);
        batchSize++;
    }

    return output;
}

// ==================== Contribute (Backpropagation) ====================

bool NeuralNetwork::contribute(double y, double p) {
    contributions.clear();

    for (size_t i = 0; i < inputNodeIds.size(); i++) {
        contribute(inputNodeIds[i], y, p);
    }

    return true;
}

double NeuralNetwork::contribute(int nodeId, const double& y, const double& p) {
    if (contributions.find(nodeId) != contributions.end()) {
        return contributions[nodeId];
    }

    visitContributeStart(nodeId);

    double outgoing = 0;

    // Check if this is an input node
    bool isInput = false;
    for (int id : inputNodeIds) {
        if (id == nodeId) { isInput = true; break; }
    }

    if (adjacencyList[nodeId].empty()) {
        // Output node: cross-entropy gradient = (p - y)
        // No multiplication by activation derivative (cancels with sigmoid)
        outgoing = p - y;
        nodes[nodeId]->delta += outgoing;
    } else {
        // Hidden or input node: recurse into neighbors
        for (auto& pair : adjacencyList[nodeId]) {
            double incoming = contribute(pair.second.dest, y, p);
            visitContributeNeighbor(pair.second, incoming, outgoing);
        }

        // Apply activation derivative and accumulate bias delta
        // but NOT for input nodes (input biases are not learned)
        if (!isInput) {
            visitContributeNode(nodeId, outgoing);
        }
    }

    contributions[nodeId] = outgoing;
    return outgoing;
}

// ==================== Update (Apply Gradients) ====================

bool NeuralNetwork::update() {
    if (batchSize == 0) return false;

    for (int i = 0; i < size; i++) {
        if (nodes[i]) {
            nodes[i]->bias -= learningRate * nodes[i]->delta;
            nodes[i]->delta = 0;
        }
    }

    for (int i = 0; i < size; i++) {
        for (auto& pair : adjacencyList[i]) {
            pair.second.weight -= learningRate * pair.second.delta;
            pair.second.delta = 0;
        }
    }

    batchSize = 0;
    return true;
}

// ==================== Assess ====================

double NeuralNetwork::assess(DataLoader dl) {
    bool wasTraining = !evaluating;
    eval();

    int correct = 0;
    std::vector<DataInstance> data = dl.getData();

    for (size_t i = 0; i < data.size(); i++) {
        std::vector<double> prediction = predict(data[i]);
        int predictedLabel = (prediction[0] >= 0.5) ? 1 : 0;
        if (predictedLabel == data[i].y) {
            correct++;
        }
    }

    if (wasTraining) train();

    return (double)correct / data.size();
}

double NeuralNetwork::assess(std::string filename) {
    DataLoader dl(filename);
    return assess(dl);
}

// ==================== Flush ====================

void NeuralNetwork::flush() {
    for (int i = 0; i < size; i++) {
        if (nodes[i]) {
            nodes[i]->preActivationValue = 0;
            nodes[i]->postActivationValue = 0;
        }
    }
}

// ==================== File I/O ====================

void NeuralNetwork::loadNetwork(std::istream& in) {
    // Read the entire file into a flat list of whitespace-separated tokens,
    // stripping any '#' comments. This is immune to stream-state quirks that
    // can arise from mixing operator>> and getline across stdlib versions.
    std::vector<std::string> tokens;
    std::string line;
    while (std::getline(in, line)) {
        std::size_t hashPos = line.find('#');
        if (hashPos != std::string::npos) {
            line = line.substr(0, hashPos);
        }
        std::istringstream iss(line);
        std::string tok;
        while (iss >> tok) {
            tokens.push_back(tok);
        }
    }

    std::size_t idx = 0;

    // numLayers numNodes
    int numLayers = std::stoi(tokens[idx++]);
    int numNodes = std::stoi(tokens[idx++]);

    resize(numNodes);

    // Per layer: layerSize activationFunction
    layers.assign(numLayers, std::vector<int>());
    int nodeId = 0;
    for (int l = 0; l < numLayers; l++) {
        int layerSize = std::stoi(tokens[idx++]);
        std::string activation = tokens[idx++];
        for (int j = 0; j < layerSize; j++) {
            NodeInfo n(activation, 0, 0);
            updateNode(nodeId, n);
            layers[l].push_back(nodeId);
            nodeId++;
        }
    }

    inputNodeIds = layers.front();
    outputNodeIds = layers.back();

    // numWeights, then src dest weight triples
    int numWeights = std::stoi(tokens[idx++]);
    for (int i = 0; i < numWeights; i++) {
        int src = std::stoi(tokens[idx++]);
        int dst = std::stoi(tokens[idx++]);
        double w = std::stod(tokens[idx++]);
        updateConnection(src, dst, w);
    }

    // numBiases, then nodeId bias pairs (may be 0 / absent)
    if (idx < tokens.size()) {
        int numBiases = std::stoi(tokens[idx++]);
        for (int i = 0; i < numBiases && idx + 1 < tokens.size() + 2; i++) {
            int nId = std::stoi(tokens[idx++]);
            double b = std::stod(tokens[idx++]);
            if (nId >= 0 && nId < numNodes && nodes[nId]) {
                nodes[nId]->bias = b;
            }
        }
    }

    // Activate all nodes so postActivation reflects activation(0)
    for (int i = 0; i < numNodes; i++) {
        if (nodes[i]) {
            nodes[i]->activate();
        }
    }
}

void NeuralNetwork::saveModel(std::string filename) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return;

    fout << layers.size() << " " << size << std::endl;
    for (size_t l = 0; l < layers.size(); l++) {
        std::string activation = getActivationIdentifier(nodes[layers[l][0]]->activationFunction);
        fout << layers[l].size() << " " << activation << std::endl;
    }

    int numWeights = 0;
    for (int i = 0; i < size; i++) numWeights += adjacencyList[i].size();
    fout << numWeights << std::endl;
    for (int i = 0; i < size; i++) {
        for (auto& pair : adjacencyList[i]) {
            fout << pair.second.source << " " << pair.second.dest << " " << pair.second.weight << std::endl;
        }
    }

    std::vector<std::pair<int, double>> biases;
    for (int i = 0; i < size; i++) {
        if (nodes[i] && std::abs(nodes[i]->bias) > 0.00001) {
            biases.push_back({i, nodes[i]->bias});
        }
    }
    fout << biases.size() << std::endl;
    for (auto& b : biases) {
        fout << b.first << " " << b.second << std::endl;
    }
    fout.close();
}

// ==================== operator<< ====================

std::ostream& operator<<(std::ostream& out, const NeuralNetwork& nn) {
    // Print layer structure (trailing space after each node id)
    for (size_t l = 0; l < nn.layers.size(); l++) {
        out << "layer " << l << ":";
        for (size_t n = 0; n < nn.layers[l].size(); n++) {
            out << " " << nn.layers[l][n];
        }
        out << " " << std::endl;
    }

    // Print digraph with leading tab on each edge
    out << "digraph G {" << std::endl;
    for (int i = 0; i < nn.size; i++) {
        std::vector<Connection> edges;
        for (auto& pair : nn.adjacencyList[i]) {
            edges.push_back(pair.second);
        }
        std::sort(edges.begin(), edges.end(), [](const Connection& a, const Connection& b) {
            return a.dest > b.dest;
        });
        for (auto& c : edges) {
            out << "\t" << c.source << " -> " << c.dest
                << "[label=\"" << c.weight << "\"]" << std::endl;
        }
    }
    out << "}" << std::endl;

    // Print node details
    for (int i = 0; i < nn.size; i++) {
        out << "node " << i << ": "
            << "(z=" << nn.nodes[i]->preActivationValue
            << "\t, a=" << nn.nodes[i]->postActivationValue
            << "\t, bias=" << nn.nodes[i]->bias
            << "\t, activation=" << getActivationIdentifier(nn.nodes[i]->activationFunction)
            << ")" << std::endl;
    }

    return out;
}