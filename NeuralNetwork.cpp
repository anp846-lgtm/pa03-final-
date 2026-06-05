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

// ==================== Predict (Forward Pass, BFT) ====================

std::vector<double> NeuralNetwork::predict(DataInstance instance) {
    flush();

    int n = (int)adjacencyList.size();

    // Set input node values from the data instance.
    for (size_t i = 0; i < inputNodeIds.size(); i++) {
        nodes[inputNodeIds[i]]->preActivationValue = instance.x[i];
    }

    // Breadth-First Traversal with in-degree tracking so that a node is only
    // visited (activated) after every incoming connection has contributed,
    // i.e. only after the previous layer has finished.
    std::vector<int> inDegree(n, 0);
    for (int i = 0; i < n; i++) {
        for (auto it = adjacencyList[i].begin(); it != adjacencyList[i].end(); ++it) {
            inDegree[it->second.dest]++;
        }
    }

    std::queue<int> q;
    for (size_t i = 0; i < inputNodeIds.size(); i++) {
        q.push(inputNodeIds[i]);
    }

    std::vector<bool> visited(n, false);
    while (!q.empty()) {
        int v = q.front();
        q.pop();
        if (visited[v]) continue;
        visited[v] = true;

        visitPredictNode(v);

        for (auto it = adjacencyList[v].begin(); it != adjacencyList[v].end(); ++it) {
            visitPredictNeighbor(it->second);
            int d = it->second.dest;
            inDegree[d]--;
            if (inDegree[d] == 0) {
                q.push(d);
            }
        }
    }

    // Collect outputs.
    std::vector<double> output;
    for (size_t i = 0; i < outputNodeIds.size(); i++) {
        output.push_back(nodes[outputNodeIds[i]]->postActivationValue);
    }

    // In training mode, run backpropagation to accumulate gradients.
    if (!evaluating) {
        contribute(instance.y, output[0]);
        batchSize++;
    }

    return output;
}

// ==================== Contribute (Backpropagation, DFT) ====================

bool NeuralNetwork::contribute(double y, double p) {
    contributions.clear();
    for (size_t i = 0; i < inputNodeIds.size(); i++) {
        contribute(inputNodeIds[i], y, p);
    }
    return true;
}

double NeuralNetwork::contribute(int nodeId, const double& y, const double& p) {
    // contributions doubles as the "visited" set and memoizes results.
    if (contributions.find(nodeId) != contributions.end()) {
        return contributions[nodeId];
    }

    visitContributeStart(nodeId);

    double outgoingContribution = 0;

    if (adjacencyList[nodeId].empty()) {
        // Output node: base case. With a sigmoid output under cross-entropy,
        // the gradient w.r.t. the pre-activation simplifies to (p - y).
        outgoingContribution = p - y;
        nodes[nodeId]->delta += outgoingContribution;
    } else {
        // Recurse forward to each neighbor, accumulating gradients on the way back.
        for (auto it = adjacencyList[nodeId].begin(); it != adjacencyList[nodeId].end(); ++it) {
            double incomingContribution = contribute(it->second.dest, y, p);
            visitContributeNeighbor(it->second, incomingContribution, outgoingContribution);
        }
        // Apply this node's activation derivative and record its bias gradient,
        // but only for non-input nodes (input nodes have no trainable bias).
        bool isInput = std::find(inputNodeIds.begin(), inputNodeIds.end(), nodeId) != inputNodeIds.end();
        if (isInput) {
            outgoingContribution *= nodes[nodeId]->derive();
        } else {
            visitContributeNode(nodeId, outgoingContribution);
        }
    }

    contributions[nodeId] = outgoingContribution;
    return outgoingContribution;
}

// ==================== Update ====================

bool NeuralNetwork::update() {
    if (batchSize == 0) return false;

    for (int i = 0; i < (int)nodes.size(); i++) {
        if (nodes[i] != nullptr) {
            nodes[i]->bias -= learningRate * nodes[i]->delta;
            nodes[i]->delta = 0;
        }
    }

    for (int i = 0; i < (int)adjacencyList.size(); i++) {
        for (auto it = adjacencyList[i].begin(); it != adjacencyList[i].end(); ++it) {
            it->second.weight -= learningRate * it->second.delta;
            it->second.delta = 0;
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
    for (int i = 0; i < (int)nodes.size(); i++) {
        if (nodes[i] != nullptr) {
            nodes[i]->preActivationValue = 0;
            nodes[i]->postActivationValue = 0;
        }
    }
}

// ==================== File I/O ====================

void NeuralNetwork::loadNetwork(std::istream& in) {
    // Read the whole stream into whitespace-separated tokens, stripping '#'
    // comments. This is robust against any operator>> / getline interaction.
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

    int numLayers = std::stoi(tokens[idx++]);
    int numNodes = std::stoi(tokens[idx++]);

    resize(numNodes);

    layers.assign(numLayers, std::vector<int>());
    int nodeId = 0;
    for (int l = 0; l < numLayers; l++) {
        int layerSize = std::stoi(tokens[idx++]);
        std::string activation = tokens[idx++];
        for (int j = 0; j < layerSize; j++) {
            updateNode(nodeId, NodeInfo(activation, 0, 0));
            layers[l].push_back(nodeId);
            nodeId++;
        }
    }

    inputNodeIds = layers.front();
    outputNodeIds = layers.back();

    int numWeights = std::stoi(tokens[idx++]);
    for (int i = 0; i < numWeights; i++) {
        int src = std::stoi(tokens[idx++]);
        int dst = std::stoi(tokens[idx++]);
        double w = std::stod(tokens[idx++]);
        updateConnection(src, dst, w);
    }

    if (idx < tokens.size()) {
        int numBiases = std::stoi(tokens[idx++]);
        for (int i = 0; i < numBiases && idx + 1 <= tokens.size(); i++) {
            int nId = std::stoi(tokens[idx++]);
            double b = std::stod(tokens[idx++]);
            if (nId >= 0 && nId < numNodes && nodes[nId] != nullptr) {
                nodes[nId]->bias = b;
            }
        }
    }

    // Activate every node so postActivation reflects activation(0)
    // (e.g. a sigmoid output node reads a = 0.5 before any prediction).
    for (int i = 0; i < numNodes; i++) {
        if (nodes[i] != nullptr) {
            nodes[i]->activate();
        }
    }
}

void NeuralNetwork::saveModel(std::string filename) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return;

    fout << layers.size() << " " << nodes.size() << std::endl;
    for (size_t l = 0; l < layers.size(); l++) {
        std::string activation = getActivationIdentifier(nodes[layers[l][0]]->activationFunction);
        fout << layers[l].size() << " " << activation << std::endl;
    }

    int numWeights = 0;
    for (size_t i = 0; i < adjacencyList.size(); i++) numWeights += adjacencyList[i].size();
    fout << numWeights << std::endl;
    for (size_t i = 0; i < adjacencyList.size(); i++) {
        for (auto it = adjacencyList[i].begin(); it != adjacencyList[i].end(); ++it) {
            fout << it->second.source << " " << it->second.dest << " " << it->second.weight << std::endl;
        }
    }

    std::vector<std::pair<int, double>> biases;
    for (size_t i = 0; i < nodes.size(); i++) {
        if (nodes[i] != nullptr && std::abs(nodes[i]->bias) > 0.00001) {
            biases.push_back({(int)i, nodes[i]->bias});
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
    // Layer structure (trailing space after each id, matching expected output).
    for (size_t l = 0; l < nn.layers.size(); l++) {
        out << "layer " << l << ": ";
        for (size_t k = 0; k < nn.layers[l].size(); k++) {
            out << nn.layers[l][k] << " ";
        }
        out << std::endl;
    }

    // DOT digraph; edges sorted by destination descending, each indented by a tab.
    out << "digraph G {" << std::endl;
    for (size_t i = 0; i < nn.adjacencyList.size(); i++) {
        std::vector<Connection> edges;
        for (auto it = nn.adjacencyList[i].begin(); it != nn.adjacencyList[i].end(); ++it) {
            edges.push_back(it->second);
        }
        std::sort(edges.begin(), edges.end(), [](const Connection& a, const Connection& b) {
            return a.dest > b.dest;
        });
        for (size_t e = 0; e < edges.size(); e++) {
            out << "\t" << edges[e].source << " -> " << edges[e].dest
                << "[label=\"" << edges[e].weight << "\"]" << std::endl;
        }
    }
    out << "}" << std::endl;

    // Node details.
    for (size_t i = 0; i < nn.nodes.size(); i++) {
        out << "node " << i << ": (z=";
        if (nn.nodes[i] != nullptr) {
            out << nn.nodes[i]->preActivationValue
                << "\t, a=" << nn.nodes[i]->postActivationValue
                << "\t, bias=" << nn.nodes[i]->bias
                << "\t, activation=" << getActivationIdentifier(nn.nodes[i]->activationFunction);
        }
        out << ")" << std::endl;
    }

    return out;
}