#include "NeuralNetwork.hpp"
#include "Trace.hpp"
#include <algorithm>
#include <iomanip>
#include <queue>

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

void NeuralNetwork::eval() {
    evaluating = true;
}

void NeuralNetwork::train() {
    evaluating = false;
}

void NeuralNetwork::setLearningRate(double lr) {
    learningRate = lr;
}

void NeuralNetwork::setInputNodeIds(std::vector<int> ids) {
    inputNodeIds = ids;
}

void NeuralNetwork::setOutputNodeIds(std::vector<int> ids) {
    outputNodeIds = ids;
}

std::vector<int> NeuralNetwork::getInputNodeIds() const {
    return inputNodeIds;
}

std::vector<int> NeuralNetwork::getOutputNodeIds() const {
    return outputNodeIds;
}

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

    // Set input node values from the data instance
    for (size_t i = 0; i < inputNodeIds.size(); i++) {
        nodes[inputNodeIds[i]]->preActivationValue = instance.x[i];
    }

    // BFS topological sort ensures nodes are processed only after
    // all their incoming contributions have been accumulated
    std::vector<int> inDegree(size, 0);
    for (int i = 0; i < size; i++) {
        for (auto& pair : adjacencyList[i]) {
            inDegree[pair.second.dest]++;
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
        for (auto& pair : adjacencyList[nodeId]) {
            visitPredictNeighbor(pair.second);
            inDegree[pair.second.dest]--;
            if (inDegree[pair.second.dest] == 0) {
                q.push(pair.second.dest);
            }
        }
    }

    // Collect output values
    std::vector<double> output;
    for (size_t i = 0; i < outputNodeIds.size(); i++) {
        output.push_back(nodes[outputNodeIds[i]]->postActivationValue);
    }

    // If training, accumulate gradients via backpropagation
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

    if (adjacencyList[nodeId].empty()) {
        // Output node: with a sigmoid output and binary cross-entropy loss,
        // the gradient of the loss with respect to the pre-activation value
        // simplifies exactly to (prediction - label). We must NOT multiply by
        // the activation derivative again here.
        outgoing = p - y;
        nodes[nodeId]->delta += outgoing;
    } else {
        // Hidden or input node: recurse into neighbors, accumulating each
        // connection's weight gradient and this node's incoming gradient.
        for (auto& pair : adjacencyList[nodeId]) {
            double incoming = contribute(pair.second.dest, y, p);
            visitContributeNeighbor(pair.second, incoming, outgoing);
        }
        // Chain rule through this node's activation function.
        outgoing *= nodes[nodeId]->derive();
        // Input nodes only hold the data instance's feature values; their bias
        // is not a trainable parameter, so they do not accumulate a gradient.
        bool isInput = std::find(inputNodeIds.begin(), inputNodeIds.end(),
                                 nodeId) != inputNodeIds.end();
        if (!isInput) {
            nodes[nodeId]->delta += outgoing;
        }
    }

    contributions[nodeId] = outgoing;
    return outgoing;
}

// ==================== Update (Apply Gradients) ====================

bool NeuralNetwork::update() {
    if (batchSize == 0) return false;

    // Update biases
    for (int i = 0; i < size; i++) {
        if (nodes[i]) {
            nodes[i]->bias -= learningRate * nodes[i]->delta;
            nodes[i]->delta = 0;
        }
    }

    // Update weights
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
    std::string line;

    // Line 1: numLayers numNodes
    int numLayers, numNodes;
    in >> numLayers >> numNodes;
    std::getline(in, line); // consume rest of line

    resize(numNodes);

    // For each layer: numNodesInLayer activationFunction
    layers.resize(numLayers);
    int nodeId = 0;
    for (int l = 0; l < numLayers; l++) {
        int layerSize;
        std::string activation;
        in >> layerSize >> activation;
        std::getline(in, line); // consume rest of line

        for (int j = 0; j < layerSize; j++) {
            NodeInfo n(activation, 0, 0);
            updateNode(nodeId, n);
            layers[l].push_back(nodeId);
            nodeId++;
        }
    }

    // Set input and output node ids from first and last layer
    inputNodeIds = layers[0];
    outputNodeIds = layers[numLayers - 1];

    // Number of weights
    int numWeights;
    in >> numWeights;
    std::getline(in, line);

    // Each weight: source dest weight
    for (int i = 0; i < numWeights; i++) {
        int src, dst;
        double w;
        in >> src >> dst >> w;
        std::getline(in, line);
        updateConnection(src, dst, w);
    }

    // Number of biases
    int numBiases;
    in >> numBiases;
    std::getline(in, line);

    // Each bias: nodeId biasValue
    for (int i = 0; i < numBiases; i++) {
        int nId;
        double b;
        in >> nId >> b;
        std::getline(in, line);
        if (nId >= 0 && nId < numNodes && nodes[nId]) {
            nodes[nId]->bias = b;
        }
    }

    // Activate all nodes so postActivation reflects activation(0)
    // e.g. sigmoid nodes get postActivation = 0.5
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
        int layerSize = layers[l].size();
        std::string activation = getActivationIdentifier(nodes[layers[l][0]]->activationFunction);
        fout << layerSize << " " << activation << std::endl;
    }

    int numWeights = 0;
    for (int i = 0; i < size; i++) {
        numWeights += adjacencyList[i].size();
    }
    fout << numWeights << std::endl;
    for (int i = 0; i < size; i++) {
        for (auto& pair : adjacencyList[i]) {
            fout << pair.second.source << " "
                 << pair.second.dest << " "
                 << pair.second.weight << std::endl;
        }
    }

    // Write all biases
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
    // Print layer structure
    for (size_t l = 0; l < nn.layers.size(); l++) {
        out << "layer " << l << ": ";
        for (size_t n = 0; n < nn.layers[l].size(); n++) {
            out << nn.layers[l][n] << " ";
        }
        out << std::endl;
    }

    // Print digraph
    out << "digraph G {" << std::endl;
    for (int i = 0; i < nn.size; i++) {
        // Collect edges and sort by destination descending
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