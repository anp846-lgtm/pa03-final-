#include "NeuralNetwork.hpp"
#include "Trace.hpp"
#include <algorithm>
#include <iomanip>

// ==================== Constructors ====================

NeuralNetwork::NeuralNetwork() : Graph() {
    evaluating = true;
    learningRate = 1.0;
    batchSize = 0;
}

NeuralNetwork::NeuralNetwork(int size) : Graph(size) {
    evaluating = true;
    learningRate = 1.0;
    batchSize = 0;
}

NeuralNetwork::NeuralNetwork(std::string filename) : Graph() {
    evaluating = true;
    learningRate = 1.0;
    batchSize = 0;
    std::ifstream fin(filename);
    if (fin.is_open()) {
        loadNetwork(fin);
        fin.close();
    }
}

NeuralNetwork::NeuralNetwork(std::istream& in) : Graph() {
    evaluating = true;
    learningRate = 1.0;
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
    // Add bias to accumulated weighted sum, then activate
    nodes[vId]->preActivationValue += nodes[vId]->bias;
    nodes[vId]->activate();
}

void NeuralNetwork::visitPredictNeighbor(Connection c) {
    // Add this node's contribution to the destination node
    nodes[c.dest]->preActivationValue += nodes[c.source]->postActivationValue * c.weight;
}

// ==================== Visit Helpers (Contribute / Backprop) ====================

void NeuralNetwork::visitContributeStart(int vId) {
    // No-op: placeholder for tracing or future extension
}

void NeuralNetwork::visitContributeNode(int vId, double& outgoingContribution) {
    // Multiply by activation derivative, then accumulate into node's bias delta
    outgoingContribution *= nodes[vId]->derive();
    nodes[vId]->delta += outgoingContribution;
}

void NeuralNetwork::visitContributeNeighbor(Connection& c, double& incomingContribution, double& outgoingContribution) {
    // Weight gradient: delta_dest * activation_source
    c.delta += incomingContribution * nodes[c.source]->postActivationValue;
    // Accumulate for this node's delta computation
    outgoingContribution += incomingContribution * c.weight;
}

// ==================== Predict (Forward Pass) ====================

std::vector<double> NeuralNetwork::predict(DataInstance instance) {
    flush();

    // Set input node values from the data instance
    for (size_t i = 0; i < inputNodeIds.size(); i++) {
        nodes[inputNodeIds[i]]->preActivationValue = instance.x[i];
    }

    // Process each layer: activate nodes, then propagate to next layer
    for (size_t l = 0; l < layers.size(); l++) {
        for (size_t n = 0; n < layers[l].size(); n++) {
            int nodeId = layers[l][n];
            visitPredictNode(nodeId);

            // Propagate to all neighbors
            for (auto& pair : adjacencyList[nodeId]) {
                visitPredictNeighbor(pair.second);
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

    // Initiate DFS from each input node
    for (size_t i = 0; i < inputNodeIds.size(); i++) {
        contribute(inputNodeIds[i], y, p);
    }

    return true;
}

double NeuralNetwork::contribute(int nodeId, const double& y, const double& p) {
    // If already computed for this node, return cached value
    if (contributions.find(nodeId) != contributions.end()) {
        return contributions[nodeId];
    }

    visitContributeStart(nodeId);

    double outgoing = 0;

    if (adjacencyList[nodeId].empty()) {
        // Output node: initial gradient is (prediction - label)
        outgoing = p - y;
    } else {
        // Hidden or input node: recurse into neighbors
        for (auto& pair : adjacencyList[nodeId]) {
            double incoming = contribute(pair.second.dest, y, p);
            visitContributeNeighbor(pair.second, incoming, outgoing);
        }
    }

    visitContributeNode(nodeId, outgoing);

    contributions[nodeId] = outgoing;
    return outgoing;
}

// ==================== Update (Apply Gradients) ====================

bool NeuralNetwork::update() {
    if (batchSize == 0) return false;

    // Update biases
    for (int i = 0; i < size; i++) {
        if (nodes[i]) {
            nodes[i]->bias -= learningRate * (nodes[i]->delta / batchSize);
            nodes[i]->delta = 0;
        }
    }

    // Update weights
    for (int i = 0; i < size; i++) {
        for (auto& pair : adjacencyList[i]) {
            pair.second.weight -= learningRate * (pair.second.delta / batchSize);
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
    // Read number of nodes
    int numNodes;
    in >> numNodes;

    resize(numNodes);

    // Read each node: id activationFunction bias
    for (int i = 0; i < numNodes; i++) {
        int id;
        std::string activation;
        double b;
        in >> id >> activation >> b;
        NodeInfo n(activation, 0, b);
        updateNode(id, n);
    }

    // Read number of edges
    int numEdges;
    in >> numEdges;

    // Read each edge: source dest weight
    for (int i = 0; i < numEdges; i++) {
        int src, dst;
        double w;
        in >> src >> dst >> w;
        updateConnection(src, dst, w);
    }

    // Read input node ids
    int numInputs;
    in >> numInputs;
    inputNodeIds.resize(numInputs);
    for (int i = 0; i < numInputs; i++) {
        in >> inputNodeIds[i];
    }

    // Read output node ids
    int numOutputs;
    in >> numOutputs;
    outputNodeIds.resize(numOutputs);
    for (int i = 0; i < numOutputs; i++) {
        in >> outputNodeIds[i];
    }

    // Read layers
    int numLayers;
    in >> numLayers;
    layers.resize(numLayers);
    for (int i = 0; i < numLayers; i++) {
        int layerSize;
        in >> layerSize;
        layers[i].resize(layerSize);
        for (int j = 0; j < layerSize; j++) {
            in >> layers[i][j];
        }
    }
}

void NeuralNetwork::saveModel(std::string filename) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return;

    // Write number of nodes
    fout << size << std::endl;

    // Write each node
    for (int i = 0; i < size; i++) {
        fout << i << " "
             << getActivationIdentifier(nodes[i]->activationFunction) << " "
             << nodes[i]->bias << std::endl;
    }

    // Count edges
    int numEdges = 0;
    for (int i = 0; i < size; i++) {
        numEdges += adjacencyList[i].size();
    }
    fout << numEdges << std::endl;

    // Write each edge
    for (int i = 0; i < size; i++) {
        for (auto& pair : adjacencyList[i]) {
            fout << pair.second.source << " "
                 << pair.second.dest << " "
                 << pair.second.weight << std::endl;
        }
    }

    // Write input node ids
    fout << inputNodeIds.size();
    for (size_t i = 0; i < inputNodeIds.size(); i++) {
        fout << " " << inputNodeIds[i];
    }
    fout << std::endl;

    // Write output node ids
    fout << outputNodeIds.size();
    for (size_t i = 0; i < outputNodeIds.size(); i++) {
        fout << " " << outputNodeIds[i];
    }
    fout << std::endl;

    // Write layers
    fout << layers.size() << std::endl;
    for (size_t i = 0; i < layers.size(); i++) {
        fout << layers[i].size();
        for (size_t j = 0; j < layers[i].size(); j++) {
            fout << " " << layers[i][j];
        }
        fout << std::endl;
    }

    fout.close();
}

// ==================== operator<< ====================

std::ostream& operator<<(std::ostream& out, const NeuralNetwork& nn) {
    // Print as a graph
    const Graph& g = nn;
    out << g;
    return out;
}