/* Neural-Net-Tutorial.cpp : This file contains the 'main' function. Program execution begins and ends there.
	
	Copied the tutorial by David Miller.

	This net will emulate an XOR-Gate, and takes approximately 3000 datums to properly learn the gate.
	
	Source: http://www.millermattson.com/dave/?p=54
*/

#include "pch.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

// Temp Test Class:

class TrainingData {
public:
	TrainingData(const std::string filename);
	bool isEOf(void) { return m_trainingDataFile.eof(); }
	void getTopology(std::vector<unsigned> &topology);

	unsigned getNextInputs(std::vector<double> &inputVals);
	unsigned getTargetOutputs(std::vector<double> &targetOutputVals);

private:
	std::ifstream m_trainingDataFile;
};

void TrainingData::getTopology(std::vector<unsigned> &topology) {
	std::string line;
	std::string label;

	getline(m_trainingDataFile, line);
	std::stringstream ss(line);
	ss >> label;

	if (this->isEOf() || label.compare("topology:") != 0) {
		std::cout << isEOf();
		abort();
	}

	while (!ss.eof()) {
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}

	return;
}

TrainingData::TrainingData(const std::string filename) {
	m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(std::vector<double> &inputVals) {
	inputVals.clear();

	std::string line;
	getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;

	if (label.compare("in:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			inputVals.push_back(oneValue);
		}
	}

	return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(std::vector<double> &targetOutputVals) {
	targetOutputVals.clear();

	std::string line;
	getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("out:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			targetOutputVals.push_back(oneValue);
		}
	}

	return targetOutputVals.size();
}

struct Connection {
	double weight;
	double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

// ----------------- Class Neuron Begin ----------------

class Neuron {
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
private:
	static double eta; // [0.0 .. 1.0]
	static double alpha;// [0.0 .. n]
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;
	unsigned m_myIndex;
	double m_gradient;
	std::vector<Connection> m_outputWeights;
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

double Neuron::sumDOW(const Layer &nextLayer) const {
	double sum  = 0.0;

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer) {
	for (unsigned n = 0; n < prevLayer.size() - 1; ++n) {
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
		
		double newDeltaWeight =
			eta
			* neuron.getOutputVal()
			* m_gradient
			+ alpha
			* oldDeltaWeight;

		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal) {
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x) {
	// Using tanh function (tanh x = (e^x - e^-x) / (e^x + e^-x)
	// Outputs {-1.0 .. 1.0}
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) {
	// Using tanh function (tanh x = (e^x - e^-x) / (e^x + e^-x)
	// Outputs {-1.0 .. 1.0}
	return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer) {
	double sum = 0.0;

	// Sum the previous layer's output (which are our inputs),
	// including the bias node.

	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		sum += prevLayer[n].getOutputVal() *
			prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);

}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) 
{
	for (unsigned c = 0; c < numOutputs; ++c) {
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}

// ----------------- Class Neuron End ----------------------

// ----------------- Class Net Begin  ----------------------

class Net
{
public:
	Net(const std::vector<unsigned> &topology);
	void feedForward(const std::vector<double> &inputVals);
	void backProp(const std::vector<double> &targetVals);
	void getResults(std::vector<double> &resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError; }
private:
	std::vector<Layer> m_layers;
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;
};

void Net::getResults(std::vector<double> &resultVals) const {
	resultVals.clear();

	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}
void Net::backProp(const std::vector<double> &targetVals) {
	
	// Calculate overall net error (RMS of output neuron errors):

	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}

	m_error /= outputLayer.size() - 1;
	m_error = sqrt(m_error); // RMS

	// Implement a recent average measurement:

	m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
				/ (m_recentAverageSmoothingFactor + 1.0);

	// Calculate output layer gradients:

	for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// Calculate gradients on hidden layers:
	
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];


		for (unsigned n = 0; n < hiddenLayer.size(); n++) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// For all layers from output to first hidden,
	// update connection weights:

	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; n++) {
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::feedForward(const std::vector<double> &inputVals) 
{
	assert(inputVals.size() == m_layers[0].size() - 1);

	// Assign (latch) the input values into the input neurons:
	for (unsigned i = 0; i < inputVals.size(); i++) {
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	// Forward propogate:
	for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; n++) {
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}

}

Net::Net(const std::vector<unsigned> &topology) {
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		// Vertical layer of Nueurons:
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		// Neurons in that previously made layer:
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
		}

		m_layers.back().back().setOutputVal(1.0);
	}
}
// ---------------------- Class Net End ---------------------

void showVectorVals(std::string label, std::vector<double> &v) {
	std::cout << label << " ";

	for (unsigned i = 0; i < v.size(); ++i) {
		std::cout << v[i] << " ";
	}

	std::cout << std::endl;
}

int main()
{
	TrainingData trainData("tmp/trainingData.txt");

	std::vector<unsigned> topology;
	trainData.getTopology(topology);
	Net myNet(topology);

	std::vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;

	while (!trainData.isEOf()) {
		++trainingPass;

		std::cout << std::endl << "Pass " << trainingPass;

		if (trainData.getNextInputs(inputVals) != topology[0])
			break;

		showVectorVals(": Inputs:", inputVals);
		myNet.feedForward(inputVals);

		myNet.getResults(resultVals);
		showVectorVals("outputs:", resultVals);

		trainData.getTargetOutputs(targetVals);
		showVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		std::cout << "Net recent average error: "
			<< myNet.getRecentAverageError() << std::endl;

	}

	std::cout << std::endl << "Done" << std::endl;
}