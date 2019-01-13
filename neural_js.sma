function neuron_create(total_inputs) {
	var neuron = {};

	neuron.weights = new Array(total_inputs);

	for(let i = 0; i < total_inputs; i++) {
		neuron.weights[i] = Math.random() * 2.0 - 1.0;
	}

	neuron.bias = Math.random() * 2.0 - 1.0;

	return neuron;
}

function neuron_think(neuron, inputs) {
	neuron.inputs = inputs;
	
	var out = neuron.bias;

	for(let i = 0; i < inputs.length; i++) {
		out += inputs[i] * neuron.weights[i];
	}

	return neuron.sigmoid = 1 / (1 + Math.pow(2.718281828459045, -out));
}

function layer_create(total_neurons, total_inputs) {
	var layer = {};

	layer.neurons = new Array(total_neurons);

	for(let i = 0; i < layer.neurons.length; i++) {
		layer.neurons[i] = neuron_create(total_inputs);
	}

	return layer;
}

function layer_think(layer, inputs) {
	var outs = new Array(layer.neurons.length);

	for(let i = 0; i < layer.neurons.length; i++) {
		outs[i] = neuron_think(layer.neurons[i], inputs);
	}

	return outs;
}

function neural_create() {
	var neural = {};

	neural.layers = [];

	return neural;
}

function neural_layer(neural, total_neurons, total_inputs) {
	total_inputs = total_inputs || neural.layers[neural.layers.length - 1].neurons.length;

	neural.layers.push(layer_create(total_neurons, total_inputs));
}

function neural_think(neural, inputs) {
	var outs;

	for(let i = 0; i < neural.layers.length; i++) {
		outs = layer_think(neural.layers[i], inputs);
		inputs = outs;
	}

	return outs;
}

function neural_learn(neural, data, rate) {
	rate = rate || 0.3;

	var output_layer = neural.layers[neural.layers.length - 1]; //get output layer (that's, the last layer)

	var mse = 0.0; //to calculate the mean squared error

	for(let d = 0; d < data.length; d++) { //loop; data

		/*== output layer ==*/
		let outputs = neural_think(neural, data[d][0]); //input

		for(let o = 0; o < output_layer.neurons.length; o++) { //loop; neurons of output layer (1 neuron = 1 output, etc)
			let neuron = output_layer.neurons[o]; //get neuron

			neuron.error = data[d][1][o] - outputs[o]; //wanted - actual

			mse += Math.pow(neuron.error, 2); //sum every error of every neuron of output layer

			neuron.delta = neuron.sigmoid * (1 - neuron.sigmoid) * neuron.error; //sigmoid derivated * error

			//every modification are referenced, so those changes are automatically saved
		}

		/*== hidden layers ==*/
		for(let l = neural.layers.length - 2; l >= 0; l--) { //loop; hidden layers (-2 to avoid the output layer)

			for(let h = 0; h < neural.layers[l].neurons.length; h++) { //loop; neurons of hidden layer
				let neuronH = neural.layers[l].neurons[h]; //get neuron; to avoid name conflicts -> H for "hidden"

				let neurons_next = neural.layers[l + 1].neurons; //get neurons of next layer; with +1 we recover the output layer

				neuronH.error = 0.0;

				for(let i = 0; i < neurons_next.length; i++) { //loop; neurons of next layer
					neuronH.error += neurons_next[i].weights[h] * neurons_next[i].delta; //on this way, neurons are connected with the next ones respectively

					let neuron = neurons_next[i]; //just to shorten the variable

					for(let w = 0; w < neuron.weights.length; w++) {
						neuron.weights[w] += neuron.inputs[w] * neuron.delta * rate; //finally, learn!
					}
					neuron.bias += neuron.delta * rate; //the bias learn too
				}

				neuronH.delta = neuronH.sigmoid * (1 - neuronH.sigmoid) * neuronH.error; //sigmoid derivated * error
			}

		}

	}

	return mse / (output_layer.neurons.length * data.length);
}

var neural = neural_create();

neural_layer(neural, 8, 2); //hidden/input layer: 6 neurons and 2 inputs
neural_layer(neural, 8); //hidden layer with 6 neurons
//neural.layer(6); //can add more hidden layers
neural_layer(neural, 1); //output layer: 1 neuron (1 = output)

for(let i = 0, mse; i < 25000; i++) { //25.000 limit iterations
	mse = neural_learn(neural, [
		[[0, 0], [0]], //[ [ input1, input2 ], [ output1 ] ]
		[[1, 0], [1]],
		[[0, 1], [1]],
		[[1, 1], [0]]
	], 0.45); //learning rate

	if(mse < 0.00005) { //stop learn when mean squared error reach this limit
		console.log('mse threshold on iter', i);
		break;
	}

	if(!(i % 1000)) {
		console.log('iter', i, 'mse', parseFloat(mse.toFixed(7)), 'actual', parseFloat(neural_think(neural, [1, 0])[0].toFixed(5)));
	}
}

console.log('final output', neural_think(neural, [1, 0]));
