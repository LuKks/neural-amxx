#include fvault

const NEURAL_MAX_SIZE = 64; //max input, output or neurons
const NEURAL_MAX_LAYERS = 6;

enum Neuron {
	Float:N_weights[NEURAL_MAX_SIZE],
	Float:N_bias,
	Float:N_momentums[NEURAL_MAX_SIZE + 1], // +1 for bias
	N_max_inputs,
	Float:N_data[NEURAL_MAX_SIZE],
	Float:N_activation,
	Float:N_delta,
	Float:N_error
};

enum Layer {
	N_id,
	N_max_neurons,
	Float:N_input_min[NEURAL_MAX_SIZE],
	Float:N_input_max[NEURAL_MAX_SIZE],
	Float:N_rate,
	Float:N_momentum
};

//on this way, we have everything allocated in memory
//sometime ago I had made a version with dynamic arrays, it worked but had lower performance and more code, it was not worth
new layers_id;
new layers[NEURAL_MAX_LAYERS][Layer];
new neurons[NEURAL_MAX_LAYERS][NEURAL_MAX_SIZE][Neuron];

//neurons
public neuron_create(total_inputs) {
	new neuron[Neuron];

	neuron[N_max_inputs] = total_inputs;

	//initialize the neuron with random values. Also can set them with zero
	//but it's just to always randomize the output for same inputs before learn
	for(new i; i < total_inputs; i++) {
		neuron[N_weights][i] = _:random_float(-1.0, 1.0);
	}

	neuron[N_bias] = _:random_float(-1.0, 1.0);

	return neuron;
}

public Float:neuron_think(neuron[Neuron], Float:inputs[]) {
	static i, Float:out;

	//note: inputs variable in this moment might not be the real values that you use in the learn and think functions
	//normally the inputs are the previous layer outputs and this apply for almost in the most functions, just to know

	//loop every weight
	//important: the bias doesn't affect the activation basing on the input (as the weights do)
	for(i = 0, out = neuron[N_bias]; i < neuron[N_max_inputs]; i++) {
		//calculate the raw output of the neuron
		out += inputs[i] * neuron[N_weights][i];

		//by the way, cache for raw inputs, after can be used for learning
		neuron[N_data][i] = _:inputs[i];
	}

	//sigmoid, convert the raw output into a range of 0-1
	return neuron[N_activation] = _:(1 / (1 + floatpower(2.718281828459045, -out)));
}

public Float:neuron_optimizer(neuron[Neuron], layer[Layer], index, Float:value) {
	neuron[N_momentums][index] = _:(value + (layer[N_momentum] * neuron[N_momentums][index]));
	return neuron[N_momentums][index];
}

//layers
public layer_create(total_neurons, total_inputs) {
	layers[layers_id][N_id] = layers_id;
	layers[layers_id][N_max_neurons] = total_neurons;

	layers[layers_id][N_rate] = _:0.001;
	layers[layers_id][N_momentum] = _:0.999;

	for(new i; i < total_neurons; i++) {
		layers[layers_id][N_input_min][i] = _:0.0; //just to be clear
		layers[layers_id][N_input_max][i] = _:1.0;

		neurons[layers_id][i] = neuron_create(total_inputs);
	}

	return layers_id++;
}

public Float:layer_think(layer[Layer], Float:inputs[]) {
	static i, Float:outs[NEURAL_MAX_SIZE];

	//loop all neurons of the layer provided
	for(i = 0; i < layer[N_max_neurons]; i++) {
		//on this way we get the activation level of every neuron
		//note: the inputs variable is not changed inside neuron_think either in the next ones inside calls
		//so every neuron calculate their own activation based on same the inputs
		outs[i] = neuron_think(neurons[layer[N_id]][i], inputs);
	}

	return outs;
}

//networks
stock neural_layer(total_neurons, total_inputs = 0) {
	if(!layers_id && !total_inputs) {
		log_error(AMX_ERR_NATIVE, "need create the input layer in first place");
		return -1;
	}

	total_inputs = total_inputs > 0 ? total_inputs : layers[layers_id - 1][N_max_neurons];

	return layer_create(total_neurons, total_inputs);
}

public Float:neural_think_raw(Float:inputs[]) {
	static i, j, Float:outs[NEURAL_MAX_SIZE];
	static Float:_inputs[NEURAL_MAX_SIZE];

	//a copy of the inputs because the variable is referenced and need change them
	for(j = 0; j < neurons[0][0][N_max_inputs]; j++) {
		_inputs[j] = inputs[j];
	}

	//loop all layers to get outputs of every layer based on the actual values of _inputs
	for(i = 0; i < layers_id; i++) {
		//start with index 0 and it will be the input layer (also count as hidden layer) with the real inputs provided by argument
		//after will be index 1 and can be the next hidden layer or the final output layer
		//and always updating the _inputs variable with outputs to pass between layers as inputs
		outs = layer_think(layers[i], _inputs);

		//here is where the update occurs, copying outs into the _inputs variable
		//note: can't re-use outs variable because the size might not match
		for(j = 0; j < layers[i][N_max_neurons]; j++) {
			_inputs[j] = outs[j];
		}
	}

	//in the end of the loop have the outs variable with the result of the output layer because it's the last one
	return outs;
}

public Float:neural_think(Float:inputs[]) {
	static Float:outs[NEURAL_MAX_SIZE];

	//first convert our understandable input values to raw values with neural_input_values
	//second use these raw values to provide them into neural_think
	//finally use neural_output_values to convert the raw output into our understandable output values
	outs = neural_output_values(neural_think_raw(neural_input_values(inputs)));

	return outs;
}

public Float:neural_learn_raw(Float:inputs[], Float:outputs[]) {
	static o, l, h, n, w; //are not random letters, everyone are for loops index: output, layer, hidden, next and weight

	static output_layer[Layer];
	output_layer = layers[layers_id - 1]; //get output layer (that's, the last layer)

	static Float:mse; //to calculate the mean squared error
	mse = 0.0;

	static Float:actual_output[NEURAL_MAX_SIZE];
	actual_output = neural_think_raw(inputs);

	//output layer
	//loop neurons of output layer (1 neuron = 1 output, etc)
	for(o = 0; o < output_layer[N_max_neurons]; o++) {
		static neuron[Neuron];
		neuron = neurons[output_layer[N_id]][o]; //get neuron (just to shorten the variable)

		neuron[N_error] = _:(outputs[o] - actual_output[o]); //wanted - actual
		mse += neuron[N_error] * neuron[N_error]; //sum every error of every neuron of output layer

		neuron[N_delta] = _:(neuron[N_activation] * (1.0 - neuron[N_activation]) * neuron[N_error]); //sigmoid derivated * error

		neurons[output_layer[N_id]][o] = neuron; //update neuron
	}

	//hidden layers
	//loop hidden layers (-2 to avoid the output layer)
	for(l = layers_id - 2; l >= 0; l--) {
		static nexts[Layer];
		nexts = layers[l + 1]; //get neurons of next layer; with +1 we recover the output layer

		//loop neurons of hidden layer
		for(h = 0; h < layers[l][N_max_neurons]; h++) {
			static hidden[Neuron];
			hidden = neurons[l][h]; //get neuron

			//loop neurons of next layer
			hidden[N_error] = _:0.0;
			for(n = 0; n < nexts[N_max_neurons]; n++) {
				//on this way, neurons are connected with the next ones respectively
				hidden[N_error] += (neurons[nexts[N_id]][n][N_weights][h] * neurons[nexts[N_id]][n][N_delta]);
			}
			hidden[N_delta] = _:(hidden[N_activation] * (1.0 - hidden[N_activation]) * hidden[N_error]); //sigmoid derivated * error

			neurons[l][h] = hidden; //update neuron
		}

		for(n = 0; n < nexts[N_max_neurons]; n++) {
			static neuron[Neuron];
			neuron = neurons[nexts[N_id]][n]; //get neuron

			//finally learn!
			for(w = 0; w < neuron[N_max_inputs]; w++) {
				//the weights are updated based on the actual data and delta (activation * error)
				//also the learning will be more or less critical considering the custom learning rate
				//neuron[N_weights][w] += (neuron[N_data][w] * neuron[N_delta] * nexts[N_rate]);
				neuron[N_weights][w] += neuron_optimizer(neuron, nexts, w, neuron[N_data][w] * neuron[N_delta] * nexts[N_rate]);
			}

			//the bias is not affected directly by the data provided, it's the reason for being bias!
			neuron[N_bias] += neuron_optimizer(neuron, nexts, neuron[N_max_inputs], neuron[N_delta] * nexts[N_rate]);

			neurons[nexts[N_id]][n] = neuron; //update neuron
		}
	}

	//average error
	return mse / output_layer[N_max_neurons];
}

public Float:neural_learn(Float:inputs[], Float:outputs[]) {
	return neural_learn_raw(neural_input_values(inputs), outputs);
}

public neural_rate(Float:value) {
	static i;
	for(i = 0; i < layers_id; i++) {
		layers[i][N_rate] = _:value;
	}
}

public neural_momentum(Float:value) {
	static i;
	for(i = 0; i < layers_id; i++) {
		layers[i][N_momentum] = _:value;
	}
}

/*public Float:neural_values(Float:values[], bool:input) {
	static i, layer, len, Float:_values[NEURAL_MAX_SIZE];

	//layer id for input or output
	layer = input ? 0 : layers_id - 1;

	//input layer have the amount of inputs defined in N_max_inputs field
	//and the output layer have defined the total outputs in N_max_neurons
	len = input ? neurons[0][0][N_max_inputs] : layers[layer][N_max_neurons];

	//loop all values
	for(i = 0; i < len; i++) {
		//this is just (x - min) / (max - min) to convert a value into a defined range without restrictions
		//for example, you have min=0 and max=255, so [x=254 == ~0.996] and [x=256 == ~1.0039]
		_values[i] = (values[i] - layers[layer][N_input_min][i]) / (layers[layer][N_input_max] - layers[layer][N_input_min][i]);
	}

	return _values;
}*/

public Float:neural_input_values(Float:values[]) {
	static i, Float:_values[NEURAL_MAX_SIZE];

	//loop all values
	//the input layer have the amount of inputs defined in N_max_inputs field
	for(i = 0; i < neurons[0][0][N_max_inputs]; i++) {
		//this is just (x - min) / (max - min) to convert a value into a defined range without restrictions
		//for example, you have min=0 and max=255, so [x=254 == ~0.996] and [x=256 == ~1.0039]
		_values[i] = (values[i] - layers[0][N_input_min][i]) / (layers[0][N_input_max] - layers[0][N_input_min][i]);
	}

	return _values;
}

public Float:neural_output_values(Float:values[]) {
	static i, layer, Float:_values[NEURAL_MAX_SIZE];

	//layer id for output
	layer = layers_id - 1;

	//loop all values
	//the output layer have defined the total outputs in N_max_neurons
	for(i = 0; i < layers[layer][N_max_neurons]; i++) {
		//already explained in neural_input_values
		_values[i] = (values[i] - layers[layer][N_input_min][i]) / (layers[layer][N_input_max] - layers[layer][N_input_min][i]);
	}

	return _values;
}

/*
raw = neural_text("it's 5pm");
raw equal than Float:{ ... }
*/
/*
public Float:neural_text(str[]) {
	static i, max, Float:values[NEURAL_MAX_SIZE];

	for(i = 0, max = strlen(str); i < max; i++) {
		//32-126
	}

	return values;
}

public Float:neural_string_binary(str[]) {
	
}

public Float:neural_string_simple(str[]) {
	
}

public Float:neural_string_complex(str[]) {
	
}*/

public neural_input_range(index, Float:min, Float:max) {
	if(index > NEURAL_MAX_SIZE) {
		log_error(AMX_ERR_NATIVE, "index out of NEURAL_MAX_SIZE (%d/%d)", index, NEURAL_MAX_SIZE);
		return;
	}

	if(!layers_id) {
		log_error(AMX_ERR_NATIVE, "there is not input layer created");
		return;
	}

	layers[0][N_input_min][index] = _:min;
	layers[0][N_input_max][index] = _:max;
}

public neural_output_range(index, Float:min, Float:max) {
	if(index > NEURAL_MAX_SIZE) {
		log_error(AMX_ERR_NATIVE, "index out of NEURAL_MAX_SIZE (%d/%d)", index, NEURAL_MAX_SIZE);
		return;
	}

	if(layers_id < 2) {
		log_error(AMX_ERR_NATIVE, "there is not output layer created");
		return;
	}

	layers[layers_id - 1][N_input_min][index] = _:min;
	layers[layers_id - 1][N_input_max][index] = _:max;
}

public neural_save(name[]) {
	if(layers_id < 2) {
		log_error(AMX_ERR_NATIVE, "there is not enough layers created");
		return;
	}

	new key[32], buffer[2048];

	//max layers
	formatex(buffer, charsmax(buffer), "%d", layers_id);
	fvault_set_data(name, "max layers", buffer);

	//layers
	for(new l; l < layers_id; l++) {
		//layer info
		formatex(key, charsmax(key), "layer %d", l);
		
		formatex(buffer, charsmax(buffer), "%d", layers[l][N_max_neurons]);

		fvault_set_data(name, key, buffer);

		//the next values are part of layer info but would be better have it separated for more readability in the file

		//check if it is the input or output layer
		if(!l || l == layers_id - 1) {
			//this is already explained in neural_input_values and input_output_values
			new len = !l ? neurons[0][0][N_max_inputs] : layers[l][N_max_neurons];
			
			//loop all the inputs or outputs
			for(new i; i < len; i++) {
				//input/range info
				formatex(key, charsmax(key), "%s range %d", !l ? "input" : "output", i);

				formatex(buffer, charsmax(buffer), "%f %f", layers[l][N_input_min][i], layers[l][N_input_max][i]);

				fvault_set_data(name, key, buffer);
			}
		}
	}

	//neurons
	for(new l; l < layers_id; l++) {
		//neuron
		for(new n; n < layers[l][N_max_neurons]; n++) {
			//info
			formatex(key, charsmax(key), "neuron l:%d n:%d", l, n);

			formatex(buffer, charsmax(buffer), "%d", neurons[l][n][N_max_inputs]);
			for(new w; w < neurons[l][n][N_max_inputs]; w++) {
				format(buffer, charsmax(buffer), "%s %f", buffer, neurons[l][n][N_weights][w]);
			}
			format(buffer, charsmax(buffer), "%s %f", buffer, neurons[l][n][N_bias]);
			format(buffer, charsmax(buffer), "%s %f", buffer, neurons[l][n][N_activation]);
			format(buffer, charsmax(buffer), "%s %f", buffer, neurons[l][n][N_delta]);
			format(buffer, charsmax(buffer), "%s %f", buffer, neurons[l][n][N_error]);

			fvault_set_data(name, key, buffer);
		}
	}
}

public neural_load(name[]) {
	new key[32], temp[64], buffer[2048];
	
	//max layers
	if(!fvault_get_data(name, "max layers", buffer, charsmax(buffer))) {
		neural_reset();
		return false;
	}

	layers_id = str_to_num(buffer);

	//layers
	for(new l; l < layers_id; l++) {
		//layer info
		formatex(key, charsmax(key), "layer %d", l);

		if(!fvault_get_data(name, key, buffer, charsmax(buffer))) {
			neural_reset();
			return false;
		}

		layers[l][N_id] = l;
		layers[l][N_max_neurons] = str_to_num(buffer);
	}

	//neurons
	for(new l; l < layers_id; l++) {
		//neuron
		for(new n; n < layers[l][N_max_neurons]; n++) {
			//info
			formatex(key, charsmax(key), "neuron l:%d n:%d", l, n);

			if(!fvault_get_data(name, key, buffer, charsmax(buffer))) {
				neural_reset();
				return false;
			}

			_neural_str_parse(buffer, temp, charsmax(temp));
			neurons[l][n][N_max_inputs] = str_to_num(temp);

			for(new w; w < neurons[l][n][N_max_inputs]; w++) {
				_neural_str_parse(buffer, temp, charsmax(temp));
				neurons[l][n][N_weights][w] = _:str_to_float(temp);
			}

			_neural_str_parse(buffer, temp, charsmax(temp));
			neurons[l][n][N_bias] = _:str_to_float(temp);

			_neural_str_parse(buffer, temp, charsmax(temp));
			neurons[l][n][N_activation] = _:str_to_float(temp);

			_neural_str_parse(buffer, temp, charsmax(temp));
			neurons[l][n][N_delta] = _:str_to_float(temp);

			parse(buffer, temp, charsmax(temp));
			neurons[l][n][N_error] = _:str_to_float(temp);
		}

		//the next values are part of layer info but we were need the value N_max_inputs of the first neuron of the input layer

		//check if it is the input or output layer
		if(!l || l == layers_id - 1) {
			//this is already explained in neural_input_values and input_output_values
			new len = !l ? neurons[0][0][N_max_inputs] : layers[l][N_max_neurons];
			
			//loop all the inputs or outputs
			for(new i; i < len; i++) {
				//input/range info
				formatex(key, charsmax(key), "%s range %d", !l ? "input" : "output", i);

				if(!fvault_get_data(name, key, buffer, charsmax(buffer))) {
					neural_reset();
					return false;
				}

				_neural_str_parse(buffer, temp, charsmax(temp));
				layers[l][N_input_min][i] = _:str_to_float(temp);

				parse(buffer, temp, charsmax(temp));
				layers[l][N_input_max][i] = _:str_to_float(temp);
			}
		}
	}

	return true;
}

public neural_delete(name[]) {

}

public neural_reset() {
	
}

public _neural_string_cut(str[], pos) {
	new i, max = strlen(str);

	for(; i < max; i++) {
		if(i >= pos) {
			str[i - pos] = str[i];
		}
	}

	str[max - pos] = '^0';
}

public _neural_str_parse(str[], temp[], size) {
	parse(str, temp, size);
	_neural_string_cut(str, strlen(temp) + 1);
}

/*
if you want more performance, you must use the raw functions (neural_learn_raw, etc) but it's get a bit complicated

we understand numbers like "5.000 hp", "17 ping", "$800", etc
but neural networks understand the data in a different scale, they are more simple
we need calculate the raw value based on a range:
((x / y) * v)
where X it's the max for the neural, Y it's our max and V it's the value
and actually it's always ((1 / y) * v)

for example, one of the inputs is the health:
range of 0-100 with 57hp : the raw value ((1 / 100) * 57)  -> 0.57
range of 0-400 with 57hp : the raw value ((1 / 400) * 57)  -> 0.1425
range of 0-400 with 395hp: the raw value ((1 / 400) * 395) -> 0.9875
range of 0-400 with 420hp: the raw value ((1 / 400) * 420) -> 1.05

conclusion: it's like make the value lite and easy to learn for the neural network
and you are not limited by the max as you see in the last example with 420hp and 1.05, it's just a reference
*/
