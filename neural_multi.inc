#include fvault

const NEURAL_MAX_INPUTS = 64;
const NEURAL_MAX_NEURONS = 64;
const NEURAL_MAX_OUTPUTS = 64;
const NEURAL_MAX_LAYERS = 6;

enum Neuron {
	Float:N_weights[NEURAL_MAX_NEURONS],
	Float:N_bias,
	N_max_inputs,
	Float:N_data[NEURAL_MAX_INPUTS],
	Float:N_activation,
	Float:N_delta,
	Float:N_error
};

enum Layer {
	N_id,
	N_max_neurons
}

new layers_id;
new layers[NEURAL_MAX_LAYERS][Layer];
new all_layers[NEURAL_MAX_LAYERS][NEURAL_MAX_NEURONS][Neuron];

//neurons
public neuron_create(total_inputs) {
	new neuron[Neuron];

	neuron[N_max_inputs] = total_inputs;

	for(new i; i < total_inputs; i++) {
		neuron[N_weights][i] = _:random_float(-1.0, 1.0);
	}

	neuron[N_bias] = _:random_float(-1.0, 1.0);

	return neuron;
}

public Float:neuron_think(neuron[Neuron], Float:inputs[]) {
	new i, Float:out;

	for(i = 0; i < neuron[N_max_inputs]; i++) {
		neuron[N_data][i] = _:inputs[i];
	}
	
	out = neuron[N_bias];

	for(i = 0; i < neuron[N_max_inputs]; i++) {
		out += inputs[i] * neuron[N_weights][i];
	}

	return neuron[N_activation] = _:(1 / (1 + floatpower(2.718281828459045, -out)));
}

//layers
public layer_create(total_neurons, total_inputs) {
	layers[layers_id][N_id] = layers_id;
	layers[layers_id][N_max_neurons] = total_neurons;

	for(new i; i < total_neurons; i++) {
		all_layers[layers_id][i] = neuron_create(total_inputs);
	}

	layers_id++;
}

public Float:layer_think(layer[Layer], Float:inputs[]) {
	static i, Float:outs[NEURAL_MAX_OUTPUTS];

	for(i = 0; i < layer[N_max_neurons]; i++) {
		outs[i] = neuron_think(all_layers[layer[N_id]][i], inputs);
	}

	return outs;
}

//networks
stock neural_layer(total_neurons, total_inputs = 0) {
	total_inputs = total_inputs > 0 ? total_inputs : layers[layers_id - 1][N_max_neurons];

	layer_create(total_neurons, total_inputs);
}

public Float:neural_think(Float:inputs[]) {
	static i, j, Float:outs[NEURAL_MAX_OUTPUTS];
	static Float:_inputs[NEURAL_MAX_NEURONS];

	for(j = 0; j < all_layers[0][0][N_max_inputs]; j++) {
		_inputs[j] = inputs[j];
	}

	for(i = 0; i < layers_id; i++) {
		outs = layer_think(layers[i], _inputs);

		for(j = 0; j < layers[i][N_max_neurons]; j++) {
			_inputs[j] = outs[j];
		}
	}

	return outs;
}

public Float:neural_learn(Float:inputs[], Float:outputs[], Float:rate) {
	static o, l, h, i, w;

	static output_layer[Layer];
	output_layer = layers[layers_id - 1];

	static Float:mse;
	mse = 0.0;

	static Float:actual_output[NEURAL_MAX_OUTPUTS];
	actual_output = neural_think(inputs);

	for(o = 0; o < output_layer[N_max_neurons]; o++) {
		static neuron[Neuron];
		neuron = all_layers[output_layer[N_id]][o];

		neuron[N_error] = _:(outputs[o] - actual_output[o]);
		mse += floatpower(neuron[N_error], 2.0);
		neuron[N_delta] = _:(neuron[N_activation] * (1.0 - neuron[N_activation]) * neuron[N_error]);

		all_layers[output_layer[N_id]][o] = neuron;
	}

	for(l = layers_id - 2; l >= 0; l--) {

		for(h = 0; h < layers[l][N_max_neurons]; h++) {
			static neuronH[Neuron];
			neuronH = all_layers[l][h];

			static neurons_next[Layer];
			neurons_next = layers[l + 1];

			neuronH[N_error] = _:0.0;

			for(i = 0; i < neurons_next[N_max_neurons]; i++) {
				neuronH[N_error] += (all_layers[neurons_next[N_id]][i][N_weights][h] * all_layers[neurons_next[N_id]][i][N_delta]);

				static neuron[Neuron];
				neuron = all_layers[neurons_next[N_id]][i];

				for(w = 0; w < neuron[N_max_inputs]; w++) {
					neuron[N_weights][w] += (neuron[N_data][w] * neuron[N_delta] * rate);
				}
				neuron[N_bias] += (neuron[N_delta] * rate);

				all_layers[neurons_next[N_id]][i] = neuron;
			}

			neuronH[N_delta] = _:(neuronH[N_activation] * (1.0 - neuronH[N_activation]) * neuronH[N_error]);

			all_layers[l][h] = neuronH;
		}
	
	}

	return mse / output_layer[N_max_neurons];
}

public Float:neural_normalize(Float:inputs[]) {
	new Float:min, Float:max;

	for(new i; i < all_layers[0][0][N_max_inputs]; i++) {
		min = inputs[i] < min ? inputs[i] : min;
		max = inputs[i] > max ? inputs[i] : max;
	}

	for(new i; i < all_layers[0][0][N_max_inputs]; i++) {
		inputs[i] = (inputs[i] - min) / (max - min);
	}

	return inputs;
}

public neural_denormalize() {
	//
}

stock Float:neural_range(Float:v, Float:min = 0.0, Float:max = 1.0) {
	return (v - min) / (max - min);
}

/*stock Float:neural_range_values(Float:values[], Float:min = 0.0, Float:max = 1.0) {
	for(new i; i < all_layers[0][0][N_max_inputs]; i++) {
		values[i] = (values[i] - min) / (max - min);
	}

	return values;
}*/

public neural_reset() {
	//
}

public neural_save(name[]) {
	new key[32], export[2048];

	//max layers
	formatex(export, charsmax(export), "%d", layers_id);
	fvault_set_data(name, "max layers", export);

	//layers
	for(new l; l < layers_id; l++) {
		//layer info
		formatex(key, charsmax(key), "layer %d", l);
		formatex(export, charsmax(export), "%d", layers[l][N_max_neurons]);
		fvault_set_data(name, key, export);
	}

	//neurons
	for(new l; l < layers_id; l++) {
		//neuron
		for(new n; n < layers[l][N_max_neurons]; n++) {
			//info
			formatex(key, charsmax(key), "neuron l:%d n:%d", l, n);

			formatex(export, charsmax(export), "%d ", all_layers[l][n][N_max_inputs]);
			for(new w; w < all_layers[l][n][N_max_inputs]; w++) {
				format(export, charsmax(export), "%s%f ", export, all_layers[l][n][N_weights][w]);
			}
			format(export, charsmax(export), "%s%f ", export, all_layers[l][n][N_bias]);
			format(export, charsmax(export), "%s%f ", export, all_layers[l][n][N_activation]);
			format(export, charsmax(export), "%s%f ", export, all_layers[l][n][N_delta]);
			format(export, charsmax(export), "%s%f", export, all_layers[l][n][N_error]);

			fvault_set_data(name, key, export);
		}
	}
}

public neural_load(name[]) {
	new key[32], temp[64], import[2048];
	
	//max layers
	if(!fvault_get_data(name, "max layers", import, charsmax(import))) {
		neural_reset(); //no need but return false means neural resetted
		return false;
	}

	layers_id = str_to_num(import);

	//layers
	for(new l; l < layers_id; l++) {
		//layer info
		formatex(key, charsmax(key), "layer %d", l);

		if(!fvault_get_data(name, key, import, charsmax(import))) {
			neural_reset();
			return false;
		}

		layers[l][N_id] = l;
		layers[l][N_max_neurons] = str_to_num(import);
	}

	//neurons
	for(new l; l < layers_id; l++) {
		//neuron
		for(new n; n < layers[l][N_max_neurons]; n++) {
			//info
			formatex(key, charsmax(key), "neuron l:%d n:%d", l, n);

			if(!fvault_get_data(name, key, import, charsmax(import))) {
				neural_reset();
				return false;
			}

			_neural_str_parse(import, temp, charsmax(temp));
			all_layers[l][n][N_max_inputs] = str_to_num(temp);

			for(new w; w < all_layers[l][n][N_max_inputs]; w++) {
				_neural_str_parse(import, temp, charsmax(temp));
				all_layers[l][n][N_weights][w] = _:str_to_float(temp);
			}

			_neural_str_parse(import, temp, charsmax(temp));
			all_layers[l][n][N_bias] = _:str_to_float(temp);

			_neural_str_parse(import, temp, charsmax(temp));
			all_layers[l][n][N_activation] = _:str_to_float(temp);

			_neural_str_parse(import, temp, charsmax(temp));
			all_layers[l][n][N_delta] = _:str_to_float(temp);

			parse(import, temp, charsmax(temp));
			all_layers[l][n][N_error] = _:str_to_float(temp);
		}
	}

	return true;
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
	//return temp; //don't works fine :s
}
