enum Neural {
	N_max_inputs,
	Array:N_weights,
	Float:N_bias,
	Float:N_rate
}; //prefix "N_" to avoid conflicts

public neural_create(total_inputs) {
	new neural[Neural];

	neural[N_max_inputs] = total_inputs;
	neural[N_weights] = _:ArrayCreate(1, total_inputs);
	neural[N_bias] = _:1.0;
	neural[N_rate] = _:0.01;

	for(new i; i < total_inputs; i++) {
		ArrayPushCell(neural[N_weights], random_float(-neural[N_bias], neural[N_bias]));
	}

	return neural;
}

public Float:neural_think(neural[Neural], Float:inputs[]) {
	new Float:out = neural[N_bias];

	for(new i; i < neural[N_max_inputs]; i++) {
		out += inputs[i] * Float:ArrayGetCell(neural[N_weights], i);
	}

	return out;
}

public Float:neural_learn(neural[Neural], Float:inputs[], Float:target) {
	new Float:err = target - neural_think(neural, inputs);
	
	for(new i; i < neural[N_max_inputs]; i++) {
		ArraySetCell(neural[N_weights], i, Float:ArrayGetCell(neural[N_weights], i) + (inputs[i] * err * neural[N_rate]));
	}

	neural[N_bias] += err * neural[N_rate];

	return err;
}