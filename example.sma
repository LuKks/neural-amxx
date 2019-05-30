#include <amxmodx>

#include neural_multi

public plugin_init() {
	register_plugin("Neural network example", "1.0.0", "LuKks");

	//layers
	neural_layer(8, 3); //(neurons, inputs) -> hidden/input layer
	//neural_layer(8); //(neurons) -> hidden layer
	//neural_layer(8); //(neurons) -> can add more hidden layers
	neural_layer(1); //(outputs) -> output layer is the last one
	
	//inside the include, at the beginning, there are maximums setted
	//for example, 6 layers maximum, can modify it

	//this is required if you will use neural_learn or neural_think instead of _raw
	neural_input_range(0, 0.0, 255.0); //0 -> first input
	neural_input_range(1, 0.0, 255.0); //1 -> second input
	neural_input_range(2, 0.0, 255.0); //2 -> third input

	neural_output_range(0, 0.0, 1.0); //0 -> first output
	//0.0-1.0 is the default but anyway to be clear

	new Float:rate = 0.25; //learning rate; depends on the layers, neurons, iterations, etc

	new Float:outputs[NEURAL_MAX_SIZE];

	//benchmark();

	//we query to the network if 255, 255 and 255 is light or dark
	outputs = neural_think(Float:{ 255.0, 255.0, 255.0 });
	server_print("255, 255, 255 [1.0] -> %f", outputs[0]); //random value
	
	//we query to the network if 0, 0 and 0 is light or dark
	outputs = neural_think(Float:{ 0.0, 0.0, 0.0 });
	server_print("0, 0, 0 [0.0] -> %f", outputs[0]); //random value

	for(new i, Float:mse; i < 5001; i++) { //iterations
		mse = 0.0;
	
		//two ways to pass data
		//raw
		mse += neural_learn_raw(Float:{ 1.0, 0.0, 0.0 }, Float:{ 1.0 }, rate); //255, 0, 0

		//automatic (using the range predefined), between 8% and 27% less efficient
		mse += neural_learn(Float:{ 0.0, 255.0, 0.0 }, Float:{ 1.0 }, rate);

		//must be in range with min and max in the first neural_layer
		mse += neural_learn(Float:{ 0.0, 0.0, 255.0 }, Float:{ 1.0 }, rate); //light
		mse += neural_learn(Float:{ 0.0, 0.0, 0.0 }, Float:{ 0.0 }, rate); //dark
		mse += neural_learn(Float:{ 100.0, 100.0, 100.0 }, Float:{ 1.0 }, rate); //light
		mse += neural_learn(Float:{ 107.0, 181.0, 255.0 }, Float:{ 0.0 }, rate); //dark
		mse += neural_learn(Float:{ 0.0, 53.0, 105.0 }, Float:{ 0.0 }, rate); //dark
		mse += neural_learn(Float:{ 150.0, 150.0, 75.0 }, Float:{ 1.0 }, rate); //light
		mse += neural_learn(Float:{ 75.0, 75.0, 0.0 }, Float:{ 0.0 }, rate); //dark
		mse += neural_learn(Float:{ 0.0, 75.0, 75.0 }, Float:{ 0.0 }, rate); //dark
		mse += neural_learn(Float:{ 150.0, 74.0, 142.0 }, Float:{ 1.0 }, rate); //light
		mse += neural_learn(Float:{ 50.0, 50.0, 75.0 }, Float:{ 0.0 }, rate); //dark
		mse += neural_learn(Float:{ 103.0, 22.0, 94.0 }, Float:{ 0.0 }, rate); //dark
	
		mse /= 13; //simple average of the errors in each learning
		
		//don't really need to get the mse (medium square error)
		//can teach him no matter the error

		if(mse < 0.01) { //stop learn when mean squared error reach this limit
			server_print("mse threshold on iter %d", i);
			break;
		}

		if(!(i % 1000)) {
			server_print("iter %d, mse %f", i, mse);
		}
	}

	//a simple network can't learn any pattern, only linear ones
	//a multilayer network can solve non linear patterns!
	//OR is linear and XOR isn't -> https://vgy.me/dSbEu0.png
	
	//two ways to think data
	outputs = neural_think_raw(Float:{ 0.952, 0.701, 0.039 }); //raw (243, 179, 10)
	server_print("243, 179, 10 [1.0] -> %f", outputs[0]); //0.999930

	outputs = neural_think(Float:{ 75.0, 50.0, 50.0 }); //automatic (using the range predefined)
	server_print("75, 50, 50 [0.0] -> %f", outputs[0]); //0.022316
    
	outputs = neural_think(Float:{ 95.0, 99.0, 104.0 });
	server_print("95, 99, 104 [1.0] -> %f", outputs[0]); //0.961524
    
	outputs = neural_think(Float:{ 65.0, 38.0, 70.0 });
	server_print("65, 38, 70 [0.0] -> %f", outputs[0]); //0.018278

	//if have a lot outputs, you can ->
	/*for(new i; i < layers[layers_id - 1][N_max_neurons]; i++) {
		server_print("output %d -> %f", i, outputs[i]);
	}*/

	//also can use
	//neural_save("zombie-npc");
	//neural_load("zombie-npc");

	//after load, you can't create more layers (neural_layer) and nothing involved to config
	//if have a neural created and you load a new neural, it will be overwritted, so you can do it
}

public benchmark() {
	new time_start, time_end;

	//raw
	time(_, _, time_start);
	for(new i; i < 200000; i++) { //this takes 7s
		//then this takes ~0.000035s (1/3 of a millisecond)
		//but my cpu is extremely low: AMD A-10 7860K
		neural_think_raw(Float:{ 1.0, 1.0, 1.0 });
	}
	time(_, _, time_end);
    
	server_print("neural_think_raw = %d -> %d", time_start, time_end); //33 -> 40 (7s)

	//automatic
	time(_, _, time_start);
	for(new i; i < 200000; i++) { //this takes 9s
		//then this takes ~0.000045s (less than half millisecond)
		neural_think(Float:{ 255.0, 255.0, 255.0 });
	}
	time(_, _, time_end);
    
	server_print("neural_think = %d -> %d", time_start, time_end); //24 -> 33 (9s)
	
	//with this results I'm able to make real time thinking without problem
	//and with a decent cpu you will too
}