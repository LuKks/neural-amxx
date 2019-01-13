# neural-amxx
Passing old posts to GitHub.

English: https://forums.alliedmods.net/showthread.php?t=310446

Spanish: https://amxmodx-es.com/Thread-Inteligencia-artificial-redes-neuronales

As developers, we can __create neural networks__, there are many types of networks and incredible uses.

A neural network is basically a neurons set connected to each other and the form or complexity of connection will vary its type.

In AMXX it seems that there is no API to create a neural network, that is what we are going for. Also I replicated a version in JavaScript.

Let's start with an example and then more advanced explanations.
```c++
#include <amxmodx>

#include neural_simple

public plugin_init() {
	register_plugin("Neural network (simple layer); example", "1.0.0", "LuKks");

	new neural[Neural];
	neural = neural_create(2); //crea la red con 2 inputs (entradas)
	
	//will be a random result at the beginning
	server_print("2+2 -> %f", neural_think(neural, Float:{ 2.0, 2.0 }) * 100);

	for(new i; i < 1000; i++) { //1.000 repetitions
		//Float:{ input1, input2 }, output
		//we teach him some basic sums
		neural_learn(neural, Float:{ 1.0, 1.0 }, 0.02);
		neural_learn(neural, Float:{ 1.0, 0.0 }, 0.01);
		neural_learn(neural, Float:{ 0.0, 1.0 }, 0.01);
		//a network has to know that 1 + 0 = 1
		//and 0 + 1 = 1, since they are two different operations

		//the results are in decimals, 0.04 means 4
		neural_learn(neural, Float:{ 2.0, 2.0 }, 0.04);
		//more examples = more precise learning

		//internally, the network is managed with 0-1

		/*neural_learn(neural, Float:{ 2.0, 1.0 }, 0.03); 
		neural_learn(neural, Float:{ 1.0, 2.0 }, 0.03);
		neural_learn(neural, Float:{ 0.0, 2.0 }, 0.02);*/
	}

	//	1.000 | 50.000 (repetitions)
	
	// 19.014957 | 18.999979
	server_print("15+4 -> %f", neural_think(neural, Float:{ 15.0, 4.0 }) * 100);
	//we handle the outputs in numbers like 0.04, then multiply by 100

	// -3.950722 | -3.999993
	server_print("-9+5 -> %f", neural_think(neural, Float:{ -9.0, 5.0 }) * 100);
	
	// 79.519 | 79.999908
	server_print("38+42 -> %f", neural_think(neural, Float:{ 38.0, 42.0 }) * 100); 
}
```

At first the network did not know how much it was 2 + 2 but after a little learning with some simple sums we teach the addition pattern, that way you can then handle __sums that we do not teach you__.
This exercise is very easy for the network because it does not present a difficult pattern.

You can invent a simple pattern and teach it to the network.

__You can try to teach him something else__, for example:
```
{ inputs }, output
{ 1.0,  2.0 }, 3.0
{ 2.0,  3.0 }, 4.0
{ 3.0,  4.0 }, 5.0
{ 4.0,  5.0 }, 6.0
{ 5.0,  6.0 }, 7.0
```
Teach that and then ask about 6 + 7, you should answer with almost 8.

But try to teach him something difficult, the powers, that is, 4^3 and so on, here he can not learn it.
The reason is because this API (neural_simple.inc) gives you a one-layer network (to say the least).
But relax, I have done neural_multi.inc: D (example below).

By the way, the outputs at the end of learning, __do not get to be exact 19.0, nor -4.0 and 80.0__. It has his explanation.

But first, __what is a layer?__
Well, this simple network could be seen in the following way (the image has three entries but the example has two):
https://vgy.me/r8Wq1s.png

You don't see the word "layer" because the same input layer is the only one, that's why it's simple.

A neural network with multiple layers is composed of __an input layer, 1 or more hidden layers and an output layer__.
All layers can have __different amounts of neurons__.

For example, a network could be; input layer (which is also hidden) with 3 inputs and 3 neurons, a hidden layer with 4 other neurons and an output layer with 2 neurons (equivalent to 2 outputs).
Visually, it would be ->
https://vgy.me/X2S2KN.png

Just to show that there may be more hidden layers, here we have 6 inputs, a hidden layer with 4 neurons, another hidden layer with 3 neurons and an output layer with a neuron.
https://vgy.me/mtYddt.png

Okay, everything nice but what's the use? because __sums is not very useful__.

__Remember what an NPC is?__ Non-Player Character

You could create a neural network (not simple) so that it manages all the NPCs of your game or mod in a realistic way.
And it would not be very difficult to do it, I mean, to teach the network you should just play as if you were the NPC and make him learn your movements.

Neural networks require a lot of processing (always depends on the case) but once trained, it is not a large computation that is used to use them and __they are exportable to any other environment__.
So, once you have your expert network in the situation, you can __save him memory (number of layers, neurons, him weights and bias, etc)__ to use it even in another programming language or in any other part.

In addition, you can always __collect the information first and then perform the learning__ on another platform (or same), for example, some desktop program (maximum performance) or if you know JavaScript you have the same implementation attached.

Let's see an __example for the multi layer__.
```c++
#include <amxmodx>

#include neural_multi

public plugin_init() {
	register_plugin("Neural network (multi layer); example", "1.0.0", "LuKks");

	//layers
	neural_layer(8, 2); //(neurons, inputs) -> hidden/input layer
	neural_layer(8); //(neurons) -> hidden layer
	//neural_layer(2); //(neurons) -> can add more hidden layers
	neural_layer(1, -1); //(neurons, -1) -> output layer
	
	//inside the include, at the beginning, there are maximums setted
	//for example, 6 layers maximum, can modify it

	for(new i, Float:mse; i < 5000; i++) { //5.000 limit iterations
		mse += neural_learn(Float:{ 0.0, 0.0 }, Float:{ 0.0 }, 0.45);
		mse += neural_learn(Float:{ 1.0, 0.0 }, Float:{ 1.0 }, 0.45);
		mse += neural_learn(Float:{ 0.0, 1.0 }, Float:{ 1.0 }, 0.45);
		mse += neural_learn(Float:{ 1.0, 1.0 }, Float:{ 0.0 }, 0.45);
		mse /= 4; //simple average of the errors in each learning
		
		//0.45 is the learning rate
		//don't really need to get the mse (medium square error)
		//can teach him no matter the error

		if(mse < 0.00025) { //stop learn when mean squared error reach this limit
			server_print("mse threshold on iter %d", i);
			break;
		}

		if(!(i % 1000)) {
			server_print("iter %d, mse %f", i, mse);
		}
	}

	//in theory, a simple network can't learn this pattern
	//because technically it's not a linear problem
	//a multilayer network can solve non linear patterns!
	//OR is linear and XOR isn't -> https://vgy.me/dSbEu0.png

	new Float:outputs[NEURAL_MAX_OUTPUTS];
	outputs = neural_think(Float:{ 0.0, 0.0 });
	server_print("0.0 [0.0] -> %f", outputs[0]); //0.014664

	outputs = neural_think(Float:{ 1.0, 0.0 });
	server_print("1-0 [1.0] -> %f", outputs[0]); //0.992533

	//if have a lot outputs, you can ->
	/*for(new i; i < layers[layers_id - 1][max_neurons]; i++) {
		server_print("output %d -> %f", i, outputs[i]);
	}*/

	//also, can use
	//neural_save("zombie-npc");
	//neural_load("zombie-npc");

	//after load, you can't create more layers (neural_layer) and nothing involved to config
	//if have a neural created and you load a new neural, it will be overwritted, so you can do it
}
```

This is just a beginning, as I said before, there are __many types of neural networks__.
These two are the ones that I have used the most, for other types I have always helped myself from libraries or systems already done because for me *it is not worth learning and re-creating so much functionality*.
The current systems always served me __as learning and to get greater reasoning__.

A neural network claims to be equal to or better than a human.
A neural network __will not distracted, doesn't rest, doesn't look the other way__, etc.
A network has a margin of __error__ (it's not probability, that is different).
__How does the margin of error works?__ Well, you saw it right at the beginning with the simple network (and again with multi-layer).
```
-3.999993 instead of -4.0
79.999908 instead of 80.0
```
A multi-layer network could reach exact values in that case but in most cases you will not want that because for real uses, getting to an error of zero is very fucked up (or impossible), usually a very low error like 19.014957 for 19.0 is ok but the case depends.

A network that intends to perform surgical operations can't have so much margin of error. So much? In other words, can you have a small margin in that case? well, the humans don't move with 100% accuracy, *as long as the network is the same or better there should not be a problem, right?*

__Why is not there a neural network that simulates a human brain?__
An adult human brain *(not so much, as long as it's not newborn)* have so many neurons and __too many__ synapses, that is, neural connections that make it impossible for a technical matter to equalize the capacity of necessary computation.
Surely there are attempts with less capacity but not there to be close but still it was achieved and is achieving much with the current. We have to keep a close eye on the advances in quantum computing (although I don't know if that will work, I have not read about it).

__See this comparison of number of neurons.__
```
Bee = 960.000
Cat = 300.000.000
Chimpanzee = 6.200.000.000
Human = 100.000.000.000
```
Source of quantities (sorry, it's in Spanish): https://psicologiaymente.com/neurociencias/cuantas-neuronas-tiene- cerebro-humano

__How do neural networks learn?__
The version __neural_simple.inc is great to learn and understand the concept__, so that they don't bother to understand the code in the include, I will make a __textual example__.

As you know, a simple network has only one layer with a certain number of neurons (ignore the bias, a technical question of mathematics/ algebra).

A neuron has multiple weights but in a simple network let's say that "each weight simulates each neuron", is just a way of saying.
So, let's say that this simple network has 2 inputs and two neurons, so, the weights of a neuron would be ->
```
Weights = [0.21749, -0.18305]
```
Why these values? they are really random, __all neurons start with random values__.
Therefore, the network at the beginning gives you the wrong results.

__How is the current output calculated for this neuron?__
Let's say that our two inputs are 0.5 and 0.5 ->
```
Output = 0.5 * Weights [0] + 0.5 * Weights [1]
Output -> 0.01722
```

We wanted to do 0.5 + 0.5 and the network currently tells us that it is 0.01722.

You have to teach him that it is a little higher value, how is it corrected?
Modifying the weights with a learning rate, for example 0.01.

In addition to the learning rate, there is also the margin of error with respect to current entries.
```
Error = 1.0 - 0.01722
```
That is -> desired value - current value

__I had written the whole procedure but it did not convince me, I feel that it would be something complicated to understand textually.__
To simplify the process, what happens is that with some mathematical operations, all the weights of the current neuron are modified to approximate values ​​corresponding to a lower error (also considering the learning rate).
Therefore, having multiple neurons a network can learn multiple knowledge, since some __neurons learn more or less depending on their current error__.

That way there will come a time when __weights will be adjusted__ in values that at the time of entering similar inputs will correspond to correct outputs.
In case you did not notice, it's basically __brute force__.

__Same learning, different weights?__
It's interesting that you see the following, I created the neural network of the multi-layer example twice and I saved both, so, both networks produce practically the same result but that is how they look internally (open both images in two tabs to better see the difference) ->
https://vgy.me/Xp47Zz.jpg
https://vgy.me/MXfnoF.jpg
It's interesting to see that very different weights produce similar results.

__Why is it difficult to handle so many neurons?__
You will imagine that having a thousand neurons, __it will not be easy to find the right weights__, even if you are giving the simplest pattern that you can think of (to say it in some way).
Like I said, it's brute force so it will take as long as your processing power.

The humans, after __interacting with something multiple times you'll understand it more and more__.
"Practice makes a master" is a typical phrase (at least on Latin America) but technically correct.

As I said at the beginning, there are many types of networks. For example, a multi-layer perceptron neural network with back-propagation can handle patterns, image recognition and surely many other uses, probably for translation of languages according to my memory.
Surely driving a car can also, is that there are so many types of networks that sometimes there is a better solution but it is understood.

__Learn about neural networks__ (Google, YouTube, etc):
You can always __Google__ to delve into every detail, although I recommend __YouTube__ because it's more visual, it will be difficult to understand everything to pure text and mathematics (well, it will depends of your weights, haha).
To understand how a neural network would do to recognize images -> https://www.youtube.com/watch?v=aircAruvnKk
It occurred to me to look for the videos that I saw about two years ago, it's like a small course -> https://www.youtube.com/watch?v=jaEIv_E29sk
Now that I remember, I like an example that gives, if you don't see it, it was something like ->
In psychology a professional has many patient profiles, so, a neural network could learn everything about patients and in that way, the network can predict information for current and future patients.
More or less I modified it a lot to what he said but it is understood.

__Regarding the examples, you can always make a comment and ask me.__
If it's a problem that you are having with a plugin maybe you should create a separate publication (send me a private message with the link to the post because I don't check the new ones, thank you) but if it is a specific question to the code or the topic, I think you could directly comment it so the rest see your question as well.

Note: I was not allowed to upload .js, so I called it neural_javascript.sma for the JavaScript one. I did it similar to amxx, I know that the js can be improved a lot but it was not the point.

__Requires fvault.inc__, download -> https://forums.alliedmods.net/attachment.php?attachmentid=47365&d=1297052495

__Changelog__
* __Initial__
- I removed the version with dynamic arrays (multiple reasons, especially performance and code readability).
- To the neural_multi version I added two useful functions: neural_normalize and neural_range.
- I also want to do neural_denormalize, the function is created but it has no content, if someone wants to do the inverse of normalize, I would appreciate it.
- Added neural_save and neural_load, requires fvault.
- I changed the enums with the prefix N_ * so that they do not conflict with their plugins (N_id, N_error, etc.).
- Note: I do not plan to add the same functionalities to neural_simple because it's too limited, I consider it a library to learn in a basic way how a simple neural network works internally, you can always ask.
