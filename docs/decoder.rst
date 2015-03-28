Decoders
========
This segment describes implementational details of riglib.bmi.Decoder


Architecture
------------
..	image:: decoder_internal.png

A typical decoder object will have a filter (``filt``) and a state-space model (``ssm``). The filter encapsulates the mathematical decoding algorithm which infers a hidden state $x_t$ from observations $y_t$, without regard to what these variables represent in the specific application. The state-space model supplies the details about what each element of the hidden state represents (e.g., horizontal position, angular velocity of second joint, etc.). The decoder is thus the combination of the mathematical inference algorithm and the interpretation of the hidden state vector. This division of labor is intended to allow decoders to be constructed to control different plants by simply swapping the state-space models and while retaining the same inference algorithm, which is agnostic to the state interpretation details.

Hidden Markov model filters
---------------------------
The dominant approach in BMI inference alorithms centers around hidden-markov models



Serializing (saving) and deserializing decoder/filter objects
-------------------------------------------------------------
Decoder objects must be serialized (saved) so that the same decoder can be used for different tasks, e.g., training a decoder using CLDA in one block and then using it in a future block, or using the same decoder parameters across days. 

Decoder objects are serialized using Python's ``pickle`` module. To supply custom serialization and deserialization routines for the pickling/unpickling of a type of object, Python allows you to override the __getstate__ and __setstate__. (For a good guide on magic functions in general, see http://www.rafekettler.com/magicmethods.html). Overriding the __getstate__ method allows you to specify which attributes of the decoder you wish to serialize. It may not always be appropriate to serialize all attribibutes of the decoder (the default behavior)

