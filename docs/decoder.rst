Decoders
========
This segment describes implementational details of riglib.bmi.Decoder


Architecture
------------
..	image:: decoder_internal.png

A typical decoder object will have a filter (``filt``) and a state-space model (``ssm``). The filter encapsulates the mathematical decoding algorithm which infers a hidden state $x_t$ from observations $y_t$, without regard to what these variables represent in the specific application. The state-space model supplies the details about what each element of the hidden state represents (e.g., horizontal position, angular velocity of second joint, etc.). The decoder is thus the combination of the mathematical inference algorithm and the interpretation of the hidden state vector. This division of labor is intended to allow decoders to be constructed to control different plants by simply swapping the state-space models and while retaining the same inference algorithm, which is agnostic to the state interpretation details.

Hidden Markov model filters
---------------------------
[this section is incomplete]

The dominant approach in BMI inference alorithms centers around hidden-markov models



Serializing (saving) and deserializing decoder/filter objects
-------------------------------------------------------------
Decoder objects must be serialized (saved) so that the same decoder can be used for different tasks, e.g., training a decoder using CLDA in one block and then using it in a future block, or using the same decoder parameters across days. 

When saving a decoder, not all attributes of the decoder must be saved. For example, the current state of the BMI, which lives in the Decoder object, should not be saved. This is because the next time you re-open the Decoder object, you may want it to start from a different state. Hence, custom seriialization and deserialization must be used so that we can pick and choose which attributes get saved to file.

Decoder objects are serialized using Python's ``pickle`` module. To supply custom serialization and deserialization routines for the pickling/unpickling of a type of object, Python allows you to override the ``__getstate__`` and ``__setstate__``. (For a good guide on magic functions in general, see http://www.rafekettler.com/magicmethods.html). Overriding the ``__getstate__`` method allows you to specify which attributes of the decoder you wish to serialize. It may not always be appropriate to serialize all attribibutes of the decoder (the default behavior)

Since not all attributes of the decoder are pickled, some attributes must get default values. These should be specified in either the ``__setstate__`` methods or in the ``_pickle_init`` method. Since some of the deserialization functionality is also needed when creating a brand new decoder object, common code between deserialization and creating a new object should reside in ``_pickle_init``. 