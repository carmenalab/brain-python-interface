Feature extractors
==================
[This section is still incomplete]

Feature extractors are used by all BMIs to convert 'raw' data into 'observations' suitable for the decoding algorithm. 

In the ``BMILoop`` class, which is the parent of all BMI tasks, the feature extractor is instantiated in the function ``BMILoop.create_feature_extractor``

Simulation feature extractors are instantiated a little differently from regular feature extractors. These are instantiated with an argument being the 'task' object. This enables simulation features to be generated based on the state of the task (since real neural activity must relate to the current state of the task, otherwise BMI cannot work), without changing the rest of the task code. This enables more transparent simulations. 