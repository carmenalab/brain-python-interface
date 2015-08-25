## ******* Raw Data ******
## 
## In [15]: pos.continuous_move(10000, 10000, 10000)
## c move: 5184, 3359, 4011
## 
## Limit switches read: 110101
## 
## 
## In [16]: pos.continuous_move(-10000, -10000, -10000)
## c move: 5060, 3367, 3985
## 
## Limit switches read: 111011
## 
## 
## In [17]: pos.continuous_move(10000, 10000, 10000)
## c move: 5183, 3359, 3994
## 
## Limit switches read: 110101
## 
## 
## In [18]: pos.continuous_move(-10000, -10000, -10000)
## c move: 5044, 3367, 4000
## 
## Limit switches read: 111010
## 
## 
## In [19]: pos.continuous_move(10000, 10000, 10000)
## c move: 5183, 3359, 3994
## 
## Limit switches read: 110101
## 
## 
## In [20]: pos.continuous_move(-10000, -10000, -10000)
## c move: 5044, 3367, 742
## 
## Limit switches read: 111011
## 
## 
## In [21]: pos.continuous_move(-10000, -10000, -10000)
## c move: 2, 0, 3248
## 
## Limit switches read: 110111
## 
## 
## In [22]: pos.continuous_move(10000, 10000, 10000)
## c move: 5184, 3358, 3995
## 
## Limit switches read: 111101
## 
## 
## In [23]: pos.continuous_move(-10000, -10000, -10000)
## c move: 5044, 3367, 490
## 
## Limit switches read: 111010
## 
## 
## In [24]: pos.continuous_move(-10000, -10000, -10000)
## c move: 4, 0, 3488
## 
## Limit switches read: 110111

############################## End Raw Data #####################################
import numpy as np
n_steps_min_to_max = np.vstack([
	(5184, 3359, 4011),
	(5183, 3359, 3994),
	(5183, 3359, 3994),
	(5184, 3358, 3995),
])

n_steps_max_to_min = np.vstack([
	(5060, 3367, 3985),
	(5044, 3367, 4000),
	(5044, 3367, 742+3248),
	(5044, 3367, 490+3488)
])
