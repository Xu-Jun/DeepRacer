# DeepRacer
AWS DeepRacer Reward Function &amp; Action Space

# left/left-right/right model
## Desc: 
Drive on left lane if turn left, drive on right lane if turn right

## Reward Function:
```
from numpy import log as ln
def reward_function(params):
    reward = 0
    
    if params['is_offtrack']:
        reward -= 20

    if params['all_wheels_on_track']:
        reward += 1.5
    else:
        reward += 0.001
    
    if params['steering_angle'] < 0:
        if params['is_left_of_center']:
            reward -= 0.2
        else:
            reward += 0.2
    else:
        if params['is_left_of_center']:
            reward += 0.3
        else:
            reward -= 0.3
    
    reward += params['speed']/10.0
    
    if params['progress'] > 2:
        reward += ln(params['progress'])
    else:
        reward += 0.5
    
    return float(reward)
```
## Hyperparameter
Gradient descent batch size	64
Entropy	0.01
Discount factor	0.99
Loss type	Huber
Learning rate	0.0003
Number of experience episodes between each policy-updating iteration	20
Number of epochs	10

## Action Space
No.
Steering angle (°)
Speed (m/s)
0	-15.0	2.00
1	-10.0	3.00
2	0.0	2.00
3	0.0	3.00
4	0.0	4.00
5	5.0	3.00
6	10.0	2.50
7	15.0	2.00
8	20.0	1.50
9	30.0	1.00

# Waypoints Model

# Race Line Model

