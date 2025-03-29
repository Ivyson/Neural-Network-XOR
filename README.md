# Neural-Network-XOR
This repo has the Nueral Network which was trained to perform an XOR Operation on a 2 inputted array. 
> The Documentation and Upgrades from this is still pending.

- A changelog will be attached to track changes made, TO-DO list to list all the functionalities to be introduced in future 

## Adam Optimiser 
- An optimisation algorithm used to update the learning rate, biases and the weights of the nueral network by moving the learning rate to find the best global minimum instead of being stuck on the local minimum.. This is really useful for cases whereby you might be having more than 1 minimum for your dataset. This algorithm uses the combinations of Momemntum and root mean square propagation.
### Formula

$m_t = 0   \text{First Moment Vector} $

$v_t = 0   \text{Second Momemnt Vector}$

$ t = 0  \text{Time step} $

1. To update the rules for each time step $t$ :
   - Compute the gradient $nabla \theta$
  
     
        $m_t = \beta_1 \times m_{t-1} +(1-\beta_2)\times \nabla \theta_t$

   
        $v_t = \beta_2 \times v_{t-1}+ (1-\beta_2)\times \(\nabla \theta_t\)^2$
