# Future Improvements

## Training

### Huber loss delta scaling
Currently using `smooth_l1_loss` with default delta=1.0, but rewards are in big blinds
(typical range ±100 BB at 100BB deep). This means almost all TD errors exceed the delta,
so the loss is effectively MAE — losing the quadratic penalty near zero that helps with
fine-grained value accuracy. Consider delta=10.0 to get squared error for small
mispredictions (<10 BB) and linear for large ones.

### Gradient clipping
Not currently used. With large Q-values from properly scaled rewards, gradient norms
may spike during early training. Consider `torch.nn.utils.clip_grad_norm_` with
max_norm=10.0 to stabilize.

### Prioritized experience replay for BR buffer
Currently uniform sampling from circular buffer. Prioritized replay (proportional to
TD error) would focus training on surprising transitions — hands where the Q-value
prediction was most wrong. Expect faster convergence but adds implementation complexity
and slight CPU overhead.

### Separate LR schedules for BR and AS
Currently both piggyback on the same cosine schedule. BR might benefit from a faster
initial LR (learns value estimation) while AS needs a consistently low LR (stability
of the average). Could decouple with independent cosine schedules.

## Architecture

### Dueling DQN
Split the value head into V(s) + A(s,a) streams. The advantage stream learns relative
action values while the state value stream learns position value. Should improve
Q-value accuracy, especially for states where most actions have similar value.

### Noisy nets for exploration
Replace epsilon-greedy with NoisyLinear layers in the BR value head. Learned exploration
is more efficient than random — the network explores where it's uncertain rather than
uniformly. Would eliminate the epsilon schedule entirely.

## Evaluation

### Position-aware eval reporting
Break down bb/100 by button vs big blind to catch positional strategy imbalances
earlier. Currently tracked per-seat but not prominently surfaced.

### Eval against previous checkpoints
Periodically evaluate current AS against AS from N episodes ago to measure monotonic
improvement. A regression suggests the strategy is oscillating rather than converging.
