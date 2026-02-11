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

### Lower epsilon_start
Currently 0.12 — means ~1 in 8 BR actions are random, which adds significant noise
to the replay buffer early on. Previous run used 0.06. For heads-up with 9 actions,
0.06 is likely sufficient exploration. The higher value doesn't hurt convergence long
term but delays early Q-value accuracy.

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

## Deployment

### Temperature-based difficulty scaling
Use a single trained model for all difficulty levels by scaling AS policy logits
before softmax at inference time: `probs = softmax(logits / temperature)`.

| Difficulty | Temperature | Effect |
|---|---|---|
| Easy | 3.0 | Loose, passive, frequent mistakes |
| Medium | 1.5 | Plays recognizable poker but makes errors |
| Hard | 1.0 | Full-strength trained policy |
| Expert | 0.7 | Sharper than training — exploits marginal edges |

Implementation: single multiply in the backend ONNX inference path, before the
softmax. Expose as a difficulty setting per bot. No extra checkpoints or models
needed — one ONNX file serves all difficulty levels.

Can also combine with epsilon injection (force X% random actions) for a different
flavor of weakness: temperature makes the bot loose/passive, epsilon makes it
chaotic/unpredictable.

## Evaluation

### Position-aware eval reporting
Break down bb/100 by button vs big blind to catch positional strategy imbalances
earlier. Currently tracked per-seat but not prominently surfaced.

### Eval against previous checkpoints
Periodically evaluate current AS against AS from N episodes ago to measure monotonic
improvement. A regression suggests the strategy is oscillating rather than converging.
