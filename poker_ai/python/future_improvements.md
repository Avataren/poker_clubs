# Future Improvements

## Training

### ~~Huber loss delta scaling~~ (DONE)
Implemented configurable `huber_delta` (default 10.0). Squared error for <10 BB
mispredictions, linear for larger swings. Exposed as `--huber-delta` CLI arg.

### ~~Gradient clipping~~ (DONE)
Implemented `clip_grad_norm_` with max_norm=10.0 on both BR and AS networks.
Prevents gradient spikes from large Q-value errors during early training.

### Prioritized experience replay for BR buffer
Currently uniform sampling from circular buffer. Prioritized replay (proportional to
TD error) would focus training on surprising transitions — hands where the Q-value
prediction was most wrong. Expect faster convergence but adds implementation complexity
and slight CPU overhead.

### ~~Lower epsilon_start~~ (DONE)
Default changed from 0.12 to 0.06. Sufficient exploration for 9 actions, less replay
buffer pollution.

### Separate LR schedules for BR and AS
Currently both piggyback on the same cosine schedule. BR might benefit from a faster
initial LR (learns value estimation) while AS needs a consistently low LR (stability
of the average). Could decouple with independent cosine schedules.

## Architecture

### ~~Dueling DQN~~ (DONE)
Implemented dueling architecture: V(s) + A(s,a) - mean(A) in PokerNet value head.
Value stream learns situation quality, advantage stream learns marginal action differences.

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
