# PPO + CNN Hex Agent Report

## 1. Introduction

This project implements an agent for the game Hex using a PPO-trained CNN policy. The
agent was developed in several stages. The first goal was to train a neural network to
choose good moves through reinforcement learning. Later, after testing showed that the
raw CNN policy was weak against a deterministic greedy opponent, the final version was
changed into a hybrid agent:

```text
PPO-trained CNN policy + Hex-specific tactical and positional safeguards
```

The final agent is therefore not a pure CNN-only player. PPO is used to train the CNN,
the CNN proposes moves, and a hand-designed Hex heuristic is used as a safety filter
when the CNN move is strategically worse.

## 2. What PPO And CNN Mean In This Project

PPO means Proximal Policy Optimization. It is the reinforcement learning algorithm used
to train the neural network. PPO is only used during training.

The CNN is the neural network that receives the board position and outputs move scores.
During actual play, PPO itself does not directly select moves. Instead, the trained CNN
selects moves using the policy it learned from PPO training.

So the relationship is:

```text
PPO trains the CNN.
The trained CNN produces move scores.
The final agent compares the CNN move with tactical Hex logic.
```

## 3. Board Encoding

The board is encoded from the perspective of the current player. Instead of giving the
network raw board values, the board is converted into three planes:

1. Current player's stones.
2. Opponent's stones.
3. Empty cells.

For an `N x N` board, the CNN input has shape:

```text
3 x N x N
```

This is useful because the network always sees the position as:

```text
my stones / opponent stones / empty cells
```

This avoids needing separate logic for red and blue.

## 4. CNN Architecture

The current CNN model is a small fully convolutional network. It has:

1. Three convolution layers.
2. A policy head.
3. A value head.

The general structure is:

```text
Input board
-> Conv2D + ReLU
-> Conv2D + ReLU
-> Conv2D + ReLU
-> Policy head
-> Value head
```

The policy head outputs one logit for each board cell. On an 11x11 board, this means:

```text
11 * 11 = 121 logits
```

The value head outputs one number estimating how good the current board position is for
the current player.

## 5. What Logits And Masked Logits Are

The CNN outputs raw move scores called logits. A higher logit means the CNN prefers that
move more.

However, the CNN outputs a score for every cell, including occupied cells. Occupied
cells are illegal moves. Therefore, before choosing a move, the code creates
`masked_logits`.

The idea is:

```text
legal moves keep their CNN score
illegal moves get score -1e9
```

Example:

```text
CNN logits:
cell 0:  2.1
cell 1:  0.5
cell 2:  3.0
cell 3: -0.2
```

If only cells `0` and `3` are legal:

```text
masked_logits:
cell 0:  2.1
cell 1: -1000000000
cell 2: -1000000000
cell 3: -0.2
```

Then the agent chooses the highest remaining legal score:

```python
best_action = torch.argmax(masked_logits).item()
```

So `masked_logits` means:

```text
CNN move scores after illegal moves have been removed.
```

## 6. PPO Training Procedure

During training, the model plays games and stores the data needed for PPO updates:

1. Encoded board state.
2. Selected action.
3. Old log probability of the action.
4. Value estimate.
5. Legal move mask.
6. Reward.

After the game ends, rewards are converted into discounted returns. The advantage is
computed as:

```text
advantage = return - value_estimate
```

PPO then compares the new policy probability with the old probability:

```text
ratio = new_probability / old_probability
```

This ratio is clipped:

```text
clipped_ratio = clamp(ratio, 1 - epsilon, 1 + epsilon)
```

The clipping prevents the model from changing too much in a single update. This makes
PPO more stable than a simple policy-gradient method.

The loss contains:

1. PPO policy loss.
2. Value loss.

The model is trained with the Adam optimizer.

## 7. Reward Design

The main reward is based on the final game result:

```text
+1 if the player eventually wins
-1 if the player eventually loses
```

Because Hex has sparse rewards, reward shaping was also used. Sparse reward means the
agent only receives the most important signal at the end of the game. This makes early
learning hard.

The shaping reward is based on connection advantage:

```text
opponent_connection_distance - own_connection_distance
```

If a move improves the current player's connection, it receives a small positive
shaping reward. If it makes the position worse, it receives a small negative shaping
reward.

The shaping reward is clipped so that it cannot become more important than actually
winning the game.

## 8. Reducing Shaping Over Time

The shaping reward was reduced during training. The reason is that shaping is useful
early, but the final goal should be winning the game, not only maximizing the heuristic.

The shaping schedule used was:

```text
0.10 -> 0.05 -> 0.02 -> 0.00
```

This means:

```text
early training: more guidance
late training: no shaping, only game outcome
```

This helps avoid the agent learning behavior that looks good according to the shaping
function but does not actually win.

## 9. Curriculum Learning

Training uses a curriculum. The model starts on easier tasks and gradually moves to the
full target setting.

The current curriculum is:

```python
CURRICULUM_PHASES = [
    (0.10, 5, "random", 0.0, 0.10),
    (0.25, 7, "epsilon_greedy", 0.3, 0.05),
    (0.60, 11, "greedy", 0.0, 0.02),
    (1.00, 11, "league", 0.0, 0.00),
]
```

This means:

1. First 10%: 5x5 board against random opponent.
2. Next phase: 7x7 board against epsilon-greedy opponent.
3. Middle phase: 11x11 board against greedy opponent.
4. Final phase: 11x11 league training.

The greedy phase was extended because testing showed that the model had difficulty
against the greedy baseline.

## 10. Old-Self Opponents

During training, the model saves older versions of itself as snapshots. These older
models are frozen and used as opponents later.

This is called old-self training.

The purpose is:

1. Prevent forgetting.
2. Add opponent diversity.
3. Make the current model play against previous strategies.

The code keeps up to six old-self opponents:

```python
MAX_OLD_SELF_MODELS = 6
```

When the training log says:

```text
Loaded 6 old-self opponents
```

it means the training script found six saved older model checkpoints and loaded them as
frozen opponents.

## 11. Baseline Opponents

The agent was evaluated against three baselines:

1. Random.
2. Epsilon-greedy.
3. Greedy.

The greedy baseline works like this:

1. If it can win immediately, it wins.
2. If the opponent can win immediately, it blocks.
3. Otherwise it chooses the move closest to the center.

Even though this is simple, it is a strong baseline because center control and immediate
tactics are useful in Hex.

## 12. What Was Tried Before The Final Approach

Several approaches were tested before the final hybrid agent.

### 12.1 Pure CNN-PPO

The first version relied mostly on the CNN-PPO policy. The agent used the CNN logits to
select moves after masking illegal actions.

This worked well against random opponents, but it was weak against greedy play. The
model learned legal moves and some basic patterns, but it did not reliably learn a
strong connection strategy.

An observed result before the final heuristic filter was:

```text
CNN-PPO win rate vs epsilon-greedy: about 42-43%
CNN-PPO win rate vs random: about 98%
CNN-PPO win rate vs greedy: 0%
```

This showed that the raw neural policy was not enough.

### 12.2 Stronger Curriculum Against Greedy

The curriculum was changed to train longer against greedy on 11x11 boards. The idea was
that if the model failed against greedy, it should see greedy more often during
training.

The league phase was also changed to sample greedy and epsilon-greedy more often.

This helped make the training objective more relevant, but it still did not solve the
main problem. The model still achieved 0% against greedy before the final heuristic
filter.

### 12.3 Reward Shaping

Connection-distance shaping was added so the model could receive intermediate feedback
before the end of the game.

The idea was:

```text
reward moves that improve own connection
penalize moves that improve opponent connection
```

This helped guide training, but it was not enough by itself to make the raw CNN beat the
greedy baseline.

### 12.4 Bridge Reward

A bridge reward was also considered and added as a small shaping signal:

```python
bridge_reward = 0.01 * (new_bridges - old_bridges)
```

In Hex, a bridge is a strong two-stone pattern where two stones are indirectly connected
through two shared empty cells. Rewarding bridge creation gives the agent more Hex
knowledge during training.

This is useful as training guidance, but like connection shaping, it is still indirect.
The CNN must learn from noisy reinforcement learning updates, so the effect is slower
than directly using the heuristic at move-selection time.

### 12.5 Larger ResNet-Style CNN

A deeper CNN with residual blocks was also tried. The goal was to increase model
capacity. A residual CNN is still a CNN, but it has skip connections that make deeper
networks easier to train.

The attempted model was a 6-block ResNet-style CNN.

However, training became unstable and produced non-finite values such as NaNs in the
logits, losses, or gradients. This caused errors like:

```text
Categorical expected valid logits but found NaN values
```

Stabilization ideas included:

1. Lower learning rate.
2. Gradient clipping.
3. Group normalization.
4. Skipping non-finite optimizer steps.
5. Separating old-self checkpoint folders for incompatible architectures.

The larger model was promising in theory, but it made training harder and less stable in
the available setup.

### 12.6 GAE, Entropy Bonus, And Symmetry Augmentation

Other PPO improvements were considered or experimented with:

1. Entropy bonus.
2. Generalized Advantage Estimation (GAE).
3. State/action symmetry augmentation.

Entropy bonus encourages exploration. GAE can produce better advantage estimates.
Symmetry augmentation uses Hex board symmetries to create more training examples.

These are good ideas for a stronger PPO system, but the most important practical issue
remained that the final agent needed to avoid obvious strategic mistakes against the
greedy baseline.

## 13. Final Move Selection

The final agent selects moves in this order:

```text
1. Check immediate win.
2. Check immediate block.
3. Ask the trained CNN for its preferred move.
4. Ask the hand-designed Hex heuristic for its preferred move.
5. Compare both moves with the heuristic score.
6. Choose the heuristic move unless the CNN move has a better heuristic score.
```

This means the CNN is still used, but it is not trusted blindly.

The final decision rule is:

```python
if positional_score >= model_score:
    return positional_move

return model_move
```

So if the hand-designed heuristic move is better than or equal to the CNN move, the
agent plays the heuristic move. The CNN move is only used when it receives a better
heuristic score.

## 14. Immediate Win And Block

At the beginning of the game, the immediate win check usually does nothing because no
one can win in one move on an empty 11x11 board.

This part becomes important later in the game.

The logic is:

```text
Try every legal move.
If placing a stone there completes a winning path, play it.
```

The block check uses the same idea for the opponent:

```text
If the opponent could win next turn, block that cell now.
```

So early game usually goes to the CNN-versus-heuristic comparison, while late game often
uses the immediate win/block logic.

## 15. Hand-Designed Hex Heuristic

The hand-designed heuristic is the part that uses explicit Hex knowledge. It is not
learned by the CNN.

It includes:

1. Immediate win detection.
2. Immediate block detection.
3. Shortest connection distance.
4. Bridge counting.
5. Center tie-breaking.

The connection score is:

```text
10 * (opponent_distance - own_distance)
+ own_bridges
- opponent_bridges
```

This score rewards positions where:

1. The current player is closer to connecting their sides.
2. The opponent is farther from connecting their sides.
3. The current player has useful bridge structures.
4. The opponent has fewer bridge structures.

## 16. Why The Heuristic Is Not Inside The CNN

The CNN does not automatically know shortest paths or bridge rules. It only receives
board planes and learns from examples and rewards.

The heuristic can be used in several ways:

1. At play time as a safety filter.
2. During training as reward shaping.
3. As extra feature planes for the CNN.
4. As supervised pretraining targets.

The current final version uses the first method strongly and uses shaping during
training. This gives good practical performance quickly.

A future improvement would be supervised pretraining:

```text
generate positions -> let heuristic choose move -> train CNN to imitate heuristic
-> fine-tune with PPO
```

That would make the CNN itself learn more of the heuristic behavior.

## 17. Final Results

After adding the tactical and positional heuristic filter, the final agent achieved:

```text
CNN-PPO win rate vs epsilon-greedy: 100.00%
CNN-PPO win rate vs random: 100.00%
CNN-PPO win rate vs greedy: 100.00%
```

These results are for the complete final agent, not the raw CNN alone.

The correct description is:

```text
Hybrid PPO-trained CNN agent with Hex-specific heuristic safeguards.
```

## 18. Strengths

The final approach has several strengths:

1. PPO gives a reinforcement learning framework for improving the CNN policy.
2. The CNN can learn spatial board patterns.
3. Action masking prevents illegal moves.
4. Curriculum learning makes training easier.
5. Reward shaping gives useful intermediate signals.
6. Old-self opponents create a more diverse training environment.
7. Tactical safeguards prevent simple losses.
8. Connection-distance and bridge heuristics make the final agent much stronger against
   deterministic baselines.

## 19. Limitations

The main limitation is that the final 100% win rates are not produced by the raw CNN
alone. The hand-designed heuristic is an important part of the final move selection.

This means the final agent is not a pure neural PPO agent. It is a hybrid system.

Another limitation is that the CNN still has limited capacity and training time. A
stronger pure neural agent would likely need:

1. A more stable residual network.
2. More training games.
3. Better advantage estimation.
4. Supervised pretraining from heuristic moves.
5. Possibly Monte Carlo tree search.

## 20. Conclusion

The project started as a CNN-PPO Hex agent. The model was trained with reinforcement
learning, curriculum learning, shaped rewards, action masking, and old-self opponents.
Early versions performed well against random play but failed against the greedy
baseline.

Several improvements were explored, including stronger greedy training, bridge reward,
larger residual networks, entropy bonus, GAE, and symmetry augmentation. The most
effective final change was to combine the PPO-trained CNN with Hex-specific tactical and
positional safeguards.

The final agent should be described as:

```text
PPO-trained CNN with hand-designed Hex heuristic filtering.
```

This approach is practical and strong for the tested opponents because it combines
learning-based move selection with reliable Hex knowledge.
