# PPO-CNN Hex Agent Report

## 1. Project Overview

This project implements an agent for the game Hex using a hybrid strategy:

```text
PPO-trained CNN policy + Hex-specific tactical and positional safeguards
```

The final agent is not a pure neural-network-only player. The neural network learns useful spatial move preferences, but the final decision also uses explicit Hex knowledge such as immediate wins, blocking moves, connection distance, bridges, and center preference.

The final system can be described as a PPO-trained CNN agent with Hex-specific tactical safeguards. PPO trains the CNN, the CNN proposes moves, and the final agent filters the proposed move using tactical and positional Hex logic.

## 2. What PPO And CNN Mean In This Project

PPO stands for Proximal Policy Optimization. In this project, PPO is the reinforcement learning algorithm used during training.

The CNN is the neural network that receives the board state and outputs move scores.

Important distinction:

```text
PPO is used during training.
The trained CNN is used during actual play.
```

So during a real game, PPO is not actively optimizing anymore. The saved model `ppo_cnn_hex.pt` contains the learned CNN weights.

The relationship is:

```text
PPO trains the CNN.
The CNN learns a policy.
The final agent uses that trained policy plus Hex heuristics.
```

## 3. Board Encoding

The board is encoded from the perspective of the current player. Instead of giving the CNN raw values, the board is converted into three input planes:

```text
1. Current player's stones
2. Opponent's stones
3. Empty cells
```

For an `N x N` board, the input shape is:

```text
3 x N x N
```

For an 11x11 board, this becomes:

```text
3 x 11 x 11
```

This makes the input consistent for Red and Blue. The CNN always sees the position as:

```text
my stones / opponent stones / empty cells
```

That avoids needing completely separate logic for each player.

## 4. CNN Architecture

The CNN is a small fully convolutional model. It has:

```text
1. Three convolution layers
2. A policy head
3. A value head
```

The structure is:

```text
Input board
-> Conv2D + ReLU
-> Conv2D + ReLU
-> Conv2D + ReLU
-> Policy head
-> Value head
```

The policy head outputs one score for each board cell. On an 11x11 board:

```text
11 * 11 = 121 move scores
```

These scores are called logits. A higher logit means the CNN prefers that move more.

The value head outputs one number estimating how good the current board state is for the current player. This value estimate is used during PPO training to compute advantages.

## 5. Action Masking

The CNN outputs a score for every cell, including occupied cells. But occupied cells are illegal moves.

To solve this, the agent uses action masking:

```text
Legal moves keep their CNN score.
Illegal moves receive a very large negative score, -1e9.
```

Then the model samples or chooses only from legal moves.

This is important because the policy should not waste probability on moves that cannot be played.

Action masking ensures that the PPO policy only learns from legal actions and that the final agent does not intentionally select occupied cells.

## 6. PPO Training Loop

The training process is based on self-contained game episodes. Each episode plays one full Hex game.

During a model-controlled turn, the training code stores:

```text
1. Board state
2. Chosen action
3. Old log probability of that action
4. Value prediction
5. Legal action mask
6. Current player
7. Reward
```

The model uses a categorical probability distribution over legal actions:

```text
masked logits -> Categorical distribution -> sampled action
```

This means the agent does not always choose the highest-scoring move during training. It samples from the probability distribution, which allows exploration.

## 7. Rewards

The main reward is based on the final game result:

```text
+1 if the player wins
-1 if the player loses
```

This is the most important reward because the real goal is winning the game.

However, Hex has sparse rewards. The agent may make many moves before it receives the final win/loss result. That makes it hard to know which earlier moves were good or bad. This credit assignment problem motivated the use of reward shaping.

## 8. Reward Shaping During Training

Reward shaping is one of the important training components in this project. It gives the agent small intermediate feedback during the game instead of waiting only for the final win/loss reward.

The purpose of reward shaping was to make early learning easier. Hex is a connection game, so the shaping signal was based on whether a move improved the current player's connection and made the opponent's connection harder.

The shaping reward is based on connection advantage:

```text
opponent_connection_distance - own_connection_distance
```

The idea is:

```text
If my connection becomes shorter, the move is better.
If the opponent's connection becomes longer, the move is better.
If my move makes my position worse, the reward decreases.
```

In the training code, the connection advantage is measured before and after the selected move.

The code compares the connection advantage before and after the move:

```text
shaping_reward = shaping_scale * (new_advantage - old_advantage)
```

This reward is clipped:

```text
maximum shaping reward = 0.2
minimum shaping reward = -0.2
```

Clipping is important because shaping should guide learning, but it should not become more important than actually winning.

Reward shaping therefore acted as a training guide. It gave intermediate feedback about whether a move improved the player's connection, but the main objective remained winning the game.

## 9. Reward Shaping Schedule

The shaping reward was reduced during the curriculum:

```text
0.10 -> 0.05 -> 0.02 -> 0.00
```

This means:

```text
Early training: more guidance
Late training: less guidance
Final phase: only win/loss reward
```

This prevents the model from overfitting to the heuristic. The final goal should not be just to maximize connection distance; the final goal should be to win.

The shaping scale was strongest at the beginning because the model needed more guidance. It was then reduced over time so the final policy focused on actual game outcomes instead of only optimizing the heuristic.

## 10. PPO Return And Advantage Calculation

After a game ends, rewards are converted into discounted returns using:

```text
gamma = 0.99
```

The return is the future reward expected from each move:

```text
return_t = reward_t + gamma * reward_(t+1) + gamma^2 * reward_(t+2) + ...
```

Then the advantage is computed:

```text
advantage = return - value_prediction
```

The advantage tells PPO whether an action was better or worse than the value head expected.

If the advantage is positive:

```text
The action was better than expected, so PPO should increase its probability.
```

If the advantage is negative:

```text
The action was worse than expected, so PPO should decrease its probability.
```

The advantages are normalized when there is more than one move:

```text
advantages = (advantages - mean) / standard_deviation
```

This helps stabilize training.

## 11. PPO Clipped Objective

PPO uses the probability ratio between the new policy and the old policy:

```text
ratio = exp(new_log_probability - old_log_probability)
```

This ratio tells how much the policy changed for the selected action.

PPO uses two surrogate objectives:

```text
surrogate_1 = ratio * advantage
surrogate_2 = clipped_ratio * advantage
```

The clipping range is:

```text
clip epsilon = 0.2
```

So the ratio is clipped to:

```text
0.8 to 1.2
```

This is the main PPO idea: the policy improves gradually, but the clipping mechanism prevents overly large policy updates. This makes training more stable than basic policy gradient methods.

## 12. PPO Loss Function

The training loss has two main parts:

```text
1. Policy loss
2. Value loss
```

The policy loss improves move selection:

```text
policy_loss = negative clipped PPO objective
```

The value loss trains the value head:

```text
value_loss = (return - predicted_value)^2
```

The final loss is:

```text
loss = policy_loss + 0.5 * value_loss
```

The value loss is weighted by `0.5` so it helps the network learn board evaluation without dominating policy learning.

The implementation also uses gradient clipping:

```text
max gradient norm = 1.0
```

This prevents very large gradients from destabilizing training.

The optimizer is Adam:

```text
learning rate = 0.0003
```

Each episode's collected data is reused for:

```text
PPO epochs = 4
```

This means the model updates four times on the collected episode data.

## 13. Curriculum Learning

The model was trained using curriculum learning. Instead of starting directly with the hardest 11x11 setting, the agent gradually moved from easier tasks to harder tasks.

The curriculum was:

```text
First 10%: 5x5 board vs random opponent, shaping scale 0.10
Until 25%: 7x7 board vs epsilon-greedy opponent, epsilon 0.3, shaping scale 0.05
Until 60%: 11x11 board vs greedy opponent, shaping scale 0.02
Final 40%: 11x11 league phase, shaping scale 0.00
```

The purpose was:

```text
1. Learn basic connection behavior on small boards.
2. Add more diverse positions using epsilon-greedy play.
3. Train against stronger tactical behavior on 11x11.
4. Finish with a mixed league of opponents.
```

## 14. What Was Considered During Curriculum Learning

The curriculum did not only consider board size, opponent type, and reward shaping. It also considered the following eight design factors.

1. Overall difficulty progression

The training started with easier situations and gradually moved to harder ones. The agent first played on a smaller 5x5 board, where games are shorter and connection patterns are easier to discover. Then it moved to 7x7, and finally to the target 11x11 board. This avoided giving the model the hardest version of the problem immediately.

2. Opponent randomness

Opponent randomness was controlled using epsilon in the epsilon-greedy opponent. A higher epsilon means the opponent sometimes plays random moves. This creates more varied board positions and helps exploration. A lower epsilon makes the opponent more tactical and predictable.

In the curriculum:

```text
epsilon 0.3 was used in the 7x7 epsilon-greedy phase
epsilon 0.2 was used in part of the final league phase
epsilon 0.0 was used for greedy play
```

Opponent randomness exposed the model to diverse positions early in training. Later phases reduced randomness so the agent had to handle stronger tactical play.

3. Phase length

The phases were not all the same length. The curriculum spent more time on the harder 11x11 phases because those were closer to the final target environment.

The phase schedule was:

```text
First 10%: easy 5x5 random phase
Next 15%: 7x7 epsilon-greedy phase
Next 35%: 11x11 greedy phase
Final 40%: 11x11 league phase
```

The greedy phase was especially important because testing showed that the raw CNN struggled against greedy tactical play. More training time was therefore assigned to the difficult 11x11 phases because they were more relevant to final agent performance.

4. Final league mixture

In the final phase, the agent did not train against only one opponent. It trained against a league-style mixture:

```text
45% greedy
30% epsilon-greedy
15% old-self
10% self-play
```

This prevents the agent from overfitting to a single opponent style. Greedy opponents test tactical defense, epsilon-greedy opponents create variety, old-self opponents preserve older strategies, and self-play lets the model challenge itself.

5. Old-self training

Old versions of the model were saved and reused as frozen opponents. This means the current model had to play against earlier versions of itself.

This helped with:

```text
Preventing forgetting
Increasing opponent diversity
Making the model improve beyond its previous strategies
```

Old-self training helped the agent avoid forgetting earlier strategies because it continued to face previous versions of itself.

6. Self-play

Self-play means the current model controls both players. This was used in the final phase, after the model had already learned basic behavior.

Self-play is useful because the opponent can improve as the model improves. However, it was not used too early because early self-play can produce weak or noisy games.

7. Legal action masking

Action masking was used throughout training. The CNN outputs a score for every board cell, but occupied cells are illegal. The mask gives illegal cells a score of `-1e9`, so the model only samples legal moves.

This matters in curriculum learning because the board size changes. On 5x5, 7x7, and 11x11 boards, the number of possible actions changes, but the same masking idea keeps the action selection valid.

8. Stable PPO hyperparameters

The PPO settings were kept stable while the curriculum changed the environment difficulty.

Important PPO settings were:

```text
gamma = 0.99
clip epsilon = 0.2
PPO epochs = 4
learning rate = 0.0003
gradient clipping max norm = 1.0
```

Keeping these stable made it easier to understand the effect of the curriculum. The environment became harder over time, but the PPO update method stayed consistent.

In summary, the curriculum considered difficulty progression, opponent randomness, phase length, final league mixture, old-self training, self-play, legal action masking, stable PPO hyperparameters, reward shaping schedule, board size, and opponent type.

## 15. Old-Self Training

During training, older versions of the model are saved as snapshots. These older models are frozen and later used as opponents.

The code starts old-self snapshots after:

```text
4000 episodes
```

It saves snapshots every:

```text
500 episodes
```

It keeps up to:

```text
6 old-self models
```

The purpose of old-self training is:

```text
1. Prevent forgetting
2. Add opponent diversity
3. Make the current model beat previous versions of itself
```

Old-self training gave the agent a moving set of opponents, so it could not overfit to only random or greedy play.

## 16. Self-Play

In self-play, both sides are controlled by the current model.

This is useful because the agent can generate training games from its own current strategy. As the model improves, the self-play games can also become more challenging.

Self-play was used in the final league phase, not from the beginning. That is important because early self-play can be low quality if the model has not learned useful behavior yet.

## 17. Final Move Selection Strategy

During actual play, the final agent chooses moves in this order:

```text
1. Check immediate win.
2. Check immediate block.
3. Ask the trained CNN for its preferred move.
4. Ask the hand-designed Hex heuristic for its preferred move.
5. Compare the CNN move and heuristic move.
6. Choose the heuristic move unless the CNN move receives a better heuristic score.
```

The final decision rule is:

```python
if positional_score >= model_score:
    return positional_move

return model_move
```

This means the CNN is used, but it is not trusted blindly. The CNN proposes a learned move, while the heuristic acts as a safety filter. If the heuristic move is at least as good, the agent chooses the safer positional move.

## 18. Immediate Win And Block

The first tactical check is immediate winning:

```text
Try every legal move.
If placing a stone there completes my winning path, play it.
```

The second tactical check is immediate blocking:

```text
Try every legal move for the opponent.
If the opponent could win next turn, block that move.
```

These checks matter more in the late game. In the early game, no one can usually win immediately, so the agent normally continues to the CNN and heuristic comparison.

## 19. Positional Hex Heuristic

The hand-designed heuristic uses explicit Hex strategy.

It includes:

```text
1. Shortest connection distance
2. Bridge counting
3. Center tie-breaking
```

The main connection score is:

```text
10 * (opponent_distance - own_distance)
+ own_bridges
- opponent_bridges
```

This rewards positions where:

```text
1. My connection path is shorter.
2. The opponent's connection path is longer.
3. I have useful bridge structures.
4. The opponent has fewer bridge structures.
```

The shortest connection distance is calculated using a path search where:

```text
My stones cost 0.
Empty cells cost 1.
Opponent stones are blocked.
```

So a lower distance means fewer empty cells are needed to complete a connection.

## 20. Bridges

A bridge is a strong Hex pattern where two stones are indirectly connected through shared empty cells.

Bridges are useful because they create flexible connection threats. If the opponent blocks one part, the player often still has another way to connect.

The heuristic rewards the agent for having more bridges and penalizes positions where the opponent has more bridges. Bridge counting gives the agent explicit knowledge of Hex structure because bridges are one of the important tactical patterns in the game.

## 21. Why The Final Agent Is Hybrid

The pure CNN-PPO policy was able to learn useful patterns, but it struggled against simple tactical greedy play.

The final solution was to combine:

```text
Learned neural policy
+ Tactical win/block checks
+ Positional connection heuristic
```

This made the agent more reliable.

The honest explanation is:

```text
The final agent is not a pure PPO agent. It is a hybrid PPO-trained CNN agent with hand-designed Hex heuristic filtering.
```

This is a strength because it shows an engineering improvement based on testing.

## 22. Move Time

The agent's move time can change depending on the board position and opponent.

It depends on:

```text
1. Number of legal moves remaining
2. Whether an immediate win is found
3. Whether an immediate block is found
4. Whether the CNN has to be used
5. How expensive the positional heuristic is on that board
```

The opponent affects move time indirectly because different opponents create different board positions.

For example:

```text
Against random: positions may be less structured.
Against greedy: positions may create more tactical checks.
Against self-play: both sides may create stronger connection patterns.
```

The first move may also be slower because the model can be loaded from `ppo_cnn_hex.pt`. Later moves are faster because the model is cached.

## 23. Complete Strategy Summary

This project uses PPO to train a CNN policy for Hex. The board is encoded as three planes: current player's stones, opponent stones, and empty cells. The CNN outputs one move score for every board position, and action masking removes illegal moves.

During training, PPO samples legal moves, stores log probabilities and value estimates, and updates the policy using the clipped PPO objective. The reward is mainly +1 for winning and -1 for losing, and reward shaping based on connection advantage is also used to help the model learn earlier in the game. This shaping is gradually reduced so the final model focuses on actual wins.

The curriculum starts with small boards and random opponents, then moves to larger boards, greedy opponents, and finally a league phase with greedy, epsilon-greedy, old-self, and self-play opponents. This makes training more stable and reduces overfitting.

During actual play, the final agent first checks for an immediate win, then blocks the opponent's immediate win. If there is no tactical emergency, the CNN proposes a move and a Hex heuristic proposes another move. The heuristic evaluates connection distance, bridges, and center position. The agent chooses the heuristic move unless the CNN move receives a better heuristic score.

The final strategy is hybrid: PPO provides learned spatial pattern recognition, while explicit Hex logic provides tactical safety and stronger positional play.

## 24. Important Limitations

The main limitation is that the final strong performance comes from the hybrid system, not from the raw CNN alone.

The CNN learned useful behavior, but the tactical and positional safeguards were important for avoiding obvious mistakes.

Another limitation is that the model is relatively small. A stronger future version could use:

```text
1. A deeper residual CNN
2. More training episodes
3. More self-play
4. Better opponent league management
5. Extra feature planes for connection distance or bridge information
6. Imitation learning from the heuristic before PPO fine-tuning
```

## 25. Conclusion

The final agent is a hybrid PPO-CNN Hex agent. PPO was used to train a CNN policy. The CNN receives the board as three planes: current player's stones, opponent stones, and empty cells. It outputs a score for every possible board cell, and illegal moves are masked out.

Training used win/loss reward, plus reward shaping based on connection advantage. This helped the agent learn before the final game result. The shaping was reduced over time, and the curriculum moved from smaller boards and easier opponents to 11x11 league training with greedy, epsilon-greedy, old-self, and self-play opponents.

At play time, the agent first checks whether it can win immediately, then whether it must block the opponent. If not, the CNN proposes a move. A Hex heuristic also proposes a move based on shortest connection distance, bridges, and center preference. The final move is chosen by comparing both candidates with the heuristic score.

The agent therefore combines reinforcement learning, CNN pattern recognition, and explicit Hex strategy.
