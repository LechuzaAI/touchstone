def generalized_advantage_estimate(buffer, gamma, gae_lambda, T):
    deltas = buffer.rewards[:-1] + gamma * buffer.values[1:] - buffer.values[:-1]
    gae = 0
    for t in reversed(range(T - 1)):
        gae = deltas[t] + gamma * gae_lambda * gae
        buffer.advantage_estimates[t] = gae
