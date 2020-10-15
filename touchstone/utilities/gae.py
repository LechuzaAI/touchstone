def generalized_advantage_estimate(buffer, gamma, gae_lambda, T):
    delta = buffer.rewards[:-1] + gamma * buffer.values[1:] - buffer.values[:-1]
    discounts = [1]
    [discounts.append((gamma * gae_lambda) ** (T - t + 1)) for t in range(T)]
    return discounts
