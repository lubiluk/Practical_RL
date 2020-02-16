
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """

    next_states = mdp.get_next_states(state, action)

    Q = 0

    for s, p in next_states.items():
        r = mdp.get_reward(state, action, s)
        Q += p * (r + gamma * state_values[s])

    return Q
