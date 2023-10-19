
def preserve_key(state, preserve_prefix: str):
    """Preserve part of model weights based on the
       prefix of the preserved module name.
    """
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if preserve_prefix + "." in key:
            newkey = key.replace(preserve_prefix + '.', "")
            state[newkey] = state.pop(key)
        else:
            state.pop(key)
    return state

def wise_merge(zeroclip, ftclip, alpha=0.5):
    # merge the same part of two models and return the merged model
    theta_0 = {k: v for k, v in zeroclip.state_dict().items()}
    theta_1 = {k: v for k, v in ftclip.state_dict().items() if k in theta_0.keys()}

    theta = {key: (1 - alpha) * theta_0[key] + alpha * theta_1[key] for key in theta_0.keys()}


    zeroclip.load_state_dict(theta)

    return zeroclip