from .bsarec import BSARecModel


def BSARec(config):
    config['max_seq_length'] = config.get('max_seq_length', 50)
    config['hidden_size'] = config.get('hidden_size', 64)
    config['num_hidden_layers'] = config.get('num_hidden_layers', 2)
    config['hidden_act'] = config.get('hidden_act', 'gelu')
    config['num_attention_heads'] = config.get('num_attention_heads', 2)
    config['attention_probs_dropout_prob'] = config.get('attention_probs_dropout_prob', 0.5)
    config['hidden_dropout_prob'] = config.get('hidden_dropout_prob', 0.5)
    config['initializer_range'] = config.get('initializer_range', 0.02)

    config['c'] = config.get('c', 3)
    config['alpha'] = config.get('alpha', 0.9)

    return BSARecModel(config)
