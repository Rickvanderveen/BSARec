import torch.nn as nn
from .bsarec import BSARecModel


def compute_loss(self, interaction):
    input_ids, answers = interaction['history_ids'], interaction['item_ids']
    return self.calculate_loss(input_ids, answers, None, None, None)


def full_predict(self, user_dict, items):
    user_embeds = self.add_position_embedding(user_dict['history_ids']).mean(dim=1)
    print(user_embeds.mean(dim=1).shape)
    item_embeds = self.item_embeddings(items)
    scores = (user_embeds * item_embeds)
    scores = scores.sum(dim=-1)  # [B, H]
    return nn.Sigmoid()(scores)


def BSARec(config):
    config['no_cuda'] = config.get('no_cuda', False)
    config['gpu_id'] = config.get('gpu_id', '0')

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

    # convert the dict to an object, such that config['max_seq_length'] can be accessed as config.max_seq_length
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    config = Config(config)

    model = BSARecModel(config)
    model.type = ["sequential"]
    model.IR_type = ["retrieval", "ranking"]

    # bind the methods to the model instance
    model.compute_loss = compute_loss.__get__(model, BSARecModel)
    model.full_predict = full_predict.__get__(model, BSARecModel)

    return model
