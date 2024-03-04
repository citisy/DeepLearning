import torch


def enum_search(x):
    pass


def greedy_search(x):
    """

    Args:
        x (torch.Tensor): (batch_size, seq_length, vocab_size) after log softmax

    Returns:
        preds: (batch_size, beam_size, seq_length)
        probs: (batch_size, beam_size)

    """
    probs, preds = x.max(2)
    return preds, probs


def beam_search(x, beam_size=4):
    """

    Args:
        x (torch.Tensor): (batch_size, seq_length, vocab_size) after log softmax
        beam_size:

    Returns:
        preds: (batch_size, beam_size, seq_length)
        probs: (batch_size, beam_size)

    """
    batch, seq_len, vocab_size = x.shape
    probs, pred = x[:, 0, :].topk(beam_size, sorted=True)
    preds = pred.unsqueeze(-1)
    for i in range(1, seq_len):
        probs = probs.unsqueeze(-1) + x[:, i, :].unsqueeze(1).repeat(1, beam_size, 1)
        probs, pred = probs.view(batch, -1).topk(beam_size, sorted=True)
        idx = torch.div(pred, vocab_size, rounding_mode='trunc')
        pred = pred % vocab_size
        preds = torch.gather(preds, 1, idx.unsqueeze(-1).repeat(1, 1, i))
        preds = torch.cat([preds, pred.unsqueeze(-1)], dim=-1)
    return preds, probs


def prefix_beam_search(x):
    pass


def bpe(tags, word_inv_dict, byte_decoder_dict=None):
    segments = []
    for tag in tags:
        text = ''.join([word_inv_dict[t] for t in tag])
        s = bytearray([byte_decoder_dict[byte] for byte in text]).decode('utf-8', errors='replace')
        segments.append(s)
    return segments
