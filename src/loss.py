import torch

cross_entropy = torch.nn.CrossEntropyLoss()


def contrastive_loss_entropy(y, ids, temperature):
    y = torch.nn.functional.normalize(y)
    cosine_sim = torch.matmul(y, torch.transpose(y, 1, 0))

    N = len(ids)

    y_true = ids.reshape((-1, 1))

    same_subject = (y_true == y_true.transpose(1, 0)) * 1

    positive_mask = (same_subject - torch.eye(N)).cuda()
    positive_similarity = cosine_sim[positive_mask == 1].reshape(N, 1)

    negative_mask = 1 - same_subject

    negative_similarity = cosine_sim[negative_mask == 1]
    negative_similarity = negative_similarity.reshape(N, -1)

    logits = torch.concat([positive_similarity, negative_similarity], dim=1)
    labels = torch.zeros(N).long().cuda()
    loss_val = cross_entropy(logits / temperature, labels)
    return loss_val
