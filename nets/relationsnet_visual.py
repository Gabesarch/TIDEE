
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
from torchvision import models
import cv2
import ipdb
st = ipdb.set_trace

# import hyperparams as hyp
from arguments import args

from transformers import BertForSequenceClassification

device0 = torch.device("cuda")
device1 = torch.device("cuda")

class bertBase(nn.Module):
    def __init__(self, num_classes):
        """
        Args:
            app_emb_dim: dimension of appearance embedding
            n_head: number of attention heads
            h_hidden: dimension of the feedforward network
            n_layers: number of layers in the transformer model
            dropout: probability of dropping out
            num_classes (int): number of classes
            num_steps (int): number of times the model is unrolled
            n_heads (int): number of attention heads
        """
        super(bertBase, self).__init__()

        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = num_classes, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        ).to(device1)
        self.bert = nn.Sequential(*list(self.bert.children())[:-2]) # keep only up to pooler
        self.bert.train()
        
    def forward(self, bert_token_ids_batch, bert_attention_masks_batch, labels, summ_writer=None):

        output = self.bert(bert_token_ids_batch)
        output = output[1]

        return output

class OOPNet(nn.Module):
    def __init__(self, num_classes, weight_per_batch=True):
        """
        Args:
            app_emb_dim: dimension of appearance embedding
            n_head: number of attention heads
            h_hidden: dimension of the feedforward network
            n_layers: number of layers in the transformer model
            dropout: probability of dropping out
            num_classes (int): number of classes
            num_steps (int): number of times the model is unrolled
            n_heads (int): number of attention heads
        """
        super(OOPNet, self).__init__()
        self.num_classes = num_classes

        if args.do_visual_and_language_oop or args.do_language_only_oop:
            self.bert = bertBase(num_classes)

        if args.do_visual_and_language_oop:
            in_size = 768+args.hidden_dim # (768 bert + 256 visual)
        elif args.do_visual_only_oop:
            in_size = args.hidden_dim
        elif args.do_language_only_oop:
            in_size = 768
        hidden_size = 2048
        self.classifier = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, num_classes),
            )

        self.softmax = nn.Softmax(dim=1)

        self.weight_per_batch = weight_per_batch

    
    def forward(self, bert_token_ids_batch, bert_attention_masks_batch, visual_features, labels, summ_writer=None):
        """
        Args:
            bert_embeddings: object rgb crops resized (B, N, E)
            images: ego-centric views of the object out of place (B, 3, H, W)
        """

        if args.do_visual_and_language_oop or args.do_language_only_oop:
            bert_features = self.bert(bert_token_ids_batch, bert_attention_masks_batch, labels) 

        if args.do_visual_and_language_oop:
            bv_features = torch.cat([bert_features, visual_features], dim=1)
        elif args.do_visual_only_oop:
            bv_features = visual_features
        elif args.do_language_only_oop:
            bv_features = bert_features
        
        logits = self.classifier(bv_features)
        
        do_sigmoid_focal_loss = False
        if labels is not None:         
            if do_sigmoid_focal_loss: # sigmoid focal loss
                # create one-hot for focal loss
                targets = torch.zeros((logits.shape[0], logits.shape[1])).cuda()
                for j in range(targets.shape[0]):
                    targets[j,labels[j]] = 1.
                loss = sigmoid_focal_loss(logits, targets)
                probs = logits.detach().sigmoid()
            else:
                # loss = self.ce(logits, labels)
                # compute class weights per batch
                if self.weight_per_batch:
                    ip_percent = torch.sum(labels==0)/len(labels) + 1e-6
                    oop_percent = torch.sum(labels==1)/len(labels) + 1e-6
                    class_weights = 1.0 / torch.tensor ([oop_percent, ip_percent]).cuda()
                    class_weights = torch.clip(class_weights, min=0.1, max=5.)
                    class_weights = class_weights / torch.max(class_weights)
                else:
                    class_weights = None
                # loss = self.ce(logits, labels, weight=class_weights)
                loss = F.cross_entropy(logits, labels, weight=class_weights)
                probs = self.softmax(logits.detach())
        else:
            if do_sigmoid_focal_loss:
                probs = logits.detach().sigmoid()
            else:
                probs = self.softmax(logits.detach())

            loss = None

        return loss, probs


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.sum() # loss.mean(1).sum() / num_boxes