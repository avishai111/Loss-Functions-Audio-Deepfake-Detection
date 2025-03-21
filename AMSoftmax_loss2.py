import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSoftmaxLoss(nn.Module):

    def __init__(self, embedding_dim, no_classes, scale = 30.0, margin=0.4):
        '''
        AM Softmax Loss


        Attributes
        ----------
        embedding_dim : int 
            Dimension of the embedding vector
        no_classes : int
            Number of classes to be embedded
        scale : float
            Global scale factor
        margin : float
            Size of additive margin        
        '''
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.no_classes = no_classes
        self.embedding = nn.Embedding(no_classes, embedding_dim, max_norm=1) # max_norm=1 to normalize the embedding vectors 
        self.loss = nn.CrossEntropyLoss() # CrossEntropyLoss() combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

    def forward(self, x, labels):
        '''
        Input shape (N, embedding_dim)
        '''
        seed = 42
        torch.manual_seed(seed) 
        
        n, m = x.shape        
        assert n == len(labels)
        assert m == self.embedding_dim
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.no_classes

        x = F.normalize(x, dim=1) # normalize the input embedding vectors
        w = self.embedding.weight         
        cos_theta = torch.matmul(w, x.T).T 
        psi = cos_theta - self.margin 
        logits = cos_theta
        onehot = F.one_hot(labels, self.no_classes) # one-hot encode the labels
        margin_logits  = self.scale * torch.where(onehot == 1, psi, cos_theta) # scale logits by a factor of s=30.0 as mentioned in the paper    
        err = self.loss(margin_logits , labels) # compute the loss using cross entropy loss function
        
        return err, margin_logits,logits 
