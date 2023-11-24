# This code is modified from https://github.com/jakesnell/prototypical-networks and https://github.com/kjunelee/MetaOptNet

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from qpth.qp import QPFunction
import torch.nn.functional as F
from functools import partial

from methods.meta_template import MetaTemplate


class MetaOptNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support, base_learner="svm", C_reg=None, lambda_reg=None, maxIter=15):
        super(MetaOptNet, self).__init__(backbone, n_way, n_support)
        if base_learner == "svm":
            assert (C_reg > 0.0)
            self.base_learner = partial(svm, device=self.device, C_reg=C_reg, maxIter=maxIter)
        elif base_learner == "ridge":
            assert (lambda_reg > 0.0)
            self.base_learner = partial(ridge_regression, device=self.device, lambda_reg=lambda_reg)
        else:
            raise ValueError("Please specify a correct base learner.")
        self.loss_fn = nn.CrossEntropyLoss()
        
    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        n_support = self.n_way * self.n_support
        n_query = self.n_way * self.n_query

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.reshape(1, n_support)).to(self.device)

        z_query = z_query.reshape(1, n_query, -1)
        z_support = z_support.reshape(1, n_support, -1)
        logit_query = self.base_learner(z_query, z_support, y_support, self.n_way, self.n_support).squeeze(0)
        return logit_query

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.reshape(self.n_query * self.n_way)).to(self.device)
        
        logit_query = self.set_forward(x)
        loss = self.loss_fn(logit_query.reshape(-1, self.n_way), y_query.reshape(-1))
        return loss
    
    
def ridge_regression(query, support, support_labels, n_way, n_shot, device, lambda_reg, double_precision=False):
    """
    Fits the support set with ridge regression and 
    returns the classification score on the query set.

    Parameters:
    query:  a (tasks_per_batch, n_query, d) Tensor.
    support:  a (tasks_per_batch, n_support, d) Tensor.
    support_labels: a (tasks_per_batch, n_support) Tensor.
    n_way: a scalar. Represents the number of classes in a few-shot classification task.
    n_shot: a scalar. Represents the number of support examples given per class.
    lambda_reg: a scalar. Represents the strength of L2 regularization.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot

    #Here we solve the dual problem:
    #Note that the classes are indexed by m & samples are indexed by i.
    #min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i

    #where w_m(\alpha) = \sum_i \alpha^m_i x_i,
    
    #\alpha is an (n_support, n_way) matrix
    kernel_matrix = compute_gram_matrix(support, support)
    kernel_matrix += lambda_reg * torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).to(device)

    block_kernel_matrix = kernel_matrix.repeat(n_way, 1, 1) #(n_way * tasks_per_batch, n_support, n_support)
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way, device) # (tasks_per_batch * n_support, n_way)
    support_labels_one_hot = support_labels_one_hot.transpose(0, 1) # (n_way, tasks_per_batch * n_support)
    support_labels_one_hot = support_labels_one_hot.reshape(n_way * tasks_per_batch, n_support)     # (n_way*tasks_per_batch, n_support)
    
    G = block_kernel_matrix
    e = -2.0 * support_labels_one_hot
    
    #This is a fake inequlity constraint as qpth does not support QP without an inequality constraint.
    id_matrix_1 = torch.zeros(tasks_per_batch*n_way, n_support, n_support)
    C = Variable(id_matrix_1)
    h = Variable(torch.zeros((tasks_per_batch*n_way, n_support)))
    dummy = Variable(torch.Tensor()).to(device)   # We want to ignore the equality constraint.

    if double_precision:
        G, e, C, h = [x.double().to(device) for x in [G, e, C, h]]

    else:
        G, e, C, h = [x.float().to(device) for x in [G, e, C, h]]

    # Solve the following QP to fit SVM:
    #        \hat z =   argmin_z 1/2 z^T G z + e^T z
    #                 subject to Cz <= h
    # We use detach() to prevent backpropagation to fixed variables.
    qp_sol = QPFunction(verbose=False)(G, e.detach(), C.detach(), h.detach(), dummy.detach(), dummy.detach())
    #qp_sol = QPFunction(verbose=False)(G, e.detach(), dummy.detach(), dummy.detach(), dummy.detach(), dummy.detach())

    #qp_sol (n_way*tasks_per_batch, n_support)
    qp_sol = qp_sol.reshape(n_way, tasks_per_batch, n_support)
    #qp_sol (n_way, tasks_per_batch, n_support)
    qp_sol = qp_sol.permute(1, 2, 0)
    #qp_sol (tasks_per_batch, n_support, n_way)
    
    # Compute the classification score.
    compatibility = compute_gram_matrix(support, query)
    compatibility = compatibility.float()
    compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
    qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
    logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
    logits = logits * compatibility
    logits = torch.sum(logits, 1)

    return logits


def svm(query, support, support_labels, n_way, n_shot, device, C_reg, double_precision=False, maxIter=15):
    """
    Fits the support set with multi-class SVM and 
    returns the classification score on the query set.
    
    This is the multi-class SVM presented in:
    On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines
    (Crammer and Singer, Journal of Machine Learning Research 2001).

    This model is the classification head that we use for the final version.
    Parameters:
    query:  a (tasks_per_batch, n_query, d) Tensor.
    support:  a (tasks_per_batch, n_support, d) Tensor.
    support_labels: a (tasks_per_batch, n_support) Tensor.
    n_way: a scalar. Represents the number of classes in a few-shot classification task.
    n_shot: a scalar. Represents the number of support examples given per class.
    C_reg: a scalar. Represents the cost parameter C in SVM.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot

    #Here we solve the dual problem:
    #Note that the classes are indexed by m & samples are indexed by i.
    #min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
    #s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i

    #where w_m(\alpha) = \sum_i \alpha^m_i x_i,
    #and C^m_i = C if m  = y_i,
    #C^m_i = 0 if m != y_i.
    #This borrows the notation of liblinear.
    
    #\alpha is an (n_support, n_way) matrix
    kernel_matrix = compute_gram_matrix(support, support)

    id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).to(device)
    block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)
    #This seems to help avoid PSD error from the QP solver.
    block_kernel_matrix += 1.0 * torch.eye(n_way*n_support).expand(tasks_per_batch, n_way*n_support, n_way*n_support).to(device)
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way, device) # (tasks_per_batch * n_support, n_support)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, n_support * n_way)
    
    G = block_kernel_matrix
    e = -1.0 * support_labels_one_hot
    #print (G.size())
    #This part is for the inequality constraints:
    #\alpha^m_i <= C^m_i \forall m,i
    #where C^m_i = C if m  = y_i,
    #C^m_i = 0 if m != y_i.
    id_matrix_1 = torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support, n_way * n_support)
    C = Variable(id_matrix_1)
    h = Variable(C_reg * support_labels_one_hot)
    #print (C.size(), h.size())
    #This part is for the equality constraints:
    #\sum_m \alpha^m_i=0 \forall i
    id_matrix_2 = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).to(device)

    A = Variable(batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, n_way).to(device)))
    b = Variable(torch.zeros(tasks_per_batch, n_support))
    #print (A.size(), b.size())
    if double_precision:
        G, e, C, h, A, b = [x.double().to(device) for x in [G, e, C, h, A, b]]
    else:
        G, e, C, h, A, b = [x.float().to(device) for x in [G, e, C, h, A, b]]

    # Solve the following QP to fit SVM:
    #        \hat z =   argmin_z 1/2 z^T G z + e^T z
    #                 subject to Cz <= h
    # We use detach() to prevent backpropagation to fixed variables.
    qp_sol = QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())

    # Compute the classification score.
    compatibility = compute_gram_matrix(support, query)
    compatibility = compatibility.float()
    compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
    qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
    logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
    logits = logits * compatibility
    logits = torch.sum(logits, 1)

    return logits
    

def compute_gram_matrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.
    
    Parameters:
    A:  a (n_batch, n, d) Tensor.
    B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """
    
    assert(A.dim() == 3)
    assert(B.dim() == 3)
    assert(A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1,2))


def one_hot(indices, depth, device):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
    indices:  a (n_batch, m) Tensor or (m) Tensor.
    depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).to(device)
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies


def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape([matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))
