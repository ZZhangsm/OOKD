from termios import CEOL
from turtle import st
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def kd_loss(logits_student, logits_teacher, temperature, reduce=True):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature ** 2
    return loss_kd


def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss


def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class LogitKD(torch.nn.Module):
    def __init__(self):
        super(LogitKD, self).__init__()

    def forward(self, logits_student, logits_teacher, target,
                alpha=5.0, beta=5.0, temperatures=None,
                use_mixup=False, mixup_alpha=1.0):
        """
        logits_student: Logits from the student network (before softmax).
        logits_teacher: Logits from the teacher network (before softmax).
        target: Ground truth labels.
        alpha: Weight for the KD loss.
        beta: Weight for the batch and class consistency losses.
        temperatures: List of temperatures for distillation.
        use_mixup: Whether to apply mixup regularization.
        mixup_alpha: Mixup regularization hyperparameter.

        Returns the total loss combining KD, batch consistency, and class consistency losses.
        """

        if temperatures is None:
            temperatures = [2.0, 3.0, 4.0, 5.0, 6.0]
        # Apply mixup if enabled
        if use_mixup:
            logits_student, target_a, target_b, lam = mixup_data(logits_student, target, alpha=mixup_alpha)
        else:
            target_a, target_b, lam = target, target, 1.0

        # Cross-Entropy Loss (for classification)
        if use_mixup:
            ce_loss = lam * F.cross_entropy(logits_student, target_a) + (1 - lam) * F.cross_entropy(logits_student,
                                                                                                    target_b)
        else:
            ce_loss = F.cross_entropy(logits_student, target)

        # Initialize total loss
        total_loss = alpha * ce_loss

        # Loop over the temperature values
        for temp in temperatures:
            # Knowledge distillation loss (KD loss) for the current temperature
            kd_loss_value = kd_loss(logits_student, logits_teacher, temp)

            # Batch consistency loss (BC loss) for the current temperature
            bc_loss_value = bc_loss(logits_student, logits_teacher, temp)

            # Class consistency loss (CC loss) for the current temperature
            cc_loss_value = cc_loss(logits_student, logits_teacher, temp)

            # Accumulate losses for all temperatures
            total_loss += (1-alpha) * kd_loss_value + beta * (bc_loss_value + cc_loss_value)

        return total_loss
