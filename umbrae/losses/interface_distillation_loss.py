"""Paired projected-interface distillation objective for P9."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class InterfaceDistillationLoss(nn.Module):
    def __init__(self, proj_cos=1.0, proj_mse=0.01, preproj_cos=0.1, preserve=0.1, norm=0.01, eps=1e-8):
        super().__init__()
        self.weights = {
            "proj_cos": float(proj_cos), "proj_mse": float(proj_mse),
            "preproj_cos": float(preproj_cos), "preserve": float(preserve),
            "norm": float(norm),
        }
        self.eps = float(eps)

    @staticmethod
    def cosine_loss(left: Tensor, right: Tensor) -> Tensor:
        return (1.0 - F.cosine_similarity(left.float(), right.float(), dim=-1)).mean()

    def forward(self, source, adapted, visual, projected_student, projected_teacher):
        projected_student = projected_student.float()
        projected_teacher = projected_teacher.float()
        student_norm = projected_student.norm(dim=-1)
        teacher_norm = projected_teacher.norm(dim=-1)
        values = {
            "proj_cos": self.cosine_loss(projected_student, projected_teacher),
            "proj_mse": F.mse_loss(projected_student, projected_teacher),
            "preproj_cos": self.cosine_loss(adapted, visual),
            "preserve": self.cosine_loss(adapted, source),
            "norm": (((student_norm - teacher_norm).square()) / (teacher_norm.square() + self.eps)).mean(),
        }
        values["loss"] = sum(self.weights[name] * value for name, value in values.items())
        return values
