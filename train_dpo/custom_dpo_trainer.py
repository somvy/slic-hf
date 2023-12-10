from trl import DPOTrainer
import torch
import torch.nn.functional as F
from typing import Tuple


class CustomDPOTrainer(DPOTrainer):
    def __init__(self, divergence_alpha: float = None, **kwargs):
        super().__init__(**kwargs)
        self.divergence_alpha = 0.5 if divergence_alpha is None else divergence_alpha

    def dpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
            reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        # maybe apply exp to logps
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        if self.loss_type == "sigmoid":  # rkl
            losses = -F.logsigmoid(self.beta * logits)

        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)

        elif self.loss_type == "alpha_divergence":
            chosen_ratios = torch.exp(policy_chosen_logps - reference_chosen_logps)
            rejected_ratios = torch.exp(policy_rejected_logps - reference_rejected_logps)
            f_grad = lambda u: (1 - u ** (-self.divergence_alpha)) / self.divergence_alpha
            losses = -F.logsigmoid(self.beta * (f_grad(chosen_ratios) - f_grad(rejected_ratios)))

        elif self.loss_type == "fkl":
            inv_chosen_ratios = torch.exp(reference_chosen_logps - policy_chosen_logps)
            inv_rejected_ratios = torch.exp(reference_rejected_logps - policy_rejected_logps)
            losses = -F.logsigmoid(self.beta * (inv_rejected_ratios - inv_chosen_ratios))

        elif self.loss_type == "jsd":
            chosen_ratios = torch.exp(policy_chosen_logps - reference_chosen_logps)
            rejected_ratios = torch.exp(policy_rejected_logps - reference_rejected_logps)
            f_grad = lambda u: torch.log((2 * u) / (1 + u))
            losses = -F.logsigmoid(self.beta * (f_grad(chosen_ratios) - f_grad(rejected_ratios)))

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'alpha_divergence','fkl', 'jsd']")

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards
