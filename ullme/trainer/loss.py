import einops
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F 
from pytorch_metric_learning import losses, miners, distances


class AllGather(torch.autograd.Function):
    """
    all_gather with gradient back-propagation
    """

    @staticmethod
    def forward(ctx, tensor_list, tensor, group, async_op):
        torch.distributed.all_gather(
            tensor_list, tensor, group=group, async_op=async_op
        )
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = torch.distributed.get_rank()

        dist_ops = [
            torch.distributed.reduce(grad_list[i], i, async_op=True)
            for i in range(torch.distributed.get_world_size())
        ]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank], None, None


all_gather_with_grad = AllGather.apply

def mismatched_sizes_all_gather(
    tensor: torch.Tensor, group=None, async_op=False, mismatched_axis=0
):
    # all_gather doesn't support tensor lists where the first dimension is mismatched. This does.
    assert torch.distributed.is_initialized(), "torch.distributed not initialized"
    world_size = torch.distributed.get_world_size()
    # let's get the sizes for everyone
    mismatched_sizes = torch.tensor(
        [tensor.shape[mismatched_axis]], dtype=torch.int64, device="cuda"
    )
    sizes = [torch.zeros_like(mismatched_sizes) for _ in range(world_size)]
    torch.distributed.all_gather(
        sizes, mismatched_sizes, group=group, async_op=async_op
    )
    sizes = torch.cat(sizes).cpu().tolist()
    # now pad to the max dim-0 size
    max_size = max(sizes)
    padded = torch.zeros(
        (
            *tensor.shape[:mismatched_axis],
            max_size,
            *tensor.shape[mismatched_axis + 1 :],
        ),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    # selects the place where we're adding information
    padded_to_fill = padded.narrow(mismatched_axis, 0, tensor.shape[mismatched_axis])
    padded_to_fill[...] = tensor
    # gather the padded tensors
    tensor_list = [
        torch.zeros(padded.shape, device=padded.device, dtype=padded.dtype)
        for _ in range(world_size)
    ]
    all_gather_with_grad(tensor_list, padded, group, async_op)
    # trim off the padding
    for rank in range(world_size):
        # checks that the rest is 0
        assert (
            not tensor_list[rank]
            .narrow(
                mismatched_axis,
                sizes[rank],
                padded.shape[mismatched_axis] - sizes[rank],
            )
            .count_nonzero()
            .is_nonzero()
        ), "This would remove non-padding information"
        tensor_list[rank] = tensor_list[rank].narrow(mismatched_axis, 0, sizes[rank])
    return tensor_list


class ContrastiveLoss:
    def __init__(
            self,
            loss_type: str = 'NTXentLoss', # 'NTXentLoss' or 'SupConLoss'
            temperature: float = 0.05,
            is_distance: bool = True,
            use_miner: bool = False,
            ) -> None:
        self.temperature = temperature if temperature > 0 else 1.0
        distance = distances.LpDistance(normalize_embeddings=True) if is_distance else distances.CosineSimilarity()
        if loss_type == 'NTXentLoss':
            self.loss_fn = losses.NTXentLoss(temperature=temperature, distance=distance)
        elif loss_type == 'SupConLoss':
            self.loss_fn = losses.SupConLoss(temperature=temperature, base_distance=distance)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        if use_miner:
            self.miner = miners.PairMarginMiner(
                pos_margin = 0.0 if is_distance else 1.0, # under 1.0 for cosine similarity or over 0.0 for LpDistance will be considered as hard positive 
                neg_margin = 0.5, # over 0.5 for cosine similarity or under 0.5 for LpDistance will be considered as hard negative.
                distance=distance
                )
        else:
            self.miner = None

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(
            self, 
            q_embeds: torch.Tensor, # (batch_size, embed_dim) 
            q_labels: torch.Tensor, # (batch_size,)
            pos_embeds: torch.Tensor, # (batch_size, num_pos, embed_dim)
            neg_embeds: torch.Tensor, # (batch_size, num_neg, embed_dim)
            cross_batch_loss: bool = True,
            ):
        
        cross_batch_loss =  cross_batch_loss and torch.distributed.is_initialized()
        cross_batch_loss = torch.tensor(cross_batch_loss, device=q_embeds.device)
        if cross_batch_loss:
            # gather all cross_batch_loss flags from all processes
            all_cross_batch_loss = [torch.zeros_like(cross_batch_loss) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(all_cross_batch_loss, cross_batch_loss)
            # if all the processes have cross_batch_loss=True, then the cross_batch_loss=True
            cross_batch_loss = all([x.item() for x in all_cross_batch_loss])

        pos_labels = einops.repeat(q_labels, 'b -> b n', n=pos_embeds.size(1)) # (batch_size, num_pos)
        
        if cross_batch_loss:
            pos_labels = einops.rearrange(pos_labels, 'b n -> (b n)')
            pos_embeds = einops.rearrange(pos_embeds, 'b n d -> (b n) d') # (batch_size * num_pos, embed_dim)
            neg_embeds = einops.rearrange(neg_embeds, 'b n d -> (b n) d') # (batch_size * num_neg, embed_dim)

            full_q_labels = mismatched_sizes_all_gather(q_labels)
            full_pos_labels = mismatched_sizes_all_gather(pos_labels)
            full_q_embeds = mismatched_sizes_all_gather(q_embeds) 
            full_pos_embeds = mismatched_sizes_all_gather(pos_embeds)
            full_neg_embeds = mismatched_sizes_all_gather(neg_embeds)

            full_q_labels = torch.cat(full_q_labels, dim=0) # (world_size * batch_size,)
            full_pos_labels = torch.cat(full_pos_labels, dim=0) # (num_all_pos,)
            full_q_embeds = torch.cat(full_q_embeds, dim=0) # (world_size * batch_size, embed_dim)
            full_pos_embeds = torch.cat(full_pos_embeds, dim=0) # (num_all_pos, embed_dim)
            full_neg_embeds = torch.cat(full_neg_embeds, dim=0) # (num_all_neg, embed_dim)
            max_idx = torch.max(full_q_labels)
            full_neg_labels = torch.arange(full_neg_embeds.size(0), device=full_neg_embeds.device) + max_idx + 1 # (num_all_neg,)
            
            full_labels = torch.cat([full_q_labels, full_pos_labels, full_neg_labels], dim=0)
            full_embeds = torch.cat([full_q_embeds, full_pos_embeds, full_neg_embeds], dim=0)

            if self.miner is not None:
                hard_pairs = self.miner(full_embeds, full_labels)
                loss = self.loss_fn(full_embeds, full_labels, hard_pairs)
            else:
                loss = self.loss_fn(full_embeds, full_labels)
        else:
            max_idx = torch.max(q_labels)
            neg_labels = torch.arange(neg_embeds.size(0)*neg_embeds.size(1), device=neg_embeds.device) + max_idx + 1 # (num_neg)
            neg_labels = einops.rearrange(neg_labels, '(b n) -> b n', b=neg_embeds.size(0), n=neg_embeds.size(1)) # (batch_size, num_neg)
            
            candidate_labels = torch.cat([pos_labels, neg_labels], dim=1) # (batch_size, num_pos + num_neg)
            candidate_embeds = torch.cat([pos_embeds, neg_embeds], dim=1) # (batch_size, num_pos + num_neg, embed_dim)
            
            # labels[i][j] = 1  if candidate_labels[i][j] == q_labels[i] else 0
            labels = torch.eq(candidate_labels, einops.repeat(q_labels, 'b -> b n', n=candidate_labels.size(1))) # (batch_size, num_pos + num_neg)
            labels = torch.softmax(labels.float(), dim=-1) # (batch_size, num_pos + num_neg)
            # get scores where scores[i][j] = cosine_similarity(q_embeds[i], candidate_embs[i][j])
            scores = F.cosine_similarity(q_embeds.unsqueeze(1), candidate_embeds, dim=-1) / self.temperature # (batch_size, num_pos + num_neg)
            loss = self.cross_entropy_loss(scores, labels)

        return loss


class PreferenceLoss:
    def __init__(
            self,
            loss_type: str = "sigmoid",
            reference_free: bool = False,
            label_smoothing: float = 0.0,
            beta: float = 0.1,
            ) -> None:
        self.loss_type = loss_type
        self.reference_free = reference_free
        self.label_smoothing = label_smoothing
        self.beta = beta

    def __call__(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor = None,
            reference_rejected_logps: torch.FloatTensor = None,
            ) -> torch.FloatTensor:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            loss_type: The type of DPO loss to compute. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair'].
            reference_free: Whether to ignore the reference model. If True, the reference model log probabilities are set to 0, i.e., cpo loss.
            label_smoothing: The label smoothing parameter. Should be in the range [0, 1].
            beta: The temperature parameter. Should be a positive scalar.

        Returns:
            The perference loss for each example in the batch. Shape: (batch_size,)
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.reference_free or reference_chosen_logps is None or reference_rejected_logps is None:
            ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )
        return losses # (batch_size,)


class KLLoss:
    def __init__(
            self,
            temperature: float = 0.1,
            ) -> None:
        self.temperature = temperature
        self.loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
    
    def __call__(
            self,
            dual_scores: torch.FloatTensor, # (batch_size, num_candidates)
            gen_scores: torch.FloatTensor, # (batch_size, num_candidates)
            ) -> torch.FloatTensor:
        dual_logps = torch.log_softmax(dual_scores/self.temperature, dim=-1)
        gen_logps = torch.softmax(gen_scores, dim=-1).detach()
        loss = self.loss_fn(dual_logps, gen_logps)
        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=dual_scores.device)
        return loss


