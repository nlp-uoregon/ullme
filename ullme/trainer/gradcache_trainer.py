import shutil
from collections import UserDict
import pathlib
from contextlib import nullcontext
import time
from typing import Any, Callable, Dict, List, Optional
import typing
import einops
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.checkpoint import get_device_states, set_device_states
import lightning as L

from ullme.model.ullme import ULLME, WrappedULLME
from ullme.trainer.loss import ContrastiveLoss, KLLoss, PreferenceLoss
from ullme.trainer.utils import clear_unused_gpu_mem, get_batch_logps, split_input
from ullme.eval.eval import eval_multilingual, eval_mteb


class RandContext:
    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


class GradCacheTrainer:
    def __init__(
            self,
            fabric: L.Fabric,
            use_gen: bool = False,
            use_kl: bool = False,
            gen_loss_type: str = 'sft',
            beta: float = 0.1,
            reference_free: bool = True,
            label_smoothing: float = 0.1,
            loss_type: str = 'NTXentLoss',
            temperature: float = 0.05,
            is_distance: bool = True,
            use_miner: bool = False,
            chunk_size: Optional[int] = 1,
            ) -> None:
        
        self.fabric = fabric
        self.chunk_size = chunk_size
        self.use_kl = use_kl
        self.use_gen = use_gen

        self.loss_fn = ContrastiveLoss(
            loss_type=loss_type,
            temperature=temperature,
            is_distance=is_distance,
            use_miner=use_miner,
        )
        
        if use_kl:
            assert use_gen and gen_loss_type != 'sft', 'KL loss requires DPO loss'
            self.kl_loss = KLLoss(temperature=temperature)
        
        if use_gen:
            self.gen_loss_type = gen_loss_type
            if gen_loss_type == 'sft':
                self.gen_loss = None
            else:
                self.gen_loss = PreferenceLoss(
                    loss_type=gen_loss_type,
                    reference_free=reference_free,
                    label_smoothing=label_smoothing,
                    beta=beta
                )
        
        self.best_en = 0.0
        self.best_multi = 0.0
        

    def get_input_tensors(self, model_input) -> List[torch.Tensor]:
        if isinstance(model_input, torch.Tensor):
            return [model_input]
        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])
        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])
        else:
            return []
    
    def forward_no_grad(
            self, 
            model: ULLME, 
            model_inputs: Dict[str, torch.Tensor],
            ):
        with torch.no_grad():
            rnd_state = RandContext(*self.get_input_tensors(model_inputs))

            P = model_inputs['min_pos_per_sample']
            B = model_inputs['cons_input_ids'].size(0)
            in_ids = model_inputs['cons_input_ids'] # (batch_size, 1 + #p + #n, in_seq_len)
            in_attention_mask = model_inputs['cons_attention_mask']

            input_ids = einops.rearrange(in_ids, 'b n s -> (b n) s')
            attention_mask = einops.rearrange(in_attention_mask, 'b n s -> (b n) s')

            # forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                is_encode=True,
            )
            projected_reps = outputs['projected_reps'] # (batch_size * (1 + #p + #n), embed_dim)
            projected_reps = einops.rearrange(projected_reps, '(b n) d -> b n d', b=B)
            projected_query = projected_reps[:, 0]
            projected_pos = projected_reps[:, 1:1+P]
            projected_neg = projected_reps[:, 1+P:]
        
        return projected_query, projected_pos, projected_neg, rnd_state

    def compute_cons_loss_from_reps(
            self,
            query_projections: torch.Tensor, # (batch_size, embed_dim)
            pos_projections: torch.Tensor, # (batch_size, num_pos, embed_dim)
            neg_projections: torch.Tensor, # (batch_size, num_neg, embed_dim)
            query_labels: torch.Tensor, # (batch_size,)
            cross_batch_loss: bool = True,
            ) -> torch.Tensor:
        con_loss = self.loss_fn(
            q_embeds=query_projections,
            q_labels=query_labels,
            pos_embeds=pos_projections,
            neg_embeds=neg_projections,
            cross_batch_loss=cross_batch_loss,
        )
        return con_loss
    
    @typing.no_type_check
    def build_cache(
            self,
            query_projections: torch.Tensor, # (batch_size, embed_dim)
            pos_projections: torch.Tensor, # (batch_size, num_pos, embed_dim)
            neg_projections: torch.Tensor, # (batch_size, num_neg, embed_dim)
            query_labels: torch.Tensor, # (batch_size,)
            cross_batch_loss: bool = True,
            ):
        B, P, _ = pos_projections.size()
        B, N, _ = neg_projections.size()
        projections = torch.cat([
            query_projections.unsqueeze(1), # (batch_size, 1, embed_dim)
            pos_projections, # (batch_size, num_pos, embed_dim)
            neg_projections, # (batch_size, num_neg, embed_dim)
        ], dim=1) # (batch_size, 1 + num_pos + num_neg, embed_dim)
        projections = projections.detach().requires_grad_(True)

        query_projections = projections[:, 0] # (batch_size, embed_dim)
        pos_projections = projections[:, 1:1+P] # (batch_size, num_pos, embed_dim)
        neg_projections = projections[:, 1+P:] # (batch_size, num_neg, embed_dim)

        with nullcontext():
            with self.fabric.autocast():
                con_loss = self.compute_cons_loss_from_reps(
                    query_projections=query_projections,
                    pos_projections=pos_projections,
                    neg_projections=neg_projections,
                    query_labels=query_labels,
                    cross_batch_loss=cross_batch_loss,
                )
        self.fabric.backward(con_loss)
        cache = projections.grad # (batch_size, 1 + num_pos + num_neg, embed_dim)
        
        if cross_batch_loss:
            con_loss = con_loss.detach() / self.fabric.world_size
        else:
            con_loss = con_loss.detach()

        return cache, con_loss

    def forward_backward(
            self,
            model: ULLME,
            model_inputs: Dict[str, torch.Tensor],
            state: RandContext, 
            cache: torch.Tensor, # (batch_size, 1 + num_pos + num_neg, embed_dim)
            ):
        with state:
            P = model_inputs['min_pos_per_sample']
            B = model_inputs['cons_input_ids'].size(0)
            cons_input_ids = model_inputs['cons_input_ids'] # (batch_size, 1 + #p + #n, in_seq_len)
            cons_attention_mask = model_inputs['cons_attention_mask']
            gen_input_ids = model_inputs['gen_input_ids'] # (batch_size, #p + #n, in_seq_len)
            gen_attention_mask = model_inputs['gen_attention_mask']
            gen_labels = model_inputs['gen_labels']

            # forward pass
            cons_input_ids = einops.rearrange(cons_input_ids, 'b n s -> (b n) s')
            cons_attention_mask = einops.rearrange(cons_attention_mask, 'b n s -> (b n) s')
            encoder_outputs = model(
                input_ids=cons_input_ids,
                attention_mask=cons_attention_mask,
                is_encode=True,
            )
            projections = encoder_outputs['projected_reps'] # (batch_size * (1 + #p + #n), embed_dim)
            cache = einops.rearrange(cache, 'b n d -> (b n) d', b=B)
            surrougate = torch.dot(projections.flatten(), cache.flatten())

            if self.use_gen:
                if self.gen_loss is None:
                    input_ids = gen_input_ids[:, :P]
                    attention_mask = gen_attention_mask[:, :P]
                    labels = gen_labels[:, :P]
                else:
                    input_ids = gen_input_ids
                    attention_mask = gen_attention_mask
                    labels = gen_labels
                    
                input_ids = einops.rearrange(input_ids, 'b n s -> (b n) s')
                attention_mask = einops.rearrange(attention_mask, 'b n s -> (b n) s')
                labels = einops.rearrange(labels, 'b n s -> (b n) s')
                gen_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels if self.gen_loss is None else None,
                    is_encode=False,
                )
                if self.gen_loss is None:
                    gen_loss = gen_outputs['loss']
                else:
                    gen_logits = gen_outputs['logits']
                    gen_logps, token_mean_logps = get_batch_logps(
                        logits=gen_logits,
                        labels=labels,
                        average_log_prob=self.gen_loss_type == "ipo",
                        label_pad_token_id=-100,
                    ) # (batch_size * num_choice,)
                    gen_logps = einops.rearrange(gen_logps, '(b n) -> b n', b=B)
                    token_mean_logps = einops.rearrange(token_mean_logps, '(b n) -> b n', b=B)
                    policy_choice_logps = gen_logps[:, 0]
                    policy_reject_logps = gen_logps[:, P]
                    gen_loss = self.gen_loss(
                        policy_chosen_logps=policy_choice_logps,
                        policy_rejected_logps=policy_reject_logps,
                        reference_chosen_logps=None,
                        reference_rejected_logps=None,
                    )
                    gen_loss = gen_loss.mean()
            else:
                gen_loss = None
            
            if self.use_kl:
                projections = einops.rearrange(projections, '(b n) d -> b n d', b=B)
                query_reps = projections[:, 0] # (batch_size, embed_dim)
                passage_reps = projections[:, 1:] # (batch_size, num_pos + num_neg, embed_dim)
                dual_scores = torch.cosine_similarity(query_reps.unsqueeze(1), passage_reps, dim=-1) # (batch_size, num_pos + num_neg)
                kl_loss = self.kl_loss(dual_scores=dual_scores, gen_scores=token_mean_logps)
            else:
                kl_loss = None

        loss = surrougate
        if gen_loss is not None:
            loss = loss + gen_loss
        else:
            gen_loss = torch.tensor(0.0)
        if kl_loss is not None:
            loss = loss + kl_loss
        else:
            kl_loss = torch.tensor(0.0)
        
        self.fabric.backward(loss)
        return kl_loss.detach(), gen_loss.detach()
    
    def train_step(
            self,
            model: ULLME,
            batch: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
        
        # Split input into chunks
        assert 'min_pos_per_sample' in batch, 'min_pos_per_sample is required for negative sampling'
        P = batch.pop('min_pos_per_sample')
        enable_cross_batch_negative_sampling = batch.pop('enable_cross_batch_negative_sampling', True)
        splitted_inputs = split_input(batch, self.chunk_size)

        # Forward pass for each chunk
        rnd_states = []
        all_query_projections = []
        all_pos_projections = []
        all_neg_projections = []
        for chunk in splitted_inputs:
            chunk['min_pos_per_sample'] = P
            query_projections, pos_projections, neg_projections, rnd_state = self.forward_no_grad(model, chunk)
            all_query_projections.append(query_projections)
            all_pos_projections.append(pos_projections)
            all_neg_projections.append(neg_projections)
            rnd_states.append(rnd_state)
        all_query_projections = torch.cat(all_query_projections, dim=0)
        all_pos_projections = torch.cat(all_pos_projections, dim=0)
        all_neg_projections = torch.cat(all_neg_projections, dim=0)

        # Build cache for representations from all chunks
        labels = batch['query_labels']
        cache, con_loss = self.build_cache(
            query_projections=all_query_projections,
            pos_projections=all_pos_projections,
            neg_projections=all_neg_projections,
            query_labels=labels,
            cross_batch_loss=enable_cross_batch_negative_sampling,
        )
        cache = cache.split(self.chunk_size, dim=0)

        # Forward and backward pass for each chunk
        accumulated_flags = [True for _ in range(len(splitted_inputs)-1)] + [False]
        all_gen_loss = []
        all_kl_loss = []
        for chunk, c, state, flag in zip(splitted_inputs, cache, rnd_states, accumulated_flags):
            chunk['min_pos_per_sample'] = P
            with self.fabric.no_backward_sync(model, enabled=flag):
                kl_loss, gen_loss = self.forward_backward(
                    model=model,
                    model_inputs=chunk,
                    state=state,
                    cache=c,
                )
            all_gen_loss.append(gen_loss)
            all_kl_loss.append(kl_loss)
        gen_loss =  torch.mean(torch.stack(all_gen_loss))
        kl_loss = torch.mean(torch.stack(all_kl_loss))
        return con_loss, gen_loss, kl_loss

    def fit_epoch(
            self,
            model: ULLME,
            train_loader: DataLoader,
            state: Dict[str, Any],
            lr_max_steps: int = 1000,
            grad_norm_clip: float = None,
            log_interval: int = 1,
            checkpoint_iterval: Optional[int] = 10000,
            checkpoint_dir: Optional[str] = './checkpoints/',
            checkpoint_filter: Optional[Callable] = None,
            model_revision: Optional[str] = 'v0.1',
            eval_batch_size: Optional[int] = 32,
            ):
        """
        Fit epoch for gradient cache training.
        """
        optimizer: torch.optim.Optimizer = state["optimizer"]
        scheduler : torch.optim.lr_scheduler.LambdaLR = state.get("scheduler", None)
        current_step = state.get("current_step", 0) # checkpoint iteration number
        epoch_num = state.get("epoch_num", 0) # checkpoint epoch number
        self.fabric.print(f"Starting epoch {epoch_num} with {len(train_loader)} iterations")
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            if current_step > len(train_loader)*epoch_num + batch_idx:
                continue
            if current_step > lr_max_steps:
                break

            current_step = current_step + 1
            if current_step == 1:
                size_info = {k: v.size() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                self.fabric.print("First batch data: {}".format(size_info))

            iter_t0 = time.perf_counter()  
            con_loss, gen_loss, kl_loss = self.train_step(model=model, batch=batch)

            if grad_norm_clip is not None:
                self.fabric.clip_gradients(model, optimizer, max_norm=grad_norm_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            # Log metrics
            if current_step % log_interval == 0:
                t1 = time.perf_counter()

                metrics = {
                    'con_loss': con_loss.item(),
                    'gen_loss': gen_loss.item(),
                    'kl_loss': kl_loss.item(),
                    'iter_time': t1 - iter_t0,
                    'epoch': epoch_num,
                    # 'iter_num': batch_idx,
                    'lr': scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr'],
                }
                self.fabric.log_dict(metrics, step=current_step)
                self.fabric.print(
                    f"Epoch {epoch_num} | Iter {batch_idx} |"
                    f" ConLoss: {metrics['con_loss']:.4f} |"
                    f" GenLoss: {metrics['gen_loss']:.4f} |"
                    f" KLLoss: {metrics['kl_loss']:.4f} |"
                    f" LR: {metrics['lr']} |"
                    f" Iter time: {metrics['iter_time']:.4f}s |"
                )
            
            # Save checkpoint and evaluate each checkpoint interval steps or at the end of the epoch or at the end of training
            if current_step % checkpoint_iterval == 0 or current_step == lr_max_steps or batch_idx + 1 == len(train_loader):
                checkpoint_path = pathlib.Path(checkpoint_dir) / "lastest.ckpt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "current_step": current_step,
                    "epoch_num": epoch_num if batch_idx < len(train_loader)-1 else epoch_num + 1,
                }
                if checkpoint_filter is not None:
                    self.fabric.save(checkpoint_path, state, filter={'model': checkpoint_filter})
                else:
                    self.fabric.save(checkpoint_path, state)
                self.fabric.print(f"Checkpoint saved at {checkpoint_path}")
                clear_unused_gpu_mem()
                self.fabric.load(checkpoint_path, state, strict=False)
                model = state.pop("model")
                optimizer = state.pop("optimizer")
                scheduler = state.pop("scheduler")
                self.fabric.barrier()
                model_hprams = model.hprams

                # Evaluate model
                if self.fabric.global_rank == 0:
                    self.fabric.print("Evaluating model")
                    _model_revision = f"{model_revision}_step-{current_step}_epoch-{epoch_num}"
                    eval_model = WrappedULLME(
                        model_revision=_model_revision, 
                        model_checkpoint=checkpoint_path,
                        **model_hprams
                        )
                    mteb_results = eval_mteb(
                        model=eval_model,
                        output_folder=checkpoint_dir,
                        batch_size=eval_batch_size,
                        is_quick_run=True,
                    )
                    multilingual_results = eval_multilingual(
                        model=eval_model,
                        langs=['ru', 'vi', 'fa', 'hi', 'bn', 'yo'],
                        output_folder=checkpoint_dir,
                        batch_size=eval_batch_size,
                        is_quick_run=True,
                    )
                    results = {
                        'Avg/mteb_quick_avg': mteb_results['avg']['all_tasks'],
                        'Avg/multilingual_quick_avg': multilingual_results['avg']['all_tasks'],
                    }
                    self.fabric.log_dict(results, step=current_step)
                    # Eval logic here
                    self.fabric.print("Model evaluation finished")
                    del eval_model
                    clear_unused_gpu_mem()

                    # Save best checkpoint based on evaluation
                    if results['Avg/mteb_quick_avg'] > self.best_en:
                        self.best_en = results['Avg/mteb_quick_avg']
                        best_checkpoint_path = pathlib.Path(checkpoint_dir) / "best_en.ckpt"
                        shutil.copy(checkpoint_path, best_checkpoint_path)
                        self.fabric.print(f"Best en checkpoint saved at {best_checkpoint_path}")
                    if results['Avg/multilingual_quick_avg'] > self.best_multi:
                        self.best_multi = results['Avg/multilingual_quick_avg']
                        best_checkpoint_path = pathlib.Path(checkpoint_dir) / "best_multi.ckpt"
                        shutil.copy(checkpoint_path, best_checkpoint_path)
                        self.fabric.print(f"Best multi checkpoint saved at {best_checkpoint_path}")
                self.fabric.barrier()
        return checkpoint_path

