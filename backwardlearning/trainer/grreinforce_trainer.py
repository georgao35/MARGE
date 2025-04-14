from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import SequentialSampler

from trl import DPOTrainer
from trl.trainer.dpo_config import FDivergenceType, FDivergenceConstants
from trl.trainer.utils import cap_exp
from transformers import PreTrainedModel
from transformers.trainer_utils import (
    enable_full_determinism,
    get_last_checkpoint,
    set_seed,
    find_executable_batch_size,
)


class GroupRelativeReinforceTrainer(DPOTrainer):
    stored_metrics_logs = defaultdict(lambda: defaultdict(list))

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        print(batch)
        super().get_batch_loss_metrics(model, batch, train_eval)

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        # print(f"{policy_chosen_logps}, {policy_chosen_logps.shape=}")
        # print(f"{policy_rejected_logps}, {policy_rejected_logps.shape=}")
        chosen_logratios = policy_chosen_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_chosen_logps.to(self.accelerator.device)
        rejected_logratios = policy_rejected_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_rejected_logps.to(self.accelerator.device)

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            # The alpha-divergence formula: (1 - u^-alpha) / alpha
            # The divergence difference between the chosen and rejected sample is:
            #     (1 - u[w]^-alpha) / alpha - (1 - u[l]^-alpha) / alpha
            #        = (u[l]^-alpha - u[w]^-alpha) / alpha
            # where u[w] and u[l] are the policy/reference probability ratios
            # for the chosen and rejected samples, respectively.
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
            else:
                ref_logratios = reference_chosen_logps - reference_rejected_logps

            pi_logratios = pi_logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        # if self.loss_type == "sigmoid":
        #     losses = (
        #         -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
        #         - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        #     )
        losses = -pi_logratios

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            # Last prompt token might get merged by tokenizer and
            # it should not be included for generation if that happens
            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # Make sure prompts only have one different token at most an
            # and length only differs by 1 at most
            num_diff_tokens = sum(
                [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
            )
            num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
            if num_diff_tokens > 1 or num_diff_len > 1:
                raise ValueError(
                    "Chosen and rejected prompt_input_ids might only differ on the "
                    "last token due to tokenizer merge ops."
                )

            # add BOS token to head of prompt
            if self.tokenizer.bos_token_id is not None:
                bos_token_id = self.tokenizer.bos_token_id
                if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
                    prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
                    prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]

            eos_token_id = self.tokenizer.eos_token_id
            if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
                chosen_tokens["input_ids"].append(eos_token_id)
                chosen_tokens["attention_mask"].append(1)
            if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
                rejected_tokens["input_ids"].append(eos_token_id)
                rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            # Create labels
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(chosen_tokens["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(rejected_tokens["prompt_input_ids"])

            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "rejected_": rejected_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

            # import pdb; pdb.set_trace()

        else:
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["rejected_labels"])
                )
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["chosen_labels"])
                )

        return batch

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
            self.stored_metrics_logs[train_eval][key].append(logs[key])
        del self._stored_metrics[train_eval]
        return super().log(logs)

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if resume_from_checkpoint and self.is_deepspeed_enabled:
            resume_from_checkpoint = get_last_checkpoint(resume_from_checkpoint)

        if resume_from_checkpoint is not None:
            if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )

        return inner_training_loop(
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )


class GRReinforceTrainer(GroupRelativeReinforceTrainer):
    
    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return per_token_logps, loss_mask
        # return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], return_size: bool=False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            is_vision_model=self.is_vision_model,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {}

        if self.is_encoder_decoder:
            model_kwargs["labels"] = concatenated_batch["concatenated_labels"]
            model_kwargs["decoder_input_ids"] = concatenated_batch.pop("concatenated_decoder_input_ids", None)

        if self.is_vision_model:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            # average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch["concatenated_labels"].clone()
        nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        # if self.loss_type == "ipo":
        #     all_logps = all_logps / size_completion

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, outputs.aux_loss)

        if return_size:
            chosen_size = size_completion[:len_chosen]
            rejected_size = size_completion[len_chosen:]
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_size, rejected_size, nll_loss)

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        forward_output = self.concatenated_forward(model, batch, return_size=True)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_mask,
            policy_rejected_mask,
            policy_nll_loss,
        ) = forward_output[:7]
        # if self.aux_loss_enabled:
        #     aux_loss = forward_output[7]

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if (
            "reference_chosen_logps" in batch
            and "reference_rejected_logps" in batch
            and self.args.rpo_alpha is not None
        ):
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_logp, rejected_logp, kl_chosen, kl_rejected = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            batch["chosen_adv"],
            batch["rejected_adv"],
            policy_chosen_mask,
            policy_rejected_mask
        )

        if self.args.rpo_alpha is not None:
            losses = losses * self.args.rpo_alpha + policy_nll_loss

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}logps/chosen"] = chosen_logp.mean().cpu()
        metrics[f"{prefix}logps/rejected"] = rejected_logp.mean().cpu()
        metrics[f"{prefix}kl/chosen"] = kl_chosen.mean().cpu()
        metrics[f"{prefix}kl/rejected"] = kl_rejected.mean().cpu()
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()

        return losses.mean(), metrics

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        chosen_advantage: float | torch.FloatTensor,
        rejected_advantage: float | torch.FloatTensor,
        chosen_mask: torch.FloatTensor,
        rejected_mask: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        chosen_advantage = torch.FloatTensor(chosen_advantage).to(self.accelerator.device)
        rejected_advantage = torch.FloatTensor(rejected_advantage).to(self.accelerator.device)

        policy_chosen_logps = policy_chosen_logps.to(self.accelerator.device)
        policy_rejected_logps = policy_rejected_logps.to(self.accelerator.device)
        chosen_mask = chosen_mask.to(self.accelerator.device)
        chosen_size = chosen_mask.sum(-1)
        rejected_mask = rejected_mask.to(self.accelerator.device)
        rejected_size = rejected_mask.sum(-1)
        chosen_logratios = reference_chosen_logps.to(self.accelerator.device) - policy_chosen_logps
        rejected_logratios = reference_rejected_logps.to(self.accelerator.device) - policy_rejected_logps
        # chosen_kl_estimate = (chosen_logratios).detach()
        # chosen_kl_estimate = (chosen_kl_estimate * chosen_mask).sum(-1)
        chosen_klloss = (chosen_logratios * chosen_mask).sum(-1) / chosen_size
        # rejected_kl_estimate = (rejected_logratios).detach()
        # rejected_kl_estimate = (rejected_kl_estimate * rejected_mask).sum(-1)
        rejected_klloss = (rejected_logratios * rejected_mask).sum(-1) / rejected_size
        # chosen_kl = chosen_logratios.exp() - chosen_logratios - 1
        # rejected_kl = rejected_logratios.exp() - rejected_logratios - 1

        # kl_losses = self.beta * (chosen_kl / chosen_size + rejected_kl / rejected_size)
        # losses = -(policy_chosen_logps * chosen_advantage / chosen_size + policy_rejected_logps * rejected_advantage / rejected_size) + kl_losses
        kl_losses = self.beta * (chosen_klloss + rejected_klloss)
        pgloss_chosen = torch.sum(policy_chosen_logps * chosen_mask, -1) / chosen_size
        pgloss_rejected = torch.sum(policy_rejected_logps * rejected_mask, -1) / rejected_size
        losses = chosen_advantage * pgloss_chosen + rejected_advantage * pgloss_rejected
        losses = -losses + kl_losses
        losses /= 2

        return losses, pgloss_chosen.detach(), pgloss_rejected.detach(), chosen_klloss.detach(), rejected_klloss.detach()


class GRBatchReinforceTrainer(GRReinforceTrainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None:
            return None

        return SequentialSampler(self.train_dataset)


class GRPOTrainer(GRBatchReinforceTrainer):

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        forward_output = self.concatenated_forward(model, batch, return_size=True)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_mask,
            policy_rejected_mask,
            policy_nll_loss,
        ) = forward_output[:7]
        # if self.aux_loss_enabled:
        #     aux_loss = forward_output[7]

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if (
            "reference_chosen_logps" in batch
            and "reference_rejected_logps" in batch
            and self.args.rpo_alpha is not None
        ):
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_logp, rejected_logp, kl_chosen, kl_rejected = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            batch["chosen_adv"],
            batch["rejected_adv"],
            policy_chosen_mask,
            policy_rejected_mask
        )

        if self.args.rpo_alpha is not None:
            losses = losses * self.args.rpo_alpha + policy_nll_loss

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}logps/chosen"] = chosen_logp.mean().cpu()
        metrics[f"{prefix}logps/rejected"] = rejected_logp.mean().cpu()
        metrics[f"{prefix}kl/chosen"] = kl_chosen.mean().cpu()
        metrics[f"{prefix}kl/rejected"] = kl_rejected.mean().cpu()
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()

        return losses.mean(), metrics

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        chosen_advantage: float | torch.FloatTensor,
        rejected_advantage: float | torch.FloatTensor,
        chosen_mask: torch.FloatTensor,
        rejected_mask: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        chosen_advantage = torch.FloatTensor(chosen_advantage).to(self.accelerator.device)
        rejected_advantage = torch.FloatTensor(rejected_advantage).to(self.accelerator.device)

        # calculate loss for chosen
        chosen_mask = chosen_mask.to(self.accelerator.device)
        chosen_size = chosen_mask.sum(-1)

        policy_chosen_logps = policy_chosen_logps.to(self.accelerator.device)
        policy_chosen_oldlogps = policy_chosen_logps.detach()
        policy_chosen_logp_ratio = policy_chosen_logps - policy_chosen_oldlogps
        ploss_chosen = chosen_advantage * policy_chosen_logp_ratio.exp()
        ploss_chosen = (ploss_chosen * chosen_mask).sum(-1) / chosen_size

        reference_chosen_logps = reference_chosen_logps.to(self.accelerator.device)
        chosen_log_ratio = (reference_chosen_logps - policy_chosen_logps) * chosen_mask
        klloss_chosen = chosen_log_ratio.sum(-1) / chosen_size
        kl_chosen = chosen_log_ratio.exp() - chosen_log_ratio - 1
        kl_chosen = kl_chosen.detach()
        kl_chosen_indicator = (kl_chosen * chosen_mask).sum(-1)
        
        # calculate loss for rejected
        rejected_mask = rejected_mask.to(self.accelerator.device)
        rejected_size = rejected_mask.sum(-1)

        policy_rejected_logps = policy_rejected_logps.to(self.accelerator.device)
        policy_rejected_oldlogps = policy_rejected_logps.detach()
        policy_rejected_logp_ratio = policy_rejected_logps - policy_rejected_oldlogps
        ploss_rejected = rejected_advantage * policy_rejected_logp_ratio.exp()
        ploss_rejected = (ploss_rejected * rejected_mask).sum(-1) / rejected_size

        reference_rejected_logps = reference_rejected_logps.to(self.accelerator.device)
        rejected_log_ratio = (reference_rejected_logps - policy_rejected_logps) * rejected_mask
        klloss_rejected = rejected_log_ratio.sum(-1) / chosen_size
        kl_rejected = rejected_log_ratio.exp() - rejected_log_ratio - 1
        kl_rejected = kl_rejected.detach()
        kl_rejected_indicator = (kl_rejected * rejected_mask).sum(-1)

        losses = - ploss_chosen - ploss_rejected + self.beta * (klloss_rejected + klloss_chosen)
        losses /= 2

        return losses, ploss_chosen.detach(), ploss_rejected.detach(), kl_chosen_indicator, kl_rejected_indicator


class GRReinforceTrainerDebug(GRReinforceTrainer):

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], return_size: bool=False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            is_vision_model=self.is_vision_model,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {}

        if self.is_encoder_decoder:
            model_kwargs["labels"] = concatenated_batch["concatenated_labels"]
            model_kwargs["decoder_input_ids"] = concatenated_batch.pop("concatenated_decoder_input_ids", None)

        if self.is_vision_model:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            # average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch["concatenated_labels"].clone()
        nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        # if self.loss_type == "ipo":
        #     all_logps = all_logps / size_completion

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        from IPython import embed; embed()

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, outputs.aux_loss)

        if return_size:
            chosen_size = size_completion[:len_chosen]
            rejected_size = size_completion[len_chosen:]
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_size, rejected_size, nll_loss)

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        # print(batch)
        

        forward_output = self.concatenated_forward(model, batch, return_size=True)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_mask,
            policy_rejected_mask,
            policy_nll_loss,
        ) = forward_output[:7]
        # if self.aux_loss_enabled:
        #     aux_loss = forward_output[7]

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if batch['query'][0] == '15 men take 21 days of 8 hrs. each to do a piece of work. How many days of 6 hrs. each would it take for 21 women if 3 women do as much work as 2 men?':
            from ipdb import set_trace; set_trace()
        if (
            "reference_chosen_logps" in batch
            and "reference_rejected_logps" in batch
            and self.args.rpo_alpha is not None
        ):
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            batch["chosen_adv"],
            batch["rejected_adv"],
            policy_chosen_mask,
            policy_rejected_mask
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses * self.args.rpo_alpha + policy_nll_loss

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()

        if self.aux_loss_enabled:
            return losses.mean() + getattr(model.config, "router_aux_loss_coef", 0.0) * aux_loss, metrics

        return losses.mean(), metrics

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        chosen_advantage: float | torch.FloatTensor,
        rejected_advantage: float | torch.FloatTensor,
        chosen_mask: torch.FloatTensor,
        rejected_mask: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        # print(f"{policy_chosen_logps}, {policy_chosen_logps.shape=}")
        # print(f"{policy_rejected_logps}, {policy_rejected_logps.shape=}")
        # chosen_logratios = policy_chosen_logps.to(self.accelerator.device) - (
        #     not self.reference_free
        # ) * reference_chosen_logps.to(self.accelerator.device)
        # rejected_logratios = policy_rejected_logps.to(self.accelerator.device) - (
        #     not self.reference_free
        # ) * reference_rejected_logps.to(self.accelerator.device)
        chosen_advantage = torch.FloatTensor(chosen_advantage).to(self.accelerator.device)
        rejected_advantage = torch.FloatTensor(rejected_advantage).to(self.accelerator.device)
        # chosen_size = torch.FloatTensor(chosen_size).to(self.accelerator.device)
        # rejected_size = torch.FloatTensor(rejected_size).to(self.accelerator.device)

        # if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
        #     # The alpha-divergence formula: (1 - u^-alpha) / alpha
        #     # The divergence difference between the chosen and rejected sample is:
        #     #     (1 - u[w]^-alpha) / alpha - (1 - u[l]^-alpha) / alpha
        #     #        = (u[l]^-alpha - u[w]^-alpha) / alpha
        #     # where u[w] and u[l] are the policy/reference probability ratios
        #     # for the chosen and rejected samples, respectively.
        #     alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
        #     if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
        #         alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
        #     logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        # else:
        #     pi_logratios = policy_chosen_logps - policy_rejected_logps
        #     if self.reference_free:
        #         ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
        #     else:
        #         ref_logratios = reference_chosen_logps - reference_rejected_logps

        #     pi_logratios = pi_logratios.to(self.accelerator.device)
        #     ref_logratios = ref_logratios.to(self.accelerator.device)
        #     logits = pi_logratios - ref_logratios

        # print(f"{policy_chosen_logps=}")
        # print(f"{policy_rejected_logps=}")
        # print(f"{reference_chosen_logps=}")
        # print(f"{reference_rejected_logps=}")
        # print(f"{chosen_advantage=}")
        # print(f"{rejected_advantage=}")
        # print(f"{chosen_mask=}")
        # print(f"{rejected_mask=}")

        # from IPython import embed; embed()

        policy_chosen_logps = policy_chosen_logps.to(self.accelerator.device)
        policy_rejected_logps = policy_rejected_logps.to(self.accelerator.device)
        chosen_mask = chosen_mask.to(self.accelerator.device)
        chosen_size = chosen_mask.sum(-1)
        # print(f"{chosen_size=}")
        rejected_mask = rejected_mask.to(self.accelerator.device)
        rejected_size = rejected_mask.sum(-1)
        # print(f"{rejected_size=}")
        chosen_logratios = reference_chosen_logps.to(self.accelerator.device) - policy_chosen_logps
        rejected_logratios = reference_rejected_logps.to(self.accelerator.device) - policy_rejected_logps
        chosen_kl_estimate = chosen_logratios.exp() - chosen_logratios - 1
        chosen_kl = (chosen_kl_estimate * chosen_mask).sum(-1) / chosen_size
        rejected_kl_estimate = rejected_logratios.exp() - rejected_logratios - 1
        rejected_kl = (rejected_kl_estimate * rejected_mask).sum(-1) / rejected_size
        # chosen_kl = chosen_logratios.exp() - chosen_logratios - 1
        # rejected_kl = rejected_logratios.exp() - rejected_logratios - 1

        # kl_losses = self.beta * (chosen_kl / chosen_size + rejected_kl / rejected_size)
        # losses = -(policy_chosen_logps * chosen_advantage / chosen_size + policy_rejected_logps * rejected_advantage / rejected_size) + kl_losses
        kl_losses = self.beta * (chosen_kl + rejected_kl)
        chosen_loss = torch.sum(policy_chosen_logps * chosen_mask * chosen_advantage, -1) / chosen_size
        print(f"logp: {policy_rejected_logps.isnan().any()}, {policy_rejected_logps.isinf().any()}")
        print(f"rejected mask: {rejected_mask.isnan().any()}, {rejected_mask.isinf().any()}")
        print(f"{rejected_size=}")
        print(f"{rejected_advantage=}")
        rejected_loss = torch.sum(policy_rejected_logps * rejected_mask * rejected_advantage, -1) / rejected_size
        losses = chosen_loss + rejected_loss
        losses = -losses + kl_losses
        losses /= 2

        return losses, kl_losses, (chosen_kl + rejected_kl) / 2
