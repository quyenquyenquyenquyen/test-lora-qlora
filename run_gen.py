# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
import glob
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist
from peft import LoraConfig, get_peft_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def save_checkpoint(model, optimizer, scheduler, epoch, loss, args):
    """Lưu trạng thái model, optimizer, scheduler và epoch hiện tại."""
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }

    filepath = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, filepath)
    logger.info(f'Saved checkpoint at epoch {epoch} to {filepath}')

def load_latest_checkpoint(model, optimizer, scheduler, args):
    """Tải checkpoint mới nhất từ output_dir."""
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    latest_checkpoint_path = None
    latest_epoch = -1

    if os.path.exists(checkpoint_dir):
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt'))
        for f in checkpoint_files:
            try:
                epoch = int(os.path.basename(f).split('_')[-1].split('.')[0])
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_checkpoint_path = f
            except ValueError:
                continue

    if latest_checkpoint_path is None:
        logger.info("No valid checkpoint found in output_dir. Starting from epoch 0.")
        return model, optimizer, scheduler, 0, None

    logger.info(f"Loading checkpoint from: {latest_checkpoint_path}")
    try:
        checkpoint = torch.load(latest_checkpoint_path, map_location=args.device, weights_only=False)
        model_state_dict = checkpoint['model_state_dict']
        if hasattr(model, 'module') and not list(model_state_dict.keys())[0].startswith('module.'):
            model_state_dict = {'module.' + k: v for k, v in model_state_dict.items()}
        elif not hasattr(model, 'module') and list(model_state_dict.keys())[0].startswith('module.'):
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}

        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint.get('loss', None)

        logger.info(f"Loaded checkpoint from {latest_checkpoint_path}. Resuming from epoch {start_epoch}.")
        return model, optimizer, scheduler, start_epoch, loss
    except Exception as e:
        logger.error(f"Error loading checkpoint from {latest_checkpoint_path}: {e}")
        return model, optimizer, scheduler, 0, None

def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    """Đánh giá Perplexity (PPL) trên tập dữ liệu đánh giá."""
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)
    logger.info("  ***** Running PPL evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0.0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval PPL"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss

        if args.n_gpu > 1:
            loss = loss.mean()

        eval_loss += loss.item()
        batch_num += 1

    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl

def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    """Đánh giá BLEU và các metrics khác trên tập dữ liệu đánh giá/kiểm thử."""
    logger.info(f"  ***** Running BLEU evaluation on {split_tag} data *****")
    logger.info(f"  Num examples = {len(eval_examples)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4 if args.data_num == -1 else 0, pin_memory=True)

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc=f"Eval BLEU for {split_tag} set"):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)
                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                preds = model.generate(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    use_cache=True,
                    num_beams=args.beam_size,
                    early_stopping=args.task == 'summarize',
                    max_length=args.max_target_length,
                    num_return_sequences=args.beam_size
                )
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(id_seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for id_seq in pred_ids]

    pred_nls_grouped = []
    if args.beam_size > 1:
        for i in range(0, len(pred_nls), args.beam_size):
            pred_nls_grouped.append(pred_nls[i])
    else:
        pred_nls_grouped = pred_nls

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    output_fn = os.path.join(args.res_dir, f"test_{criteria}.output")
    gold_fn = os.path.join(args.res_dir, f"test_{criteria}.gold")
    src_fn = os.path.join(args.res_dir, f"test_{criteria}.src")

    result = {}

    if args.task == 'defect':
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        eval_acc = np.mean([int(p.strip() == g.strip()) for p, g in zip(pred_nls_grouped, golds)])
        result = {'em': eval_acc * 100, 'bleu': 0, 'codebleu': 0}

        with open(output_fn, 'w', encoding='utf-8') as f_out, \
             open(gold_fn, 'w', encoding='utf-8') as f_gold, \
             open(src_fn, 'w', encoding='utf-8') as f_src:
            for pred_nl, gold_ex in zip(pred_nls_grouped, eval_examples):
                f_out.write(pred_nl.strip() + '\n')
                f_gold.write(target_dict[gold_ex.target] + '\n')
                f_src.write(gold_ex.source.strip() + '\n')
            logger.info(f"Saved predictions, gold labels, and sources to {args.res_dir}")

    else:
        dev_accs = []
        predictions_for_bleu = []

        with open(output_fn, 'w', encoding='utf-8') as f_out, \
             open(gold_fn, 'w', encoding='utf-8') as f_gold, \
             open(src_fn, 'w', encoding='utf-8') as f_src:
            for i, (pred_nl, gold_ex) in enumerate(zip(pred_nls_grouped, eval_examples)):
                pred_clean = pred_nl.strip()
                gold_clean = gold_ex.target.strip()
                dev_accs.append(pred_clean == gold_clean)

                if args.task == 'summarize':
                    predictions_for_bleu.append(str(gold_ex.idx) + '\t' + pred_clean)
                    f_out.write(str(gold_ex.idx) + '\t' + pred_clean + '\n')
                    f_gold.write(str(gold_ex.idx) + '\t' + gold_clean + '\n')
                    f_src.write(str(gold_ex.idx) + '\t' + gold_ex.source.strip() + '\n')
                else:
                    predictions_for_bleu.append(pred_clean)
                    f_out.write(pred_clean + '\n')
                    f_gold.write(gold_clean + '\n')
                    f_src.write(gold_ex.source.strip() + '\n')

        logger.info(f"Saved predictions, gold labels, and sources to {args.res_dir}")

        if args.task == 'summarize':
            try:
                (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions_for_bleu, gold_fn)
                bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
            except Exception as e:
                logger.error(f"Error computing smooth BLEU: {e}")
                bleu = 0.0
        else:
            try:
                bleu = round(_bleu(gold_fn, output_fn), 2)
            except Exception as e:
                logger.error(f"Error computing BLEU: {e}")
                bleu = 0.0

        if args.task == 'concode':
            try:
                codebleu = calc_code_bleu([gold_fn], output_fn, args.lang)
                codebleu = round(codebleu * 100, 2)
            except Exception as e:
                logger.error(f"Error computing CodeBLEU: {e}")
                codebleu = 0.0
            result['codebleu'] = codebleu

        result['em'] = np.mean(dev_accs) * 100
        result['bleu'] = bleu

    logger.info(f"***** Eval results ({criteria}) *****")
    for key in sorted(result.keys()):
        logger.info(f"  {key} = {result[key]:.4f}")

    return result

def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    set_dist(args)
    set_seed(args)

    config, model, tokenizer = build_or_load_gen_model(args)

    if args.use_lora:
        logger.info("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q", "v"] if args.model_type in ['t5', 'codet5'] else ["q_proj", "v_proj"],
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, lora_config)
        logger.info("LoRA model created:")
        model.print_trainable_parameters()

    model.to(args.device)

    if args.n_gpu > 1 and args.local_rank == -1:
        logger.info(f"Using {args.n_gpu} GPUs with DataParallel.")
        model = torch.nn.DataParallel(model)

    cpu_count = args.cpu_cont
    pool = multiprocessing.Pool(cpu_count)

    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+', encoding='utf-8')

    if args.do_train:
        tb_writer = None
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = os.path.join(args.summary_dir, os.path.basename(args.output_dir))
            os.makedirs(os.path.dirname(summary_fn), exist_ok=True)
            tb_writer = SummaryWriter(summary_fn)
            logger.info(f"TensorBoard logs will be written to: {summary_fn}")

        train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train')
        logger.info(f"Loaded {len(train_examples)} training examples")

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)

        model, optimizer, scheduler, start_epoch, last_loss = load_latest_checkpoint(model, optimizer, scheduler, args)
        args.start_epoch = start_epoch

        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {train_example_num}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per GPU = {args.train_batch_size // args.n_gpu if args.n_gpu > 0 else args.train_batch_size}")
        logger.info(f"  Total train batch size = {args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {t_total}")
        logger.info(f"  Starting training from epoch {args.start_epoch}")

        dev_dataset = {}
        global_step = start_epoch * (len(train_dataloader) // args.gradient_accumulation_steps)
        best_bleu_em, best_ppl = -1.0, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            logger.info(f"--- Starting Epoch {cur_epoch} ---")
            if args.local_rank != -1:
                train_sampler.set_epoch(cur_epoch)

            bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {cur_epoch} Training",
                       disable=args.local_rank not in [-1, 0])
            nb_tr_steps = 0
            tr_loss = 0.0
            model.train()

            try:
                for step, batch in enumerate(bar):
                    batch = tuple(t.to(args.device) for t in batch)
                    source_ids, target_ids = batch
                    source_mask = source_ids.ne(tokenizer.pad_token_id)
                    target_mask = target_ids.ne(tokenizer.pad_token_id)

                    if args.model_type == 'roberta':
                        outputs = model(source_ids=source_ids, source_mask=source_mask,
                                        target_ids=target_ids, target_mask=target_mask)
                        loss = outputs[0]
                    else:
                        outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                        labels=target_ids, decoder_attention_mask=target_mask)
                        loss = outputs.loss

                    if args.n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    loss.backward()
                    tr_loss += loss.item() * args.gradient_accumulation_steps

                    if args.local_rank in [-1, 0]:
                        current_loss = tr_loss / (step + 1)
                        bar.set_postfix(loss=f'{current_loss:.3f}', lr=f'{scheduler.get_last_lr()[0]:.1e}')

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                        nb_tr_steps += 1

                        if args.local_rank in [-1, 0] and args.log_steps > 0 and global_step % args.log_steps == 0:
                            avg_loss = tr_loss / (step + 1)
                            if tb_writer:
                                tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                                tb_writer.add_scalar('train_loss', avg_loss, global_step)
                            logger.info(f"Step: {global_step}, Epoch: {cur_epoch}, Avg Train Loss: {avg_loss:.4f}")

                    if args.max_steps > 0 and global_step >= args.max_steps:
                        logger.info(f"Reached max_steps ({args.max_steps}). Stopping training.")
                        break

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

                if args.do_eval and args.local_rank in [-1, 0]:
                    logger.info(f"--- Evaluating Epoch {cur_epoch} ---")
                    if 'dev_loss' not in dev_dataset:
                        logger.info(f"Loading and caching dev data from {args.dev_filename}")
                        eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev')
                        dev_dataset['dev_loss'] = eval_examples, eval_data
                        logger.info(f"Loaded {len(eval_examples)} dev examples for PPL eval.")

                    eval_examples, eval_data = dev_dataset['dev_loss']
                    eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                    result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                    logger.info("***** PPL Evaluation Results *****")
                    for key, value in result.items():
                        logger.info(f"  {key} = {value}")
                    if tb_writer:
                        tb_writer.add_scalar('dev_ppl', eval_ppl, global_step)

                    if args.save_last_checkpoints:
                        last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                        os.makedirs(last_output_dir, exist_ok=True)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        if args.use_lora:
                            model_to_save.save_pretrained(last_output_dir)
                            logger.info(f"Saved LoRA adapter model to {last_output_dir}")
                        else:
                            output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info(f"Saved full model state dict to {output_model_file}")

                    if eval_ppl < best_ppl:
                        not_loss_dec_cnt = 0
                        logger.info(f"  New best PPL: {eval_ppl:.4f} (previous: {best_ppl:.4f})")
                        best_ppl = eval_ppl
                        fa.write(f"[{cur_epoch}/{global_step}] Best PPL updated to {best_ppl:.4f}\n")

                        if args.always_save_model:
                            output_dir_ppl = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                            os.makedirs(output_dir_ppl, exist_ok=True)
                            model_to_save = model.module if hasattr(model, 'module') else model
                            if args.use_lora:
                                model_to_save.save_pretrained(output_dir_ppl)
                                logger.info(f"Saved best PPL LoRA adapter model to {output_dir_ppl}")
                            else:
                                output_model_file = os.path.join(output_dir_ppl, "pytorch_model.bin")
                                torch.save(model_to_save.state_dict(), output_model_file)
                                logger.info(f"Saved best PPL full model state dict to {output_model_file}")
                    else:
                        not_loss_dec_cnt += 1
                        logger.info(f"PPL ({eval_ppl:.4f}) did not improve. Count: {not_loss_dec_cnt}/{args.patience}")

                    torch.cuda.empty_cache()

                    if args.do_eval_bleu:
                        if 'dev_bleu' not in dev_dataset:
                            logger.info(f"Loading and caching dev data from {args.dev_filename} for BLEU eval")
                            eval_examples_bleu, eval_data_bleu = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                                        only_src=True, is_sample=False)
                            dev_dataset['dev_bleu'] = eval_examples_bleu, eval_data_bleu
                            logger.info(f"Loaded {len(eval_examples_bleu)} dev examples for BLEU eval.")

                        eval_examples_bleu, eval_data_bleu = dev_dataset['dev_bleu']
                        result_bleu = eval_bleu_epoch(args, eval_data_bleu, eval_examples_bleu, model, tokenizer, 'dev', f'e{cur_epoch}_gs{global_step}')
                        dev_bleu = result_bleu.get('bleu', 0.0)
                        dev_em = result_bleu.get('em', 0.0)
                        dev_codebleu = result_bleu.get('codebleu', 0.0)

                        if args.task == 'summarize':
                            dev_bleu_em = dev_bleu
                        elif args.task == 'defect':
                            dev_bleu_em = dev_em
                        else:
                            dev_bleu_em = dev_bleu + dev_em

                        logger.info(f"***** BLEU/EM Evaluation Results (Epoch {cur_epoch}) *****")
                        logger.info(f"  BLEU = {dev_bleu:.4f}")
                        logger.info(f"  EM = {dev_em:.4f}")
                        if 'codebleu' in result_bleu:
                            logger.info(f"  CodeBLEU = {dev_codebleu:.4f}")
                        logger.info(f"  Combined Metric = {dev_bleu_em:.4f}")

                        if tb_writer:
                            tb_writer.add_scalar('dev_bleu', dev_bleu, global_step)
                            tb_writer.add_scalar('dev_em', dev_em, global_step)
                            tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, global_step)
                            if 'codebleu' in result_bleu:
                                tb_writer.add_scalar('dev_codebleu', dev_codebleu, global_step)

                        if dev_bleu_em > best_bleu_em:
                            not_bleu_em_inc_cnt = 0
                            logger.info(f"  New best BLEU/EM: {dev_bleu_em:.4f} (previous: {best_bleu_em:.4f})")
                            best_bleu_em = dev_bleu_em
                            fa.write(f"[{cur_epoch}/{global_step}] Best BLEU/EM updated to {best_bleu_em:.4f} (BLEU: {dev_bleu:.2f}, EM: {dev_em:.2f})\n")

                            if args.always_save_model:
                                output_dir_bleu = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                                os.makedirs(output_dir_bleu, exist_ok=True)
                                model_to_save = model.module if hasattr(model, 'module') else model
                                if args.use_lora:
                                    model_to_save.save_pretrained(output_dir_bleu)
                                    logger.info(f"Saved best BLEU/EM LoRA adapter model to {output_dir_bleu}")
                                else:
                                    output_model_file = os.path.join(output_dir_bleu, "pytorch_model.bin")
                                    torch.save(model_to_save.state_dict(), output_model_file)
                                    logger.info(f"Saved best BLEU/EM full model state dict to {output_model_file}")
                        else:
                            not_bleu_em_inc_cnt += 1
                            logger.info(f"BLEU/EM ({dev_bleu_em:.4f}) did not improve. Count: {not_bleu_em_inc_cnt}/{args.patience}")

                    if args.patience > 0 and all([x >= args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                        early_stop_str = f"[{cur_epoch}] Early stopping: PPL count {not_loss_dec_cnt}, BLEU/EM count {not_bleu_em_inc_cnt} >= patience {args.patience}"
                        logger.info(early_stop_str)
                        fa.write(early_stop_str + "\n")
                        break

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Saving current state...")
                if args.local_rank in [-1, 0]:
                    avg_epoch_loss = tr_loss / nb_tr_steps if nb_tr_steps > 0 else 0.0
                    save_checkpoint(model, optimizer, scheduler, cur_epoch + 1, avg_epoch_loss, args)
                torch.cuda.empty_cache()
                break

            if args.local_rank in [-1, 0]:
                avg_epoch_loss = tr_loss / nb_tr_steps if nb_tr_steps > 0 else 0.0
                save_checkpoint(model, optimizer, scheduler, cur_epoch + 1, avg_epoch_loss, args)

            logger.info("***** CUDA.empty_cache() at end of epoch *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and tb_writer:
            tb_writer.close()
        logger.info(f"Finish training. Total time: {get_elapse_time(t0)}")
        fa.write(f"Finish training. Total time: {get_elapse_time(t0)}\n")

    if args.do_test and args.local_rank in [-1, 0]:
        logger.info("***** Running Testing *****")
        logger.info(f"  Batch size = {args.eval_batch_size}")

        checkpoints_to_test = []
        if os.path.exists(os.path.join(args.output_dir, 'checkpoint-best-bleu')):
            checkpoints_to_test.append('best-bleu')
        if os.path.exists(os.path.join(args.output_dir, 'checkpoint-best-ppl')):
            checkpoints_to_test.append('best-ppl')
        if os.path.exists(os.path.join(args.output_dir, 'checkpoint-last')):
            checkpoints_to_test.append('last')

        if not checkpoints_to_test:
            logger.warning("No checkpoints found to test in the output directory.")

        test_examples, test_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test',
                                                           only_src=True, is_sample=False)
        logger.info(f"Loaded {len(test_examples)} test examples.")

        for criteria in checkpoints_to_test:
            checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-{criteria}')
            logger.info(f"--- Testing checkpoint: {criteria} from {checkpoint_dir} ---")

            if args.use_lora:
                _, model_base, _ = build_or_load_gen_model(args)
                model = get_peft_model(model_base, LoraConfig.from_pretrained(checkpoint_dir))
                model.to(args.device)
                logger.info(f"Reloaded LoRA adapter from {checkpoint_dir}")
            else:
                file = os.path.join(checkpoint_dir, 'pytorch_model.bin')
                if os.path.exists(file):
                    model.load_state_dict(torch.load(file, map_location=args.device))
                    logger.info(f"Reloaded full model state dict from {file}")
                else:
                    logger.warning(f"Model file {file} not found for criteria {criteria}. Skipping test.")
                    continue

            if args.n_gpu > 1 and args.local_rank == -1 and not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model)

            result = eval_bleu_epoch(args, test_data, test_examples, model, tokenizer, 'test', criteria)
            test_bleu = result.get('bleu', 0.0)
            test_em = result.get('em', 0.0)
            test_codebleu = result.get('codebleu', 0.0)

            result_str = f"[{criteria}] Test Results -> BLEU: {test_bleu:.2f}, EM: {test_em:.4f}"
            if 'codebleu' in result:
                result_str += f", CodeBLEU: {test_codebleu:.4f}"
            result_str += "\n"
            logger.info(result_str.strip())
            fa.write(result_str)

            if args.res_fn:
                res_file_path = args.res_fn if os.path.isabs(args.res_fn) else os.path.join(args.output_dir, args.res_fn)
                os.makedirs(os.path.dirname(res_file_path), exist_ok=True)
                with open(res_file_path, 'a+', encoding='utf-8') as f_res:
                    f_res.write(f'[Time: {get_elapse_time(t0)}] Checkpoint: {checkpoint_dir}\n')
                    f_res.write(result_str)

            torch.cuda.empty_cache()

    logger.info(f"Finish All. Total time: {get_elapse_time(t0)}")
    fa.write(f"Finish All. Total time: {get_elapse_time(t0)}\n")
    fa.close()
    pool.close()
    pool.join()

if __name__ == "__main__":
    t0 = time.time()
    main()
