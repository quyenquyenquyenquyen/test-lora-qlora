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

# Hàm lưu checkpoint (sử dụng /kaggle/working/)
def save_checkpoint(model, optimizer, scheduler, epoch, loss, args):
    """Lưu trạng thái model, optimizer, scheduler và epoch hiện tại."""
    checkpoint_dir = os.path.join("/kaggle/working/", os.path.basename(args.output_dir), 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }

    # Lưu checkpoint với epoch + 1 vì epoch là epoch *vừa hoàn thành*
    filepath = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, filepath)
    logger.info(f'Saved checkpoint at epoch {epoch} to {filepath}')

# Hàm tải checkpoint gần nhất (tìm trong /kaggle/input/ hoặc /kaggle/working/)
def load_latest_checkpoint(model, optimizer, scheduler, args):
    """Tải checkpoint mới nhất từ /kaggle/working/ hoặc /kaggle/input/."""
    # Đường dẫn trong /kaggle/working/ (cho phiên hiện tại)
    working_checkpoint_dir = os.path.join("/kaggle/working/", os.path.basename(args.output_dir), 'checkpoints')
    # Đường dẫn trong /kaggle/input/ (cho dataset từ phiên bản trước)
    # Giả sử bạn đã tạo dataset với tên 'my-model-checkpoints'
    # Thay 'my-model-checkpoints' bằng tên dataset thực tế của bạn nếu có
    input_checkpoint_dir_pattern = "/kaggle/input/*checkpoints*/checkpoints" # Mẫu để tìm dataset checkpoints
    input_checkpoint_dirs = glob.glob(input_checkpoint_dir_pattern)
    input_checkpoint_dir = input_checkpoint_dirs[0] if input_checkpoint_dirs else None

    latest_checkpoint_path = None
    latest_epoch = -1

    # 1. Tìm checkpoint mới nhất trong /kaggle/working/
    if os.path.exists(working_checkpoint_dir):
        working_files = glob.glob(os.path.join(working_checkpoint_dir, 'checkpoint_epoch_*.pt'))
        for f in working_files:
            try:
                epoch = int(os.path.basename(f).split('_')[-1].split('.')[0])
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_checkpoint_path = f
            except ValueError:
                continue # Bỏ qua file không đúng định dạng tên

    # 2. Tìm checkpoint mới nhất trong /kaggle/input/ nếu có
    if input_checkpoint_dir and os.path.exists(input_checkpoint_dir):
        input_files = glob.glob(os.path.join(input_checkpoint_dir, 'checkpoint_epoch_*.pt'))
        for f in input_files:
             try:
                epoch = int(os.path.basename(f).split('_')[-1].split('.')[0])
                # Chỉ cập nhật nếu epoch trong input lớn hơn epoch trong working
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_checkpoint_path = f
             except ValueError:
                continue # Bỏ qua file không đúng định dạng tên

    # Nếu không tìm thấy checkpoint nào
    if latest_checkpoint_path is None:
        logger.info("Không tìm thấy checkpoint hợp lệ nào trong /kaggle/working/ hoặc /kaggle/input/. Bắt đầu từ đầu (epoch 0).")
        return model, optimizer, scheduler, 0, None # Bắt đầu từ epoch 0

    # Tải checkpoint đã tìm thấy
    logger.info(f"Đang tải checkpoint từ: {latest_checkpoint_path}")
    try:
        # Thêm map_location để đảm bảo tải được trên các thiết bị khác nhau (CPU/GPU)
        # weights_only=False là cần thiết để tải cả optimizer và scheduler state dicts
        checkpoint = torch.load(latest_checkpoint_path, map_location=args.device, weights_only=False)

        # Tải state dicts, xử lý trường hợp DataParallel/DistributedDataParallel
        model_state_dict = checkpoint['model_state_dict']
        if hasattr(model, 'module') and not list(model_state_dict.keys())[0].startswith('module.'):
             # Nếu model hiện tại là DataParallel/DDP nhưng checkpoint không có tiền tố 'module.'
             model_state_dict = {'module.' + k: v for k, v in model_state_dict.items()}
        elif not hasattr(model, 'module') and list(model_state_dict.keys())[0].startswith('module.'):
             # Nếu model hiện tại không phải DataParallel/DDP nhưng checkpoint có tiền tố 'module.'
              model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}

        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] # Epoch đã hoàn thành, nên epoch tiếp theo là start_epoch
        loss = checkpoint.get('loss', None) # Dùng get để tránh lỗi nếu key 'loss' không có

        logger.info(f"Đã tải checkpoint từ {latest_checkpoint_path}. Tiếp tục từ epoch {start_epoch}.")
        return model, optimizer, scheduler, start_epoch, loss
    except Exception as e:
        logger.error(f"Lỗi khi tải checkpoint từ {latest_checkpoint_path}: {e}")
        logger.info("Không thể tải checkpoint. Bắt đầu từ đầu (epoch 0).")
        return model, optimizer, scheduler, 0, None # Bắt đầu từ epoch 0 nếu có lỗi

def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    """Đánh giá Perplexity (PPL) trên tập dữ liệu đánh giá."""
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0.0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        # Chuyển batch lên device
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                # Roberta (Encoder-Decoder) thường có interface riêng
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            else:
                # Các model Seq2Seq khác (T5, BART) dùng interface chuẩn của Hugging Face
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss

        # Xử lý nếu dùng DataParallel hoặc DistributedDataParallel
        if args.n_gpu > 1:
            loss = loss.mean() # Lấy trung bình loss trên các GPU

        eval_loss += loss.item()
        batch_num += 1

    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    """Đánh giá BLEU và các metrics khác trên tập dữ liệu đánh giá/kiểm thử."""
    logger.info(f"  ***** Running bleu evaluation on {split_tag} data *****")
    logger.info(f"  Num examples = {len(eval_examples)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    eval_sampler = SequentialSampler(eval_data)
    # Kiểm tra args.data_num để quyết định num_workers
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        # Nếu chỉ dùng một phần dữ liệu, không cần nhiều worker
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0 # Khởi tạo giá trị

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc=f"Eval bleu for {split_tag} set"):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                 # Model Roberta (Encoder-Decoder) có thể cần gọi khác
                preds = model(source_ids=source_ids, source_mask=source_mask)
                # Giả sử preds trả về list các tensor, lấy top-1 prediction
                # Cần kiểm tra cấu trúc trả về của model roberta cụ thể của bạn
                top_preds = [pred[0].cpu().numpy() for pred in preds] # Lấy beam đầu tiên
            else:
                # Các model Seq2Seq chuẩn (T5, BART) dùng phương thức generate
                # Đã sửa lỗi: sử dụng input_ids thay vì source_ids cho generate
                preds = model.generate(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    use_cache=True,
                    num_beams=args.beam_size,
                    early_stopping=args.task == 'summarize', # Dừng sớm cho tóm tắt
                    max_length=args.max_target_length,
                    num_return_sequences=args.beam_size # Trả về đủ số beam
                )
                # Chuyển kết quả về CPU và thành list numpy
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    # Decode các IDs thành văn bản
    pred_nls = [tokenizer.decode(id_seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for id_seq in pred_ids]

    # Gom các kết quả từ beam search lại
    # Mỗi ví dụ gốc sẽ có args.beam_size kết quả dự đoán liên tiếp trong pred_nls
    pred_nls_grouped = []
    if args.beam_size > 1:
         for i in range(0, len(pred_nls), args.beam_size):
              # Lấy dự đoán tốt nhất (thường là cái đầu tiên trong mỗi nhóm beam)
              pred_nls_grouped.append(pred_nls[i])
    else:
        pred_nls_grouped = pred_nls # Nếu beam_size = 1, không cần gom nhóm

    # Tạo thư mục chứa kết quả nếu chưa tồn tại
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    # Định nghĩa đường dẫn file output
    output_fn = os.path.join(args.res_dir, f"test_{criteria}.output")
    gold_fn = os.path.join(args.res_dir, f"test_{criteria}.gold")
    src_fn = os.path.join(args.res_dir, f"test_{criteria}.src")

    result = {} # Khởi tạo dict kết quả

    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        # Tính accuracy (Exact Match - EM)
        eval_acc = np.mean([int(p.strip() == g.strip()) for p, g in zip(pred_nls_grouped, golds)])
        result = {'em': eval_acc * 100, 'bleu': 0, 'codebleu': 0} # Defect detection thường dùng accuracy

        with open(output_fn, 'w', encoding='utf-8') as f_out, \
             open(gold_fn, 'w', encoding='utf-8') as f_gold, \
             open(src_fn, 'w', encoding='utf-8') as f_src:
            for pred_nl, gold_ex in zip(pred_nls_grouped, eval_examples):
                f_out.write(pred_nl.strip() + '\n')
                f_gold.write(target_dict[gold_ex.target] + '\n')
                f_src.write(gold_ex.source.strip() + '\n')
            logger.info(f"Saved predictions, gold labels, and sources to {args.res_dir}")

    else:
        # Các task khác (summarize, translate, concode, etc.)
        dev_accs = [] # List để lưu kết quả EM cho từng ví dụ
        predictions_for_bleu = [] # List để lưu dự đoán cho tính BLEU

        with open(output_fn, 'w', encoding='utf-8') as f_out, \
             open(gold_fn, 'w', encoding='utf-8') as f_gold, \
             open(src_fn, 'w', encoding='utf-8') as f_src:
            for i, (pred_nl, gold_ex) in enumerate(zip(pred_nls_grouped, eval_examples)):
                pred_clean = pred_nl.strip()
                gold_clean = gold_ex.target.strip()

                # Tính EM
                dev_accs.append(pred_clean == gold_clean)

                # Ghi file output
                if args.task in ['summarize']:
                    # Ghi kèm index cho ROUGE/BLEU tính toán sau này nếu cần
                    predictions_for_bleu.append(str(gold_ex.idx) + '\t' + pred_clean)
                    f_out.write(str(gold_ex.idx) + '\t' + pred_clean + '\n')
                    f_gold.write(str(gold_ex.idx) + '\t' + gold_clean + '\n')
                    f_src.write(str(gold_ex.idx) + '\t' + gold_ex.source.strip() + '\n')
                else:
                    # Các task khác không cần index
                    predictions_for_bleu.append(pred_clean)
                    f_out.write(pred_clean + '\n')
                    f_gold.write(gold_clean + '\n')
                    f_src.write(gold_ex.source.strip() + '\n')

        logger.info(f"Saved predictions, gold labels, and sources to {args.res_dir}")

        # Tính BLEU score
        if args.task == 'summarize':
            # Sử dụng smooth_bleu cho summarize (thường yêu cầu file có index)
            try:
                (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions_for_bleu, gold_fn)
                bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
            except Exception as e:
                 logger.error(f"Lỗi khi tính smooth BLEU: {e}")
                 bleu = 0.0 # Đặt giá trị mặc định nếu có lỗi
        else:
             # Sử dụng _bleu (sacrebleu hoặc tương tự) cho các task khác
            try:
                bleu = round(_bleu(gold_fn, output_fn), 2)
            except Exception as e:
                 logger.error(f"Lỗi khi tính BLEU: {e}")
                 bleu = 0.0 # Đặt giá trị mặc định

        # Tính CodeBLEU nếu là task concode (hoặc các task lập trình khác nếu cần)
        # Cần đảm bảo file gold_fn và output_fn chứa code hợp lệ
        if args.task == 'concode':
             try:
                 # Giả sử task concode và ngôn ngữ là java (cần điều chỉnh nếu khác)
                 codebleu = calc_code_bleu([gold_fn], output_fn, 'java')
                 codebleu = round(codebleu * 100, 2) # Nhân 100 cho dễ đọc
             except Exception as e:
                 logger.error(f"Lỗi khi tính CodeBLEU: {e}")
                 codebleu = 0.0
             result['codebleu'] = codebleu

        # Tổng hợp kết quả
        result['em'] = np.mean(dev_accs) * 100
        result['bleu'] = bleu


    logger.info(f"***** Eval results ({criteria}) *****")
    for key in sorted(result.keys()):
        logger.info(f"  {key} = {result[key]:.4f}")

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_lora", action="store_true", help="Whether to use LoRA for fine-tuning")
    args = add_args(parser) # Hàm add_args cần được định nghĩa ở đâu đó (ví dụ: trong configs.py)
    logger.info("Arguments: %s", args)
    t0 = time.time()

    # Thiết lập môi trường distributed và seed
    set_dist(args) # Hàm set_dist cần được định nghĩa (ví dụ: trong configs.py)
    set_seed(args) # Hàm set_seed cần được định nghĩa (ví dụ: trong configs.py)

    # Xây dựng hoặc tải model và tokenizer
    config, model, tokenizer = build_or_load_gen_model(args)

    # Áp dụng LoRA nếu được yêu cầu
    if args.use_lora:
        logger.info("Applying LoRA configuration...")
        # Cấu hình LoRA có thể cần điều chỉnh tùy theo model
        lora_config = LoraConfig(
            r=args.lora_r if hasattr(args, 'lora_r') else 16,  # Rank
            lora_alpha=args.lora_alpha if hasattr(args, 'lora_alpha') else 32, # Alpha scaling
            # target_modules nên được xác định dựa trên kiến trúc model cụ thể
            # Ví dụ cho T5: ["q", "v"] hoặc cho BART: ["q_proj", "v_proj"]
            target_modules=args.lora_target_modules if hasattr(args, 'lora_target_modules') else ["q_proj", "v_proj"],
            lora_dropout=args.lora_dropout if hasattr(args, 'lora_dropout') else 0.05,
            bias="none", # Thường đặt là 'none' hoặc 'all'
            task_type="SEQ_2_SEQ_LM" # Quan trọng cho model seq2seq
        )
        model = get_peft_model(model, lora_config)
        logger.info("LoRA model created:")
        model.print_trainable_parameters() # In ra số lượng tham số cần huấn luyện

    model.to(args.device)

    # Thiết lập DataParallel nếu có nhiều GPU và không dùng DistributedDataParallel
    if args.n_gpu > 1 and args.local_rank == -1: # Chỉ dùng DataParallel nếu không phải DDP
        logger.info(f"Using {args.n_gpu} GPUs with DataParallel.")
        model = torch.nn.DataParallel(model)
    # Lưu ý: Nếu dùng DDP (local_rank != -1), việc wrap model thường được xử lý trong set_dist hoặc tương tự

    # Multiprocessing Pool cho việc xử lý dữ liệu
    # Sử dụng cpu_count() nếu không được chỉ định
    cpu_count = args.cpu_cont if hasattr(args, 'cpu_cont') and args.cpu_cont > 0 else multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)

    # Lấy đường dẫn file dữ liệu
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)

    # File log tổng hợp
    # Tạo thư mục output nếu chưa có
    if args.local_rank in [-1, 0]: # Chỉ rank chính thực hiện tạo file/thư mục
         os.makedirs(args.output_dir, exist_ok=True)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+', encoding='utf-8')


    if args.do_train:
        # TensorBoard Summary Writer (chỉ cho rank chính)
        tb_writer = None
        if args.local_rank in [-1, 0] and args.data_num == -1: # Chỉ log khi train full data
             # Tạo đường dẫn log TensorBoard
             summary_dir_path = args.summary_dir if hasattr(args, 'summary_dir') else os.path.join(args.output_dir, 'runs')
             # summary_fn = '{}/{}'.format(summary_dir_path, '/'.join(args.output_dir.split('/')[1:])) # Cách cũ có thể lỗi nếu output_dir là './'
             summary_fn = os.path.join(summary_dir_path, os.path.basename(args.output_dir))
             os.makedirs(os.path.dirname(summary_fn), exist_ok=True)
             tb_writer = SummaryWriter(summary_fn)
             logger.info(f"TensorBoard logs will be written to: {summary_fn}")


        # Tải dữ liệu huấn luyện
        logger.info(f"Loading and caching train data from {args.train_filename}")
        train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train')
        logger.info(f"Loaded {len(train_examples)} training examples")

        # Sampler và DataLoader cho training
        if args.local_rank == -1: # Non-distributed
            train_sampler = RandomSampler(train_data)
        else: # Distributed
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True) # Có thể điều chỉnh num_workers

        # Thiết lập optimizer và scheduler
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        # Tính toán tổng số bước huấn luyện
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        num_warmup_steps = int(t_total * args.warmup_ratio) if args.warmup_steps == 0 else args.warmup_steps
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=t_total)

        # Tải checkpoint nếu có
        # Đảm bảo args.start_epoch đã được gán giá trị mặc định (thường là 0) nếu không load được ckpt
        args.start_epoch = args.start_epoch if hasattr(args, 'start_epoch') else 0
        model, optimizer, scheduler, start_epoch, last_loss = load_latest_checkpoint(model, optimizer, scheduler, args)
        args.start_epoch = start_epoch # Cập nhật epoch bắt đầu từ checkpoint

        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {train_example_num}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per GPU = {args.train_batch_size // args.n_gpu if args.n_gpu > 0 else args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {t_total}")
        logger.info(f"  Starting training from epoch {args.start_epoch}")

        dev_dataset = {} # Cache để lưu dữ liệu dev đã load
        global_step = 0 # Sẽ được cập nhật nếu load checkpoint
        # Cập nhật global_step dựa trên epoch đã load và số bước mỗi epoch
        # Ước tính global_step đã qua nếu load từ checkpoint
        steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        global_step = args.start_epoch * steps_per_epoch

        best_bleu_em, best_ppl = -1.0, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0

        # Vòng lặp huấn luyện chính
        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            logger.info(f"--- Starting Epoch {cur_epoch} ---")
            # Đặt epoch cho sampler (quan trọng cho distributed training)
            if args.local_rank != -1:
                train_sampler.set_epoch(cur_epoch)

            bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {cur_epoch} Training",
                       disable=args.local_rank not in [-1, 0]) # Chỉ hiển thị progress bar ở rank chính
            nb_tr_steps = 0
            tr_loss = 0.0 # Tổng loss trong epoch này
            model.train() # Đặt model ở chế độ train

            try:
                for step, batch in enumerate(bar):
                    batch = tuple(t.to(args.device) for t in batch)
                    source_ids, target_ids = batch
                    source_mask = source_ids.ne(tokenizer.pad_token_id)
                    target_mask = target_ids.ne(tokenizer.pad_token_id)

                    if args.model_type == 'roberta':
                        outputs = model(source_ids=source_ids, source_mask=source_mask,
                                        target_ids=target_ids, target_mask=target_mask)
                        loss = outputs[0] # Giả sử loss là phần tử đầu tiên
                    else:
                        outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                        labels=target_ids, decoder_attention_mask=target_mask)
                        loss = outputs.loss

                    # Xử lý loss với DataParallel / DDP / Accumulation
                    if args.n_gpu > 1:
                        loss = loss.mean()  # Trung bình loss trên các GPU (cần thiết cho DataParallel)
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    loss.backward() # Tính gradients

                    tr_loss += loss.item() * args.gradient_accumulation_steps # Cộng dồn loss thực tế trước khi chia

                    # Cập nhật thanh progress bar (chỉ rank chính)
                    if args.local_rank in [-1, 0]:
                        current_loss = tr_loss / (step + 1) # Loss trung bình tính đến bước hiện tại
                        bar.set_postfix(loss=f'{current_loss:.3f}', lr=f'{scheduler.get_last_lr()[0]:.1e}')

                    # Gradient accumulation step
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        # Clip gradient norm
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                        optimizer.step() # Cập nhật weights
                        scheduler.step() # Cập nhật learning rate
                        optimizer.zero_grad() # Reset gradients
                        global_step += 1
                        nb_tr_steps += 1 # Số lần optimizer step thực sự

                        # Log loss định kỳ (chỉ rank chính)
                        if args.local_rank in [-1, 0] and args.log_steps > 0 and global_step % args.log_steps == 0:
                             avg_loss = tr_loss / (step + 1) # Loss trung bình từ đầu epoch
                             if tb_writer:
                                 tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                                 tb_writer.add_scalar('train_loss', avg_loss, global_step)
                             logger.info(f"Step: {global_step}, Epoch: {cur_epoch}, Avg Train Loss: {avg_loss:.4f}")

                    # Kiểm tra điều kiện dừng dựa trên max_steps
                    if args.max_steps > 0 and global_step >= args.max_steps:
                         logger.info(f"Reached max_steps ({args.max_steps}). Stopping training.")
                         break # Thoát khỏi vòng lặp batch

                # Kết thúc vòng lặp batch
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break # Thoát khỏi vòng lặp epoch

                # --- Đánh giá cuối mỗi epoch (chỉ thực hiện bởi rank chính) ---
                if args.do_eval and args.local_rank in [-1, 0]:
                    logger.info(f"--- Evaluating Epoch {cur_epoch} ---")
                    # 1. Đánh giá PPL
                    if 'dev_loss' not in dev_dataset:
                        logger.info(f"Loading and caching dev data from {args.dev_filename}")
                        eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev')
                        dev_dataset['dev_loss'] = eval_examples, eval_data
                        logger.info(f"Loaded {len(eval_examples)} dev examples for PPL eval.")
                    else:
                        eval_examples, eval_data = dev_dataset['dev_loss']

                    eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                    result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                    logger.info("***** PPL Evaluation Results *****")
                    for key, value in result.items():
                        logger.info(f"  {key} = {value}")
                    if tb_writer:
                        tb_writer.add_scalar('dev_ppl', eval_ppl, global_step) # Log PPL theo global_step

                    # Lưu checkpoint cuối cùng nếu được yêu cầu
                    if args.save_last_checkpoints:
                        last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                        os.makedirs(last_output_dir, exist_ok=True)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        # Lưu cả model PEFT nếu dùng LoRA
                        output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                        if args.use_lora:
                             model_to_save.save_pretrained(last_output_dir) # Lưu adapter config và weights
                             logger.info(f"Saved LoRA adapter model to {last_output_dir}")
                        else:
                             torch.save(model_to_save.state_dict(), output_model_file)
                             logger.info(f"Saved full model state dict to {output_model_file}")


                    # Kiểm tra và lưu model tốt nhất dựa trên PPL
                    if eval_ppl < best_ppl:
                        not_loss_dec_cnt = 0 # Reset counter
                        logger.info(f"  New best PPL: {eval_ppl:.4f} (previous: {best_ppl:.4f})")
                        best_ppl = eval_ppl
                        fa.write(f"[{cur_epoch}/{global_step}] Best PPL updated to {best_ppl:.4f}\n")

                        # Lưu model checkpoint best-ppl
                        output_dir_ppl = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                        os.makedirs(output_dir_ppl, exist_ok=True)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        if args.always_save_model or args.save_best_ppl: # Thêm cờ save_best_ppl nếu muốn
                            if args.use_lora:
                                model_to_save.save_pretrained(output_dir_ppl)
                                logger.info(f"Saved best PPL LoRA adapter model to {output_dir_ppl}")
                            else:
                                output_model_file = os.path.join(output_dir_ppl, "pytorch_model.bin")
                                torch.save(model_to_save.state_dict(), output_model_file)
                                logger.info(f"Saved best PPL full model state dict to {output_model_file}")
                    else:
                        not_loss_dec_cnt += 1
                        logger.info(f"PPL ({eval_ppl:.4f}) did not improve from best ({best_ppl:.4f}). Count: {not_loss_dec_cnt}/{args.patience}")

                    # Giải phóng bộ nhớ cache CUDA
                    torch.cuda.empty_cache()

                    # 2. Đánh giá BLEU/EM/CodeBLEU (nếu được yêu cầu)
                    if args.do_eval_bleu:
                         # Load dữ liệu dev chỉ với source, có thể sample nếu cần
                        if 'dev_bleu' not in dev_dataset:
                             logger.info(f"Loading and caching dev data from {args.dev_filename} (source only) for BLEU eval")
                             # is_sample=True nếu muốn đánh giá trên subset nhỏ hơn cho nhanh
                             eval_examples_bleu, eval_data_bleu = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                          only_src=True, is_sample=False)
                             dev_dataset['dev_bleu'] = eval_examples_bleu, eval_data_bleu
                             logger.info(f"Loaded {len(eval_examples_bleu)} dev examples for BLEU eval.")
                        else:
                             eval_examples_bleu, eval_data_bleu = dev_dataset['dev_bleu']

                        # Thực hiện đánh giá BLEU/EM
                        result_bleu = eval_bleu_epoch(args, eval_data_bleu, eval_examples_bleu, model, tokenizer, 'dev', f'e{cur_epoch}_gs{global_step}')
                        dev_bleu = result_bleu.get('bleu', 0.0)
                        dev_em = result_bleu.get('em', 0.0)
                        dev_codebleu = result_bleu.get('codebleu', 0.0) # Lấy codebleu nếu có

                        # Metric chính để so sánh (tùy thuộc vào task)
                        if args.task in ['summarize']:
                            dev_bleu_em = dev_bleu # Chỉ dùng BLEU cho tóm tắt
                        elif args.task in ['defect']:
                            dev_bleu_em = dev_em # Chỉ dùng EM/Accuracy cho defect
                        else: # Các task khác (translate, concode) dùng tổng BLEU + EM
                            dev_bleu_em = dev_bleu + dev_em
                        logger.info(f"***** BLEU/EM Evaluation Results (Epoch {cur_epoch}) *****")
                        logger.info(f"  BLEU = {dev_bleu:.4f}")
                        logger.info(f"  EM = {dev_em:.4f}")
                        if 'codebleu' in result_bleu:
                             logger.info(f"  CodeBLEU = {dev_codebleu:.4f}")
                        logger.info(f"  Combined Metric (BLEU+EM or task specific) = {dev_bleu_em:.4f}")

                        if tb_writer:
                             tb_writer.add_scalar('dev_bleu', dev_bleu, global_step)
                             tb_writer.add_scalar('dev_em', dev_em, global_step)
                             tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, global_step)
                             if 'codebleu' in result_bleu:
                                 tb_writer.add_scalar('dev_codebleu', dev_codebleu, global_step)


                        # Kiểm tra và lưu model tốt nhất dựa trên BLEU/EM
                        if dev_bleu_em > best_bleu_em:
                             not_bleu_em_inc_cnt = 0 # Reset counter
                             logger.info(f"  New best BLEU/EM: {dev_bleu_em:.4f} (previous: {best_bleu_em:.4f})")
                             best_bleu_em = dev_bleu_em
                             fa.write(f"[{cur_epoch}/{global_step}] Best BLEU/EM updated to {best_bleu_em:.4f} (BLEU: {dev_bleu:.2f}, EM: {dev_em:.2f})\n")

                             # Lưu model checkpoint best-bleu
                             output_dir_bleu = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                             os.makedirs(output_dir_bleu, exist_ok=True)
                             model_to_save = model.module if hasattr(model, 'module') else model
                             if args.always_save_model or args.save_best_bleu: # Thêm cờ save_best_bleu
                                 if args.use_lora:
                                     model_to_save.save_pretrained(output_dir_bleu)
                                     logger.info(f"Saved best BLEU/EM LoRA adapter model to {output_dir_bleu}")
                                 else:
                                     output_model_file = os.path.join(output_dir_bleu, "pytorch_model.bin")
                                     torch.save(model_to_save.state_dict(), output_model_file)
                                     logger.info(f"Saved best BLEU/EM full model state dict to {output_model_file}")
                        else:
                             not_bleu_em_inc_cnt += 1
                             logger.info(f"BLEU/EM ({dev_bleu_em:.4f}) did not improve from best ({best_bleu_em:.4f}). Count: {not_bleu_em_inc_cnt}/{args.patience}")

                    # Kiểm tra điều kiện Early Stopping
                    if args.patience > 0 and all([x >= args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                        early_stop_str = f"[{cur_epoch}] Early stopping triggered: PPL count {not_loss_dec_cnt}, BLEU/EM count {not_bleu_em_inc_cnt} >= patience {args.patience}"
                        logger.info(early_stop_str)
                        fa.write(early_stop_str + "\n")
                        break # Thoát khỏi vòng lặp epoch

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Saving current state and exiting...")
                # Lưu checkpoint hiện tại trước khi thoát (chỉ rank chính)
                if args.local_rank in [-1, 0]:
                    avg_epoch_loss = tr_loss / nb_tr_steps if nb_tr_steps > 0 else 0.0
                    save_checkpoint(model, optimizer, scheduler, cur_epoch, avg_epoch_loss, args) # Lưu epoch hiện tại (chưa hoàn thành)
                torch.cuda.empty_cache()
                break # Thoát khỏi vòng lặp epoch

            # --- Lưu checkpoint cuối mỗi epoch (chỉ rank chính) ---
            if args.local_rank in [-1, 0]:
                avg_epoch_loss = tr_loss / nb_tr_steps if nb_tr_steps > 0 else 0.0
                # Lưu với epoch + 1 vì epoch cur_epoch đã hoàn thành
                save_checkpoint(model, optimizer, scheduler, cur_epoch + 1, avg_epoch_loss, args)

            # Giải phóng bộ nhớ cache CUDA cuối mỗi epoch
            logger.info("***** CUDA.empty_cache() at end of epoch *****")
            torch.cuda.empty_cache()


        # Kết thúc training
        if args.local_rank in [-1, 0] and tb_writer:
            tb_writer.close()
        logger.info(f"Finish training. Total time: {get_elapse_time(t0)}")
        fa.write(f"Finish training. Total time: {get_elapse_time(t0)}\n")


    # --- Testing Phase ---
    if args.do_test and args.local_rank in [-1, 0]: # Chỉ rank chính thực hiện test
        logger.info("***** Running Testing *****")
        logger.info(f"  Batch size = {args.eval_batch_size}")

        # Các checkpoint cần test (ví dụ: best-bleu, best-ppl, last)
        checkpoints_to_test = []
        if os.path.exists(os.path.join(args.output_dir, 'checkpoint-best-bleu')):
            checkpoints_to_test.append('best-bleu')
        if os.path.exists(os.path.join(args.output_dir, 'checkpoint-best-ppl')):
             checkpoints_to_test.append('best-ppl')
        if os.path.exists(os.path.join(args.output_dir, 'checkpoint-last')):
            checkpoints_to_test.append('last')

        if not checkpoints_to_test:
             logger.warning("No checkpoints found to test in the output directory.")

        # Load dữ liệu test (chỉ cần source)
        logger.info(f"Loading and caching test data from {args.test_filename}")
        test_examples, test_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test',
                                                           only_src=True, is_sample=False)
        logger.info(f"Loaded {len(test_examples)} test examples.")

        for criteria in checkpoints_to_test:
            checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-{criteria}')
            logger.info(f"--- Testing checkpoint: {criteria} from {checkpoint_dir} ---")

            # Tải model từ checkpoint
            # Cần tải lại model gốc và áp dụng adapter nếu dùng LoRA
            if args.use_lora:
                 # Tải lại model base
                 _, model_base, _ = build_or_load_gen_model(args) # Lấy model gốc
                 model = get_peft_model(model_base, LoraConfig.from_pretrained(checkpoint_dir)) # Tải adapter
                 model.to(args.device)
                 logger.info(f"Reloaded LoRA adapter from {checkpoint_dir}")
            else:
                 # Tải state_dict cho model thường
                 file = os.path.join(checkpoint_dir, 'pytorch_model.bin')
                 if os.path.exists(file):
                      model.load_state_dict(torch.load(file, map_location=args.device))
                      logger.info(f"Reloaded full model state dict from {file}")
                 else:
                      logger.warning(f"Model file {file} not found for criteria {criteria}. Skipping test.")
                      continue # Bỏ qua checkpoint này nếu không tìm thấy file

            # Nếu model được wrap bằng DataParallel trong training, wrap lại khi test
            if args.n_gpu > 1 and args.local_rank == -1 and not isinstance(model, torch.nn.DataParallel):
                 model = torch.nn.DataParallel(model)


            # Thực hiện đánh giá BLEU/EM trên tập test
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

            # Ghi kết quả vào file riêng nếu có chỉ định
            if args.res_fn:
                 # Kiểm tra nếu res_fn chỉ là tên file, ghép với output_dir
                 res_file_path = args.res_fn if os.path.isabs(args.res_fn) else os.path.join(args.output_dir, args.res_fn)
                 os.makedirs(os.path.dirname(res_file_path), exist_ok=True) # Đảm bảo thư mục tồn tại
                 with open(res_file_path, 'a+', encoding='utf-8') as f_res:
                     f_res.write(f'[Time: {get_elapse_time(t0)}] Checkpoint: {checkpoint_dir}\n')
                     f_res.write(result_str)

            # Giải phóng bộ nhớ
            torch.cuda.empty_cache()

    # Kết thúc chương trình
    logger.info(f"Finish All. Total time: {get_elapse_time(t0)}")
    fa.write(f"Finish All. Total time: {get_elapse_time(t0)}\n")
    fa.close()
    pool.close() # Đóng multiprocessing pool
    pool.join()

if __name__ == "__main__":
    main()
