from layer import Linear, Embedding, Conv1d, Conv2d, Conv3d
# Import LoRA-enabled layers
torch_lora = True  # flag for clarity
from layer import Linear as LoRALinear, Embedding as LoRAEmbedding
import torch
import torch.nn as nn
import numpy as np
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer)
import logging

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, AutoTokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def inject_lora(module, r, alpha, dropout, merge_weights=True):
    """
    Recursively replace nn.Linear and nn.Embedding with LoRA-enabled versions.
    """
    for name, child in list(module.named_children()):
        # First recurse into children
        inject_lora(child, r, alpha, dropout, merge_weights)
        # Replace Linear layers
        if isinstance(child, nn.Linear):
            lora_layer = LoRALinear(
                in_features=child.in_features,
                out_features=child.out_features,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                merge_weights=merge_weights,
                bias=(child.bias is not None)
            )
            setattr(module, name, lora_layer)
        # Replace Embedding layers
        elif isinstance(child, nn.Embedding):
            lora_layer = LoRAEmbedding(
                num_embeddings=child.num_embeddings,
                embedding_dim=child.embedding_dim,
                r=r,
                lora_alpha=alpha,
                merge_weights=merge_weights
            )
            setattr(module, name, lora_layer)


def build_or_load_gen_model(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    
    if args.model_type == 'roberta':
        encoder = model_class.from_pretrained(
            args.model_name_or_path, config=config)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=6)
        model = Seq2Seq(
            encoder=encoder,
            decoder=decoder,
            config=config,
            beam_size=args.beam_size,
            max_length=args.max_target_length,
            sos_id=tokenizer.cls_token_id,
            eos_id=tokenizer.sep_token_id
        )
    else:
        model = model_class.from_pretrained(args.model_name_or_path)

    logger.info(
        "Finish loading model [%s] from %s",
        get_model_size(model),
        args.model_name_or_path
    )

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
    else:
        logger.info("Do not Load Models.")

    # Inject LoRA layers throughout the model backbone
    inject_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    logger.info(
        "Injected LoRA: r=%d, alpha=%d, dropout=%.2f",
        args.lora_r, args.lora_alpha, args.lora_dropout
    )

    return config, model, tokenizer


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(
            config.hidden_size * 2, config.hidden_size
        )
        self.out_proj = Linear(config.hidden_size, 2)

    def forward(self, x, **kwargs):
        x = x.reshape(-1, x.size(-1) * 2)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class CloneModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CloneModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args

    def get_model_vec(self, source_ids):
        attention_mask = source_ids.ne(
            self.tokenizer.pad_token_id)
        outputs = self.encoder(
            input_ids=source_ids,
            attention_mask=attention_mask,
            labels=source_ids,
            decoder_attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError(
                "All examples must have the same number of <eos> tokens.")

        vec = hidden_states[
            eos_mask, :
        ].view(
            hidden_states.size(0), -1,
            hidden_states.size(-1)
        )[:, -1, :]
        return vec

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(
            -1, self.args.max_source_length
        )

        vec = self.get_model_vec(source_ids)

        logits = self.classifier(vec)
        prob = nn.functional.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


class DefectModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(DefectModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = Linear(config.hidden_size, 2)
        self.args = args

    def get_model_vec(self, source_ids):
        attention_mask = source_ids.ne(
            self.tokenizer.pad_token_id
        )
        outputs = self.encoder(
            input_ids=source_ids,
            attention_mask=attention_mask,
            labels=source_ids,
            decoder_attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError(
                "All examples must have the same number of <eos> tokens.")

        vec = hidden_states[
            eos_mask, :
        ].view(
            hidden_states.size(0), -1,
            hidden_states.size(-1)
        )[:, -1, :]
        return vec

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(
            -1, self.args.max_source_length
        )

        vec = self.get_model_vec(source_ids)

        logits = self.classifier(vec)
        prob = nn.functional.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


# ... rest of Seq2Seq unchanged ...



# https://github.com/microsoft/CodeBERT/blob/master/CodeBERT/code2nl/model.py
class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = layer.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = layer.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head, self.encoder.layer.embeddings.word_embeddings)

    def forward(self, source_ids=None, source_mask=None, target_ids=None, target_mask=None, args=None):
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1, 0, 2]).contiguous()
        if target_ids is not None:
            attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            tgt_embeddings = self.encoder.layer.embeddings(target_ids).permute([1, 0, 2]).contiguous()
            out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
                               memory_key_padding_mask=~source_mask)
            hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])
            outputs = loss, loss * active_loss.sum(), active_loss.sum()
            return outputs
        else:
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.layer.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                       memory_key_padding_mask=~context_mask)
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds
