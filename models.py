def build_or_load_gen_model(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)

    if args.use_qlora:
        # Tải mô hình với quantization 4-bit và áp dụng LoRA
        model = model_class.from_pretrained(
            args.model_name_or_path,
            load_in_4bit=True,  # Quantize xuống 4-bit
            device_map="auto",  # Tự động phân bổ trên GPU
            torch_dtype=torch.float16  # Dùng FP16 để tính toán
        )
        lora_config = LoraConfig(
            r=8,  # Rank của LoRA
            lora_alpha=16,  # Hệ số scaling
            target_modules=["q", "v"],  # Áp dụng LoRA cho query và value
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, lora_config)
        logger.info("Applied QLoRA: trainable params: %s", model.print_trainable_parameters())
    else:
        if args.model_type == 'roberta':
            encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
            decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
            decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
            model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                            beam_size=args.beam_size, max_length=args.max_target_length,
                            sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
        else:
            model = model_class.from_pretrained(args.model_name_or_path)

        logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
    else:
        logger.info("Do not Load Models.")

    return config, model, tokenizer
