from tqdm.auto import tqdm
import torch
import argparse
import pandas as pd
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor
) 
from dataset import ASRDataset


def train_one_epoch(model, train_loader, optimizer, scheduler, device='cuda:0'):
    model.train()
    pbar = tqdm(train_loader, total=len(train_loader))
    avg_loss = 0
    for data in pbar:
        data = {k: v.to(device) for k, v in data.items()}
        loss = model(**data).loss
        loss_itm = loss.item()
        avg_loss += loss_itm
        pbar.set_description(f"loss: {loss_itm:.4f}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    return avg_loss / len(train_loader)


@torch.no_grad()
def valid_one_epoch(model, valid_loader, device='cuda:0'):
    model.eval()
    pbar = tqdm(valid_loader, total=len(valid_loader))
    avg_loss = 0
    for data in pbar:
        data = {k: v.to(device) for k, v in data.items()}
        loss = model(**data).loss
        loss_itm = loss.item()

        avg_loss += loss_itm
        pbar.set_description(f"val_loss: {loss_itm:.4f}")

    return avg_loss / len(valid_loader)


def ctc_data_collator(batch):
    """
    Custom data collator function to dynamically pad the data
    """
    input_features = [{"input_values": sample["audio"]} for sample in batch]
    label_features = [{"input_ids": sample["label"]} for sample in batch]
    batch = processor.pad(
        input_features,
        padding=True,
        return_tensors="pt",
    )
    with processor.as_target_processor():
        labels_batch = processor.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        )
        
    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
    batch["labels"] = labels
    return batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--wav_dir', type=str, default="/home/hyunseoki_rtx3090/ssd1/01_dataset/aihub/KsponSpeech/wav")
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    df = pd.read_csv('./data/metadata.csv') ## 620000
    train_df = df[:500000]
    valid_df = df[500000:]
    valid_df.reset_index(inplace=True)

    tokenizer = Wav2Vec2CTCTokenizer(
        "./vocab/vocab.json", 
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="__"
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, 
        sampling_rate=16000, 
        padding_value=0.0, 
        do_normalize=True, 
        return_attention_mask=False
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(
        'facebook/wav2vec2-base',
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size = len(tokenizer),
    )
    model.to(args.device)
    model.freeze_feature_encoder()

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        # lr=3e-4, 
        lr=1e-4, 
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6,
    )

    train_dataset = ASRDataset(wav_dir=args.wav_dir, df=train_df, processor=processor)
    valid_dataset = ASRDataset(wav_dir=args.wav_dir, df=valid_df, processor=processor)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        collate_fn=ctc_data_collator, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 1 else False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        collate_fn=ctc_data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 1 else False,
    )

    # Train the model
    best_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"{'='*40} Epoch: {epoch+1} / {args.epochs} {'='*40}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
        valid_loss = valid_one_epoch(model, valid_loader)
        print(f"train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}")
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f"wav2vec2_baseline.pt")
            print(f"Saved the best model so far with val_loss: {valid_loss:.4f}")