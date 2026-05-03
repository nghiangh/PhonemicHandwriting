import torch
from torch import nn
import math

from vocabs.viphoner import ViPhoNER

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: [B, L, D]
        seq_len = x.size(1)
        return self.pe[:seq_len, :].unsqueeze(0)  # [1, L, D]

class PhonemicTransformer(nn.Module):
    def __init__(self, config, vocab: ViPhoNER):
        super().__init__()

        self.pad_idx = vocab.pad_idx
        self.bos_idx = vocab.bos_idx
        self.eos_idx = vocab.eos_idx

        self.d_model = config.d_model
        self.device = config.device
        self.config = config
        self.vocab = vocab

        self.MAX_LENGTH = vocab.max_sentence_length + 2

        self.img_feat_proj = nn.Linear(
            in_features=config.d_feat,
            out_features=config.d_model
        )

        # Embedding
        self.embedder = nn.Embedding(vocab.size(), config.d_model, padding_idx=self.pad_idx)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_len)
        
        # Dropout for embeddings
        self.dropout = nn.Dropout(config.dropout)

        # Encoder
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.dim_ffw,
            dropout=config.dropout,
            batch_first=True
        ), num_layers=config.n_layers)

        # Decoder
        self.phonemic_fc = nn.Linear(
            in_features=config.d_model*3,
            out_features=config.d_model
        )
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.dim_ffw,
            dropout=config.dropout,
            batch_first=True
        ), num_layers=config.n_layers)

        # Output projection
        self.output_layer = nn.Linear(config.d_model, vocab.size()*3)
        # self.output_v = nn.Linear(config.d_model, vocab_size)
        # self.output_t = nn.Linear(config.d_model, vocab_size)

        self.loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def make_src_padding_mask(self, src: torch.Tensor, value=0):
        if src.dim() == 3:
            src = src.mean(dim=-1)
        return (src == value)  # [B, L]

    def make_tgt_mask(self, tgt: torch.Tensor, value=0):
        if tgt.dim() == 3:
            tgt = tgt.float().mean(dim=-1)
        _, T = tgt.size()

        # Padding mask
        padding_mask = (tgt == value)  # [B, T]

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=tgt.device), diagonal=1).bool()

        return padding_mask, causal_mask

    def forward(self, src, tgt):
        # Use tgt[:-1] as input, predict tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Masks for input sequence
        src_key_padding = self.make_src_padding_mask(src, value=0)
        tgt_padding, tgt_causal = self.make_tgt_mask(tgt_input, value=self.pad_idx)

        # Embedding + Positional encoding
        src_emb = self.img_feat_proj(src) * math.sqrt(self.d_model)
        src_pos = self.pos_encoding(src_emb)
        enc_input = self.dropout(src_emb + src_pos)
        
        # phonemic Embedding + Positional encoding
        tgt_emb = self.embedder(tgt_input) * math.sqrt(self.d_model) # (B, L, 3, D)
        B, L, _, _ = tgt_emb.shape
        tgt_emb = tgt_emb.reshape(B, L, -1) # (B, L, 3*D)
        tgt_emb = self.phonemic_fc(tgt_emb) # (B, L, D)
        tgt_pos = self.pos_encoding(tgt_emb)
        dec_input = self.dropout(tgt_emb + tgt_pos)

        # Encoder
        memory = self.encoder(
            enc_input,
            src_key_padding_mask=src_key_padding
        )  # [B, L, D]

        # Decoder
        out = self.decoder(
            dec_input,
            memory,
            tgt_mask=tgt_causal,
            tgt_key_padding_mask=tgt_padding,
            memory_key_padding_mask=src_key_padding
        )

        # logits_i = self.output_i(out)  # [B, T-1, V]
        # logits_v = self.output_v(out)  # [B, T-1, V]
        # logits_t = self.output_t(out)  # [B, T-1, V]

        # Compute loss on shifted target
        # loss_i = self.loss(logits_i.reshape(-1, logits_i.size(-1)), tgt_output.reshape(-1))
        # loss_v = self.loss(logits_v.reshape(-1, logits_v.size(-1)), tgt_output.reshape(-1))
        # loss_t = self.loss(logits_t.reshape(-1, logits_t.size(-1)), tgt_output.reshape(-1))
        # loss = loss_i + loss_v + loss_t

        logits = self.output_layer(out) # (B, L, 3*V)
        logits = logits.reshape(B, L, 3, -1)
        loss = self.loss(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        return loss

    def predict(self, src: torch.Tensor):
        self.eval()

        src_key_padding = self.make_src_padding_mask(src)
    
        # Embedding + Positional encoding
        B = src.size(0)
        src_emb = self.img_feat_proj(src) * math.sqrt(self.d_model)
        src_pos = self.pos_encoding(src_emb)
        enc_input = src_emb + src_pos
        
        # Use no_grad for inference
        with torch.no_grad():
            memory = self.encoder(enc_input, src_key_padding_mask=src_key_padding)
        
            # Start with BOS
            tgt_seq = torch.full((B, 1, 3), self.bos_idx, device=src.device, dtype=torch.long)
            finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        
            for _ in range(self.MAX_LENGTH):
                tgt_padding, tgt_causal = self.make_tgt_mask(tgt_seq)
        
                # Embedding + Positional encoding
                tgt_emb = self.embedder(tgt_seq) * math.sqrt(self.d_model) # (B, L, 3, D)
                B, L, _, _ = tgt_emb.shape
                tgt_emb = tgt_emb.reshape(B, L, -1) # (B, L, 3*D)
                tgt_emb = self.phonemic_fc(tgt_emb) # (B, L, D)
                tgt_pos = self.pos_encoding(tgt_emb)
                dec_input = self.dropout(tgt_emb + tgt_pos)
                dec_out = self.decoder(
                    dec_input,
                    memory,
                    tgt_mask=tgt_causal,
                    tgt_key_padding_mask=tgt_padding,
                    memory_key_padding_mask=src_key_padding
                )
        
                # Get logits for last position
                B = dec_out.shape[0]
                V = len(self.vocab)
                logits = self.output_layer(dec_out[:, -1, :])  # [B, 3*V]
                logits = logits.reshape(B, 3, V)
                next_token = logits.argmax(dim=-1, keepdim=True)
        
                # Append next token
                tgt_seq = torch.cat([tgt_seq, next_token], dim=1)
        
                # Check if finished
                finished |= (next_token.squeeze(1) == self.eos_idx)
        
                if finished.all():
                    break
        
        return tgt_seq[:, 1:]  # Remove BOS