from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from embed import DataEmbedding
import torch.nn.functional as F
from einops import rearrange

"""Two contrastive encoders"""
class TFC(nn.Module):
    def __init__(self, configs):
        super(TFC, self).__init__()

        encoder_layers_t = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned, nhead=configs.transformer_nhead, )
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, configs.transformer_num_layers)

        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned*configs.input_channels, configs.embedding_len*2),
            nn.BatchNorm1d(configs.embedding_len*2),
            nn.ReLU(),
            nn.Linear(configs.embedding_len*2, configs.embedding_len)
        )

        encoder_layers_f = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned,nhead=configs.transformer_nhead,)
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, configs.transformer_num_layers)

        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned*configs.input_channels, configs.embedding_len*2),
            nn.BatchNorm1d(configs.embedding_len*2),
            nn.ReLU(),
            nn.Linear(configs.embedding_len*2, configs.embedding_len)
        )


    def forward(self, x_in_t, x_in_f):
        # x_in_t: (batch_size, feat_dim, max_seq_len)
        # x_in_f: (batch_size, feat_dim, max_seq_len)
        """Use Transformer"""
        x = self.transformer_encoder_t(x_in_t)  # (batch_size, feat_dim, max_seq_len)
        h_time = x.reshape(x.shape[0], -1)      # (batch_size, feat_dim * max_seq_len)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)       # (batch_size, embedding)

        """Frequency-based contrastive encoder"""
        f = self.transformer_encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq


"""Downstream classifier only used in finetuning"""
class target_classifier(nn.Module):
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2*128, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred


"""GPT based transformers"""
class gpt4ts(nn.Module):
    
    def __init__(self, configs):
        super(gpt4ts, self).__init__()
        self.pred_len = 0
        self.seq_len = configs.max_seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.gpt_layers = configs.gpt_layers
        self.feat_dim = configs.feat_dim
        self.d_model = configs.d_model

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1
        self.patch_num += 1

        # time domain transformer 
        self.padding_patch_layer_t = nn.ReplicationPad1d((0, self.stride)) 
        self.enc_embedding_t = DataEmbedding(self.feat_dim * self.patch_size, configs.d_model, configs.dropout)
        self.gpt2_t = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        # gpt2 has 12 layers 
        self.gpt2_t.h = self.gpt2_t.h[:self.gpt_layers]
        for i, (name, param) in enumerate(self.gpt2_t.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.act = F.gelu
        self.dropout_t = nn.Dropout(0.1)
        self.ln_proj_t = nn.LayerNorm(configs.d_model * self.patch_num)
        self.out_layer_t = nn.Linear(configs.d_model * self.patch_num, configs.embedding_len)

        # frequency domain transfomer 
        self.padding_patch_layer_f = nn.ReplicationPad1d((0, self.stride)) 
        self.enc_embedding_f = DataEmbedding(self.feat_dim * self.patch_size, configs.d_model, configs.dropout)
        self.gpt2_f = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2_f.h = self.gpt2_f.h[:self.gpt_layers]
        for i, (name, param) in enumerate(self.gpt2_f.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.act = F.gelu
        self.dropout_f = nn.Dropout(0.1)
        self.ln_proj_f = nn.LayerNorm(configs.d_model * self.patch_num)
        self.out_layer_f = nn.Linear(configs.d_model * self.patch_num, configs.embedding_len)
        
    def forward(self, x_in_t, x_in_f):
        # time domain 
        B_t, M_t, L_t = x_in_t.shape # B: batch_size, M: feat_dim, L: max_seq_len 
        input_t = self.padding_patch_layer_t(x_in_t)
        input_t = input_t.unfold(dimension=-1, size=self.patch_size, step=self.stride) # (B, M, new_len, patch_size)
        input_t = rearrange(input_t, 'b m n p -> b n (p m)') # (B, new_len, new_fea)
        outputs_t = self.enc_embedding_t(input_t, None) # (B, new_len, d_model)
        outputs_t = self.gpt2_t(inputs_embeds=outputs_t).last_hidden_state  # (B, new_len, d_model)
        h_time = outputs_t.reshape(outputs_t.shape[0], -1) # (B, new_len*d_model)
        outputs_t = self.act(outputs_t).reshape(B_t, -1) # (B, new_len*d_model)
        outputs_t = self.ln_proj_t(outputs_t) # (B, new_len*d_model)
        z_time = self.out_layer_t(outputs_t) # (B, embedding_len)
        
        # freq domain 
        B_f, M_f, L_f = x_in_f.shape # B: batch_size, M: feat_dim, L: max_seq_len 
        input_f = self.padding_patch_layer_f(x_in_f)
        input_f = input_f.unfold(dimension=-1, size=self.patch_size, step=self.stride) # (B, M, new_len, patch_size)
        input_f = rearrange(input_f, 'b m n p -> b n (p m)') # (B, new_len, new_fea)
        outputs_f = self.enc_embedding_f(input_f, None) # (B, new_len, d_model)
        outputs_f = self.gpt2_f(inputs_embeds=outputs_f).last_hidden_state  # (B, new_len, d_model)
        h_freq = outputs_f.reshape(outputs_f.shape[0], -1) # (B, new_len*d_model)
        outputs_f = self.act(outputs_f).reshape(B_f, -1) # (B, new_len*d_model)
        outputs_f = self.ln_proj_f(outputs_f) # (B, new_len*d_model)
        z_freq = self.out_layer_f(outputs_f) # (B, embedding_len)
        
        return h_time, z_time, h_freq, z_freq 