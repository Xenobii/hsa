import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as func

from model.utils import plot_spec

class HSA(nn.Module):
    def __init__(
            self,
            F: int = 256,
            P: int = 32,
            T: int = 128,
            D: int = 256,
            K: int = 8,
            target_sr: int = 16000,
            fft_bins: int = 2048,
            window_length: int = 2048,
            hop_sample: int = 256,
            pad_mode: str = "constant",
            log_offset: float = 1e-8,
    ) -> None:
        super().__init__()
        self.F = F

        # --- resampler ---
        self.target_sr = target_sr
        self.default_sr = 44100
        self.resampler = torchaudio.transforms.Resample(self.default_sr, target_sr)

        # --- log mel spec ---
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=fft_bins,
            win_length=window_length,
            hop_length=hop_sample,
            pad_mode=pad_mode,
            n_mels=F,
            norm="slaney"
        )
        self.log_offset = log_offset
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

        # --- windowing --- 
        self.P = P
        self.T = T
        self.M = 2*P + 1

        # --- margin cnn ---
        self.cnn_channel = 4
        self.cnn_kernel = 5
        self.cnn_dim = self.cnn_channel * (self.M - (self.cnn_kernel - 1))
        self.conv = nn.Conv2d(1, self.cnn_channel, kernel_size=(1, self.cnn_kernel))

        # --- frequency ---
        self.D = D
        self.tok_emb_freq = nn.Linear(self.cnn_dim, D)

        # --- encoder ---
        self.encoder = Encoder(F, D, dropout=0.1)

        # --- slot attention --- 
        self.K = K
        self.slot_attention_encoder = SlotAttentionEncoder(D, F, K, 4, 512, 0.1)

        # --- reconstruction decoder ---
        self.rec_dec = ReconstructionDecoder(F, D)
        self.cnn_smooth = nn.Conv2d(2*K, 2*K, kernel_size=(1, 3), padding='same')


    def preprocess_input(self, wav_file: str) -> torch.Tensor:
        with torch.no_grad():
            spec = self._get_spectrogram(wav_file)
            spec_chunked = self._chunk_spec(spec)
        return spec_chunked


    def _get_spectrogram(self, wav_file: str) -> torch.Tensor:
        wave, sr = torchaudio.load(wav_file)
        
        # to mono
        wave_mono = torch.mean(wave, dim=0)

        # resample
        if sr != self.default_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr).to(wave_mono.device)
        else: 
            resampler = self.resampler
        wave_mono_16k = resampler(wave_mono)
        
        # log mel spec
        mel_spec = self.to_mel(wave_mono_16k)
        log_mel_spec = self.to_db(mel_spec)

        # normalize
        log_mel_spec = torch.clamp(log_mel_spec, min=-80.0, max=0.0)
        log_mel_spec = (log_mel_spec + 80.0) / 80.0

        return log_mel_spec # (F, L)
    

    def _chunk_spec(self, spec: torch.Tensor) -> torch.Tensor:
        F, L = spec.shape
        T = self.T
        
        # pad full spec
        end_padding = (T - (L % T)) % T
        spec_padded = func.pad(spec, (self.P, self.P+end_padding), value=self.log_offset, mode='constant') 
        # spec_padded: (F, L + end_padding + 2*P)

        # chunk
        spec_chunked = spec_padded.unfold(dimension=-1, size=T+2*self.P, step=T) 
        spec_chunked = spec_chunked.permute(1, 2, 0)
        # spec_chunked: (chunks, T + 2*P, F)

        return spec_chunked
    

    def inference(self, wav_file: str) -> None:
        with torch.no_grad():
            # preprocess
            chunked_spec = self.preprocess_input(wav_file) # (B, T+2P, F)

            # forward
            return self.forward(chunked_spec) # (B, T, F)
            

    def sequential_inference(self, wav_file: str) -> None:
        with torch.no_grad():
            # preprocess
            chunked_spec = self.preprocess_input(wav_file) # (B, T+2P, F)

            # forward
            spec_rec_sequence = []
            num_chunks = chunked_spec.shape[0]
            for i in range(num_chunks):
                spec_rec = self.forward(chunked_spec[i, :, :].unsqueeze(0)) # (1, T, F)
                spec_rec_sequence.append(spec_rec)
            chunked_spec_rec = torch.cat(spec_rec_sequence, dim=0) # (B, T, F)

        return chunked_spec_rec
    

    def reconstruction_loss(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # estimate: (B, T, F)
        # target: (B, T+2*P, F)

        # crop target
        target = target[:, self.P:-self.P, :] # (B, T, F)

        # calculate loss
        reconstruction_loss = func.mse_loss(estimate, target, reduction="none")

        # sum across frequency, mean across time/batch
        return reconstruction_loss.sum(-1).mean()


    def forward_train(self, chunked_spec: torch.Tensor) -> None:
        # chunked_spec: (B, T+2P, F)
        chunked_spec_rec = self.forward(chunked_spec) # (B, T, F)
        return self.reconstruction_loss(chunked_spec_rec, chunked_spec)
    
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: (B, T + 2*P, F)
        B = spec.shape[0]
        T, F, M, K = self.T, self.F, self.M, self.K        
        
        # convert to frames
        x = spec.permute(0, 2, 1).contiguous() # (B, F, T + 2*P)
        x = x.unfold(dimension=-1, size=self.M, step=1).permute(0, 2, 1, 3).contiguous() # (B, T, F, M)

        # 1d cnn
        x = x.reshape(B*T, F, M).unsqueeze(1) # (B*T, 1, F, M)
        x = self.conv(x).permute(0, 2, 1, 3).contiguous() # (B*T, F, cnn_channel, M-(cnn_kernel-1))

        # frequency
        x = x.reshape(B*T, F, self.cnn_dim) # (B*T, F, cnn_dim)
        x = self.tok_emb_freq(x) # (B*T, F, D)

        # self attention encoder
        x = self.encoder(x) # (B*T, F, D)
        
        # slot attention encoder
        slots, _, _, _ = self.slot_attention_encoder(x) # (B*T, K, D)

        # reconstruction decoder
        output = self.rec_dec(slots) # (B*T, K, F, 2)

        # smooth output
        output = output.reshape(B, T, K, F, 2).permute(0, 2, 4, 3, 1).reshape(B, 2*K, F, T) # (B, K*2, F, T)
        output = self.cnn_smooth(output) # (B, K*2, F, T)
        
        # reconstruction
        output = output.reshape(B, K, 2, F, T).permute(0, 1, 2, 4, 3) # (B, K, 2, T, F)
        specs_rec = output[:, :, 0, :, :] # (B, K, T, F) 
        masks_rec = output[:, :, 1, :, :] # (B, K, T, F)
        masks_rec = func.softmax(masks_rec, dim=1) # (B, K, T, F)
        reconstruction = torch.sum(masks_rec * specs_rec, dim=1) # (B, T, F)

        return reconstruction


    def print_num_params(self) -> None:
        names = [
            ("conv", getattr(self, "conv", None)),
            ("tok_emb_freq", getattr(self, "tok_emb_freq", None)),
            ("encoder", getattr(self, "encoder", None)),
            ("slot_attention_encoder", getattr(self, "slot_attention_encoder", None)),
            ("rec_dec", getattr(self, "rec_dec", None)),
            ("cnn_smooth", getattr(self, "cnn_smooth", None)),
        ]

        total = 0
        for name, module in names:
            if module is None:
                continue
            cnt = sum(p.numel() for p in module.parameters())
            print(f"{name:30s}: {cnt:,}")
            total += cnt
        print(f"{'TOTAL':30s}: {total:,}")



class SelfAttentionEncoderBlock(nn.Module):
    def __init__(
            self,
            D: int,
            n_heads: int,
            pf_dim: int,
            dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.self_attention = MultiHeadAttention(D, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(D, pf_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)

        # self attention
        _x, _ = self.self_attention(x, x, x) # (B, N, D)
        x = self.norm(x + self.dropout(_x)) # (B, N, D)

        # positionwise feedforward
        _x = self.positionwise_feedforward(x) # (B, N, D)
        x = self.norm(x + self.dropout(_x)) # (B, N, D)

        return x
    

class CrossAttentionDecoderBlock(nn.Module):
    def __init__(
            self,
            D: int,
            n_heads: int,
            pf_dim: int,
            dropout: float
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.cross_attention = MultiHeadAttention(D, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(D, pf_dim, dropout)

    def forward(self, x: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        # x: (B, Nq, D)
        # trg: (B, Nkv, D)
        
        # cross attention
        _trg, attention = self.cross_attention(trg, x, x) # (B, Nkv, D)
        trg = self.norm(trg + self.dropout(_trg)) # (B, Nkv, D)

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg) # (B, Nkv, D)
        trg = self.norm(trg + self.dropout(_trg)) # (B, Nkv, D)

        return trg, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(
            self,
            D: int,
            pf_dim: int,
            dropout: float
    ) -> None:
        super().__init__()
        
        self.fc_1 = nn.Linear(D, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, D)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, D)
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x) # (B, N, D)
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            D: int,
            n_heads: int,
            dropout: float,
    ) -> None:
        super().__init__()
        
        assert D % n_heads == 0

        self.D = D
        self.n_heads = n_heads
        self.head_dim = D // n_heads

        self.Wq = nn.Linear(D, D)
        self.Wk = nn.Linear(D, D)
        self.Wv = nn.Linear(D, D)
        self.Wo = nn.Linear(D, D)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B = query.shape[0]
        # query: (B, Nq, D)
        # key: (B, Nk, D)
        # value: (B, Nv, D)

        # projections
        q = self.Wq(query).view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # (B, Nq, n_heads, head_dim)
        k = self.Wk(key).view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # (B, Nk, n_heads, head_dim)
        v = self.Wv(value).view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # (B, Nv, n_heads, head_dim)

        # norm
        scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(query.device)
        attn_logits = torch.matmul(q, k.permute(0, 1, 3, 2)) / scale # (B, n_heads, Nq, Nk)

        # attention
        attention = func.softmax(attn_logits, dim=-1) # (B, n_heads, Nq, Nk)
        
        # outputs
        x = torch.matmul(self.dropout(attention), v) # (B, n_heads, Nv, head_dim)
        x = x.permute(0, 2, 1, 3).contiguous() # (B, n_heads, Nv, head_dim)
        x = x.view(B, -1, self.D) # (B, Nv, D)

        # output fc
        x = self.Wo(x) # (B, Nv, D)

        return x, attention
    

class MultiHeadSlotAttention(nn.Module):
    def __init__(
            self,
            N: int,
            D: int,
            K: int,
            n_heads: int,
            dropout: float,
            mlp_size: int,
            num_iter: int = 3,
    ) -> None:
        super().__init__()

        assert D % n_heads == 0

        self.D = D
        self.N = N
        self.K = K
        self.n_heads = n_heads
        self.head_dim = D // n_heads
        self.num_iter = num_iter
        self.epsilon = 1e-8

        self.Wq = nn.Linear(D, D, bias=False)
        self.Wk = nn.Linear(D, D, bias=False)
        self.Wv = nn.Linear(D, D, bias=False)
        
        self.norm_inputs = nn.LayerNorm(D)
        self.norm_mlp = nn.LayerNorm(D)
        self.norm_slots = nn.LayerNorm(D)
        
        self.gru = nn.GRUCell(D, D)
        
        self.mlp = nn.Sequential(
            nn.Linear(D, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, D)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, slots_init: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B = x.shape[0]
        slots = slots_init

        # projections
        x = self.norm_inputs(x)
        k = self.Wk(x).view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous() # (B, n_heads, N, head_dim)
        v = self.Wv(x).view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous() # (B, n_heads, N, head_dim)
        
        # norm
        scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(x.device)
        k = k / scale

        # slot iterations
        for i in range(self.num_iter):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # attention
            q = self.Wq(slots).view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous() # (B, n_heads, K, head_dim)
            attn_logits = torch.matmul(k, q.permute(0, 1, 3, 2)) # (B, n_heads, N, K)

            attention = func.softmax(attn_logits, dim=-1) # (B, n_heads, N, K)
            attn_vis = attention.mean(1) # (B, N, K)

            # weighted mean
            attention = attention + self.epsilon
            attention = attention / torch.sum(attention, dim=-2, keepdim=True)
            updates = torch.matmul(self.dropout(attention.permute(0, 1, 3, 2)), v) # (B, n_heads, K, head_dim)
            updates = updates.permute(0, 2, 1, 3).reshape(B, self.K, -1) # (B, K, D)

            # slot update
            slots = self.gru(
                updates.view(-1, self.D), slots_prev.view(-1, self.D)
            )
            slots = slots.view(B, self.K, self.D)
            slots = slots + self.mlp(self.norm_mlp(slots)) # (B, K, D)

        return slots, attn_vis, attn_logits


class SlotAttentionEncoder(nn.Module):
    def __init__(
            self,
            D: int,
            N: int,
            K: int,
            n_heads: int,
            pf_dim: int,
            dropout: float,
            num_iter: int = 3
    ) -> None:
        super().__init__()

        self.num_iter = num_iter
        self.N = N
        self.D = D
        self.K = K
        self.n_heads = n_heads

        self.mlp = nn.Sequential(
            nn.Linear(D, pf_dim),
            nn.ReLU(),
            nn.Linear(pf_dim, D)
        )
        self.norm = nn.LayerNorm(D)

        self.slot_mu = nn.Parameter(torch.zeros(1, 1, D))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        self.slot_attention = MultiHeadSlotAttention(N, D, K, n_heads, dropout, pf_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B = x.shape[0]

        # mlp
        x = self.mlp(self.norm(x)) # (B, N, D)

        # init slots
        init_slots = torch.empty((B, self.K, self.D), device=x.device).normal_() # (B, K, D)
        init_slots = self.slot_mu + torch.exp(self.slot_log_sigma) * init_slots

        # slot attention
        slots, attn, attn_logits = self.slot_attention(x, init_slots)

        return slots, attn, init_slots, attn_logits
    

class Encoder(nn.Module):
    def __init__(
            self,
            F: int,
            D: int,
            dropout: float
    ) -> None:
        super().__init__()

        # --- positional embedding ---
        self.pos_embedding = nn.Embedding(F, D)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # --- self attention ---
        self.selfattn1 = SelfAttentionEncoderBlock(D, 4, 512, 0.1)
        self.selfattn2 = SelfAttentionEncoderBlock(D, 4, 512, 0.1)
        self.selfattn3 = SelfAttentionEncoderBlock(D, 4, 512, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, F, D)
        B, F, D = x.shape
        device = x.device

        # positional embedding
        pos = torch.arange(0, F).unsqueeze(0).repeat(B, 1).to(device) # (B, F)
        scale = torch.sqrt(torch.FloatTensor([D])).to(device)
        x = self.dropout((x * scale) + self.pos_embedding(pos)) # (B*T, F, D)

        # self attention
        x = self.selfattn1(x)
        x = self.selfattn2(x)
        x = self.selfattn3(x) # (B, F, D)

        return x


class ReconstructionDecoder(nn.Module):
    def __init__(
            self,
            F: int,
            D: int,
    ) -> None:
        super().__init__()
    
        self.F = F
        self.pos_embedding = nn.Embedding(F, D)

        self.mlp = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, 2),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, K, D)
        device = x.device

        x = x.unsqueeze(2).contiguous() # (B, K, 1, D)

        # positional embeddings
        pos_enc = torch.arange(0, self.F).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, F)
        pos_emb = self.pos_embedding(pos_enc) # (1, 1, F, D)
        x = x + pos_emb # (B, K, F, D)

        # decoder
        x = self.mlp(x) # (B, K, F, 2)

        return x

