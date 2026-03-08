import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as func
import numpy as np
from nnAudio.features import CQT2010v2


from model.utils import plot_spec

# === MODEL ===
class HSA(nn.Module):
    def __init__(
            self,
            F: int,
            P: int,
            T: int,
            D: int,
            K: int,
            target_sr: int = 22050,
            fft_bins: int = 2048,
            hop_sample: int = 256,
            bins_per_octave: int = 36,
            weight_a: float = 0.5,
            weight_b: float = 0.5,
            weight_chroma: float = 0.2,
            **kwargs
    ) -> None:
        super().__init__()
        self.F = F
        self.weight_a = weight_a
        self.weight_b = weight_b
        self.weight_chroma = weight_chroma

        # --- resampler ---
        self.target_sr = target_sr
        self.default_sr = 44100
        self.resampler = torchaudio.transforms.Resample(self.default_sr, target_sr)

        # --- log mel spec ---
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=fft_bins,
            win_length=2048,
            hop_length=hop_sample,
            pad_mode="constant",
            n_mels=F,
            norm="slaney"
        )
        self.log_offset = -80
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

        # --- cqt ---
        self.to_cqt = CQT2010v2(
            sr=target_sr,
            hop_length=hop_sample,
            bins_per_octave=bins_per_octave,
            n_bins=F,
            fmin=20.6,
            pad_mode="constant",
            verbose=False,
        )
        self.bins_per_octave = bins_per_octave

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

        # --- self attention across time ---
        self.self_attention_time = Encoder(T, D, dropout=0.1)

        # --- reconstruction decoder ---
        self.rec_dec = ReconstructionDecoder(F, D)
        self.cnn_smooth = nn.Conv2d(2*K, 2*K, kernel_size=(3, 1), padding='same')

        # --- chroma ---
        chroma_matrix = self._build_chroma_matrix()
        self.chroma_proj = nn.Linear(F, 12, bias=False)
        self.chroma_proj.weight.data = chroma_matrix.clone().detach()
        self.chroma_proj.weight.requires_grad = False

        # --- tonnetz ---
        tonnetz_matrix = self._build_tonnetz_matrix()
        self.tonnetz_proj = nn.Linear(12, 6, bias=False)
        self.tonnetz_proj.weight.data = tonnetz_matrix.clone().detach()
        self.tonnetz_proj.weight.requires_grad = False


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
        
        # log_cqt spec
        spec = self.to_cqt(wave_mono_16k).squeeze(0)
        log_spec = self.to_db(spec)

        return log_spec # (F, L)
    

    def _chunk_spec(self, spec: torch.Tensor) -> torch.Tensor:
        # spec (F, L)
        L = spec.shape[1]
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
    

    def db_to_power(self, x: torch.Tensor) -> torch.Tensor:
        return func.relu(torch.pow(10.0, x / 80.0))
    

    def power_to_db(self, x: torch.Tensor) -> torch.Tensor:
        return 80.0 * torch.log10(x.clamp(min=1e-8))

    
    def _build_chroma_matrix(self) -> torch.Tensor:
        bins_per_pitch = self.bins_per_octave // 12
        n_bins = self.F

        P = torch.zeros((12, n_bins))

        for i in range(n_bins):
            pitch_class = (i // bins_per_pitch) % 12
            P[pitch_class, i] = 1 / bins_per_pitch

        return P


    def _build_tonnetz_matrix(self) -> torch.Tensor:
        M = torch.zeros((6, 12))

        for k in range(12):
            M[0, k] = np.cos(7*np.pi*k/6)
            M[1, k] = np.sin(7*np.pi*k/6)
            
            M[2, k] = np.cos(3*np.pi*k/2)
            M[3, k] = np.sin(3*np.pi*k/2)
            
            M[4, k] = np.cos(2*np.pi*k/3)
            M[5, k] = np.sin(2*np.pi*k/3)
        
        return M


    def inference(self, wav_file: str) -> None:
        with torch.no_grad():
            # preprocess
            chunked_spec = self.preprocess_input(wav_file) # (B, T+2P, F)

            # forward
            specs_a, specs_b, masks_a, masks_b = self.forward(chunked_spec) # (B, 2*K, T, F)

            # reconstruction
            rec_a = torch.sum(masks_a * specs_a, dim=1) # (B, T, F)
            rec_b = torch.sum(masks_b * specs_b, dim=1) # (B, T, F)

            # to log
            rec_a = self.power_to_db(rec_a) # (B, T, F)
            rec_b = self.power_to_db(rec_b) # (B, T, F)

        return rec_a, rec_b, masks_a, masks_b
            

    def sequential_inference(self, wav_file: str) -> None:
        with torch.no_grad():
            # preprocess
            chunked_spec = self.preprocess_input(wav_file) # (B, T+2P, F)

            specs_a_seq = []
            specs_b_seq = []
            masks_a_seq = []
            masks_b_seq = []

            # forward
            num_chunks = chunked_spec.shape[0]
            for i in range(4):
                specs_a, specs_b, masks_a, masks_b = self.forward(chunked_spec[i, :, :].unsqueeze(0))
                specs_a_seq.append(specs_a) # (1, K, T, F)
                specs_b_seq.append(specs_b) # (1, K, T, F)
                masks_a_seq.append(masks_a) # (1, K, T, F)
                masks_b_seq.append(masks_b) # (1, K, T, F)
            
            # concat chunks
            specs_a = torch.cat(specs_a_seq, dim=0) # (B, K, T, F)
            specs_b = torch.cat(specs_b_seq, dim=0) # (B, K, T, F)
            masks_a = torch.cat(masks_a_seq, dim=0) # (B, K, T, F)
            masks_b = torch.cat(masks_b_seq, dim=0) # (B, K, T, F)

            # reconstruction
            rec_a = torch.sum(masks_a * specs_a, dim=1) # (B, T, F)
            rec_b = torch.sum(masks_b * specs_b, dim=1) # (B, T, F)

            # log spectrogram
            rec_a = self.power_to_db(rec_a) # (B, T, F)
            rec_b = self.power_to_db(rec_b) # (B, T, F)

        return rec_a, rec_b, masks_a, masks_b
    

    def reconstruction_loss(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # estimate: (B, T, F)
        # target: (B, T+2*P, F)

        # crop target
        target = target[:, self.P:-self.P, :] # (B, T, F)

        # also compute power loss
        target_power = self.db_to_power(target)
        estimate_power = self.db_to_power(estimate)

        # calculate loss
        reconstruction_loss = \
            0.1 * func.mse_loss(estimate_power, target_power, reduction="mean") + \
            func.l1_loss(estimate, target, reduction="mean")

        # sum across frequency, mean across time/batch
        return reconstruction_loss
    

    def tonnetz_loss(self, reconstructed_slots: torch.Tensor) -> torch.Tensor:
        """
        Input: Power spectrograms of the reconstructed slots
        """
        # (B, K, T, F)
        device = reconstructed_slots.device
        B = reconstructed_slots.shape[0]
        K, T = self.K, self.T
        
        # power chroma
        chroma = self.chroma_proj(reconstructed_slots) # (B, K, T, 12)
        chroma = chroma / (chroma.sum(dim=-1, keepdim=True) + 1e-8)
        chroma = chroma.permute(0, 2, 1, 3).reshape(B*T, K, 12) # (B*T, K, 6)

        # tonnetz
        # tonnetz = self.tonnetz_proj(chroma) # (B, K, T, 6)
        # tonnetz = func.normalize(tonnetz, dim=-1)

        # # reshape
        # tonnetz = tonnetz.permute(0, 2, 1, 3).reshape(B*T, K, 6) # (B*T, K, 6)

        # calculate similarity
        sim = torch.matmul(chroma, chroma.transpose(1, 2)) # (B*T, K, K)
        
        # remove self similarity
        mask = 1 - torch.eye(K, device=device)
        sim = sim * mask.unsqueeze(0)

        # calculate loss
        loss = (sim ** 2).mean()

        return loss


    def forward_train(self, chunked_spec: torch.Tensor) -> None:
        specs_a, specs_b, masks_a, masks_b = self.forward(chunked_spec) # (B, K, T, F)

        # mask
        rec_slots_a = masks_a * specs_a # (B, K, T, F)
        rec_slots_b = masks_b * specs_b # (B, K, T, F)
        
        # reconstruction
        rec_a = torch.sum(rec_slots_a, dim=1) # (B, T, F)
        rec_b = torch.sum(rec_slots_b, dim=1) # (B, T, F)
        
        # log spectrogram
        rec_a = self.power_to_db(rec_a) # (B, T, F)
        rec_b = self.power_to_db(rec_b) # (B, T, F)
        
        # reconstruction loss
        loss_rec_a = self.reconstruction_loss(rec_a, chunked_spec)
        loss_rec_b = self.reconstruction_loss(rec_b, chunked_spec)
        
        # chroma loss
        loss_chroma_a = self.tonnetz_loss(rec_slots_a)
        loss_chroma_b = self.tonnetz_loss(rec_slots_b)

        # loss weighting
        rec_loss    = self.weight_a*loss_rec_a + self.weight_b*loss_rec_b
        chroma_loss = self.weight_a*loss_chroma_a + self.weight_b*loss_chroma_b

        return (1 - self.weight_chroma)*rec_loss + self.weight_chroma*chroma_loss
    
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: (B, T + 2*P, F)
        B = spec.shape[0]
        T, F, M, K, D = self.T, self.F, self.M, self.K, self.D
        
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
        slots_a, _, _, _ = self.slot_attention_encoder(x) # (B*T, K, D)

        # temporal self attention
        slots_b = slots_a.reshape(B, T, K, D).permute(0, 2, 1, 3).reshape(B*K, T, D) # (B*K, T, D)
        slots_b = self.self_attention_time(slots_b) # (B*K, T, D)
        slots_b = slots_b.reshape(B, K, T, D).permute(0, 2, 1, 3).reshape(B*T, K, D) # (B*T, K, D)

        # reconstruction decoder
        output_a = self.rec_dec(slots_a) # (B*T, K, F, 2)
        output_b = self.rec_dec(slots_b) # (N*T, K, F, 2)

        # smooth output
        output_a = output_a.reshape(B, T, K, F, 2).permute(0, 2, 4, 1, 3).reshape(B, 2*K, T, F) # (B, K*2, T, F)
        output_b = output_b.reshape(B, T, K, F, 2).permute(0, 2, 4, 1, 3).reshape(B, 2*K, T, F) # (B, K*2, T, F)
        output_a = self.cnn_smooth(output_a).reshape(B, K, 2, T, F) # (B, K, 2, T, F)
        output_b = self.cnn_smooth(output_b).reshape(B, K, 2, T, F) # (B, K, 2, T, F)

        # reconstruction 
        specs_a = self.db_to_power(output_a[:, :, 0, :, :]) # (, K, T, F)
        specs_b = self.db_to_power(output_b[:, :, 0, :, :]) # (B, K, T, F)
        masks_a = func.softmax(output_a[:, :, 1, :, :], dim=1) # (B, K, T, F)
        masks_b = func.softmax(output_b[:, :, 1, :, :], dim=1) # (B, K, T, F)

        return specs_a, specs_b, masks_a, masks_b
    

    def print_num_params(self) -> None:
        names = [
            ("Pre-encoder", getattr(self, "tok_emb_freq", None)),
            ("Spectral Self Attention", getattr(self, "encoder", None)),
            ("Slot Attention", getattr(self, "slot_attention_encoder", None)),
            ("Temporal Self Attention", getattr(self, "self_attention_time", None)),
            ("Decoder", getattr(self, "rec_dec", None)),
        ]

        total = 0
        for name, module in names:
            if module is None:
                continue
            cnt = sum(p.numel() for p in module.parameters())
            print(f"{name:30s}: {cnt:,}")

        total = sum(p.numel() for p in self.parameters())
        print(f"{'TOTAL':30s}: {total:,}")


# === MAIN MODULES ===
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


# === HELPER MODULES ===
class SelfAttentionEncoderBlock(nn.Module):
    def __init__(
            self,
            D: int,
            n_heads: int,
            pf_dim: int,
            dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.self_attention = MultiHeadAttention(D, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(D, pf_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)

        # self attention
        _x, _ = self.self_attention(x, x, x) # (B, N, D)
        x = self.norm1(x + self.dropout(_x)) # (B, N, D)

        # positionwise feedforward
        _x = self.positionwise_feedforward(x) # (B, N, D)
        x = self.norm2(x + self.dropout(_x)) # (B, N, D)

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
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.cross_attention = MultiHeadAttention(D, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(D, pf_dim, dropout)

    def forward(self, x: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        # x: (B, Nq, D)
        # trg: (B, Nkv, D)
        
        # cross attention
        _trg, attention = self.cross_attention(trg, x, x) # (B, Nkv, D)
        trg = self.norm1(trg + self.dropout(_trg)) # (B, Nkv, D)

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg) # (B, Nkv, D)
        trg = self.norm2(trg + self.dropout(_trg)) # (B, Nkv, D)

        return trg, attention


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
