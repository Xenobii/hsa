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

        # --- margin cnn ---
        self.cnn_channel = 4
        self.M           = 2*P + 1
        self.convmargin  = nn.Conv2d(1, self.cnn_channel, kernel_size=(self.M, 1))

        # --- encoder --- 
        self.P = P
        self.T = T
        self.D = D
        f = F
        for _ in range(4):
            f = f // 2 - 1
        self.encoder = Encoder(F, D, self.cnn_channel)

        # --- slot attention --- 
        self.K = K
        self.slot_attention_encoder = SlotAttentionEncoder(D, F, K, 4, 512, 0.1)

        # --- reconstruction decoder ---
        self.rec_dec = ReconstructionDecoder(F, T, D)
        self.cnn_smooth = nn.Conv2d(2*K, 2*K, kernel_size=(3, 3), padding='same')

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
        # spec: (B, T+2P, F)
        B = spec.shape[0]
        T, F, M, K = self.T, self.F, self.M, self.K

        # convert to frames
        x = self.convmargin(spec.unsqueeze(1)) # (B, C1, T, F) 

        # conv encoder
        h = self.encoder(x).transpose(1, 2) # (B, T, D)
        
        # slot attention encoder
        slots, _, _, _ = self.slot_attention_encoder(h) # (B, K, D)

        # reconstruction decoder
        output_a = self.rec_dec(slots) # (B, K, T, F, 2)

        # smooth output
        output_a = output_a.reshape(B, T, K, F, 2).permute(0, 2, 4, 1, 3).reshape(B, 2*K, T, F) # (B, K*2, T, F)
        output_a = self.cnn_smooth(output_a).reshape(B, K, 2, T, F) # (B, K, 2, T, F)

        # reconstruction 
        specs_a = self.db_to_power(output_a[:, :, 0, :, :]) # (B, K, T, F)
        masks_a = func.softmax(output_a[:, :, 1, :, :], dim=1) # (B, K, T, F)

        specs_b = specs_a
        masks_b = masks_a

        return specs_a, specs_b, masks_a, masks_b
    

    def print_num_params(self) -> None:
        names = [
            ("Pre-encoder", getattr(self, "tok_emb_freq", None)),
            ("Temporal Self Attention", getattr(self, "self_attention_time", None)),
            ("Slot Attention", getattr(self, "slot_attention_encoder", None)),
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
            ch_in: int,
            model_complexity: int = 2
    ) -> None:
        super().__init__()

        base = 2 ** (model_complexity - 1)
        channels = [2*base, 4*base, 8*base, 16*base, 32*base]
        
        if D is None:
            D = channels[-1]

        # --- encoding ---
        self.convin = nn.Sequential(
            nn.Conv2d(ch_in, channels[0], kernel_size=3, padding='same'),
            nn.ELU(inplace=True)
        )

        # --- frequency encoding ---
        self.block1 = EncoderBlock(channels[0], channels[1], stride=2)
        self.block2 = EncoderBlock(channels[1], channels[2], stride=2)
        self.block3 = EncoderBlock(channels[2], channels[3], stride=2)
        self.block4 = EncoderBlock(channels[3], channels[4], stride=2)

        # --- output process ---
        f = F
        for _ in range(4):
            f = f // 2 - 1
        self.D = D
        self.convlat = nn.Conv2d(channels[4], D, kernel_size=(1, f))


    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, F)
        x = self.convin(spectrogram) # (B, C1, T-4, F)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x) # (B, C, T, f)
        
        h = self.convlat(x) # (B, D, T, 1)
        h = h.squeeze(-1) # (B, D, T, f)


        return h 
    

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
            T: int,
            D: int,
    ) -> None:
        super().__init__()
    
        self.F = F
        self.T = T
        self.time_embedding = nn.Embedding(T, D)
        self.freq_embedding = nn.Embedding(F, D)

        self.mlp = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, 2*D),
            nn.ReLU(),
            nn.Linear(2*D, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, K, D)
        device = x.device

        x = x[:, :, None, None, :] # (B, K, 1, 1, D)

        # positional indices
        t_idx = torch.arange(self.T, device=device) 
        f_idx = torch.arange(self.F, device=device) 
        
        # positional embeddings
        t_emb = self.time_embedding(t_idx) # (T, D)
        f_emb = self.freq_embedding(f_idx) # (T, D)
        pos_emb = t_emb[:, None, :] + f_emb[None, :, :] # (T, F, D)
        pos_emb = pos_emb[None, None] # (1, 1, T, F, D)
        
        x = x + pos_emb # (B, K, T, F, D)

        # decoder
        x = self.mlp(x) # (B, K, T, F, 2)

        return x


# === HELPER MODULES ===
class EncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 2
    ) -> None:
        super().__init__()

        self.block1 = ResidualConv2dBlock(in_channels, in_channels, kernel_size=3, dilation=1)
        self.block2 = ResidualConv2dBlock(in_channels, in_channels, kernel_size=3, dilation=2)
        self.block3 = ResidualConv2dBlock(in_channels, in_channels, kernel_size=3, dilation=3)

        self.hop = stride
        self.win = 2 * stride

        self.sconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, self.win), stride=(1, self.hop)),
            nn.ELU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process features
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)

        # Downsample
        y = self.sconv(y)

        return y
    

class ResidualConv2dBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            dilation: int = 1
    ) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same', dilation=dilation),
            nn.ELU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ELU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process features
        y = self.conv1(x)
        y = self.conv2(y)

        # Residual
        y = y + x

        return y
    

class TemporalTransformer(nn.Module):
    def __init__(self, channels, nhead=8, layers=4, dim_ff=1024, dropout=0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x):
        # x: B,C,F,T

        B, C, F, T = x.shape

        x = x.permute(0, 2, 3, 1)      # B,F,T,C
        x = x.reshape(B * F, T, C)     # (B*F),T,C

        x = self.encoder(x)

        x = x.reshape(B, F, T, C)
        x = x.permute(0, 3, 1, 2)      # B,C,F,T

        return x
    

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
