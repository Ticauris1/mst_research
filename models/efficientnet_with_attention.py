from models.attention.self_attention import SelfAttentionBlock
from models.skin_mlp import Skin_Multi_Layer_Perceptron
from models.classifier import TwoLayerClassifierHead
import torch.nn.functional as F  # type: ignore
import timm # type: ignore
from models.attention.film import FiLM
from models.attention.cbam import CBAM
import torch # type: ignore
import torch.nn as nn # type: ignore

class ResNetWithAttention(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_name="resnet101d",
        attention_type="none",
        use_film_before=False,
        use_film_in_cbam=False,
        include_skin_vec=True,
        use_triplet_embedding=False,
        triplet_embedding_dim=512,
        drop_path_rate=0.3,
        dropout_rate=0.3,
        dropout_before_classifier=True,
        fusion_mode="concat",          # "concat" | "mlp" | "gated"
        fusion_hidden_dim=128
    ):
        super().__init__()
        self.include_skin_vec = include_skin_vec
        self.use_triplet_embedding = use_triplet_embedding
        self.triplet_embedding_dim = triplet_embedding_dim
        self.use_film_before = use_film_before
        self.use_film_in_cbam = use_film_in_cbam
        self.attention_type = attention_type
        self.dropout_before_classifier = dropout_before_classifier
        self.fusion_mode = fusion_mode

        # Backbone
        self.base = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, drop_path_rate=drop_path_rate
        )
        C = self.base.num_features  # feature channels out of forward_features()
        self._feat_dim = C

        # Optional FiLM before attention
        if self.use_film_before:
            self.film = FiLM(in_features=12, feature_map_channels=C)

        # Attention
        if attention_type == "self":
            self.attn = SelfAttentionBlock(C)
        elif attention_type == "cbam":
            self.attn = CBAM(C, use_film=self.use_film_in_cbam, film_in_dim=12)
        else:
            self.attn = nn.Identity()

        # Skin MLP: 12 -> 8
        self.skin_mlp = Skin_Multi_Layer_Perceptron(input_dim=12)

        # Fusion setup
        if fusion_mode in ["mlp", "gated"]:
            self.image_proj = nn.Linear(C, fusion_hidden_dim)
            self.skin_proj = nn.Linear(8, fusion_hidden_dim)
            if self.use_triplet_embedding:
                self.triplet_proj = nn.Linear(triplet_embedding_dim, fusion_hidden_dim)

            if fusion_mode == "gated":
                gate_input_dim = C + 8 + (triplet_embedding_dim if self.use_triplet_embedding else 0)
                self.gate = nn.Sequential(
                    nn.Linear(gate_input_dim, 3),
                    nn.Softmax(dim=1)
                )
            final_in_dim = fusion_hidden_dim
        else:
            # concat mode
            final_in_dim = C + 8 + (triplet_embedding_dim if self.use_triplet_embedding else 0)

        self.expected_final_dim = final_in_dim
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = TwoLayerClassifierHead(input_dim=final_in_dim, output_dim=num_classes)

    # --- Utility: good Grad-CAM layer for resnets from timm ---
    def get_gradcam_target_layer(self):
        # timm resnetd variants still expose layer4 typically
        if hasattr(self.base, "layer4"):
            return self.base.layer4
        # fallback: last child
        children = list(self.base.children())
        return children[-1] if len(children) else self.base

    def forward_features(self, x, skin_vec):
        x = self.base.forward_features(x)
        if self.use_film_before:
            assert skin_vec is not None and skin_vec.dim() == 2, \
                f"[ResNet] FiLM needs skin_vec [B,12], got {None if skin_vec is None else skin_vec.shape}"
            x = self.film(x, skin_vec)
        # Only pass skin_vec into CBAM when CBAM is used
        x = self.attn(x, skin_vec) if isinstance(self.attn, CBAM) else self.attn(x)
        return x

    def forward(self, x, skin_vec=None, triplet_embedding=None, return_features=False):
        B = x.size(0)
        # Ensure skin_vec exists shape [B,12]
        if self.include_skin_vec:
            if skin_vec is None:
                skin_vec = torch.zeros((B, 12), device=x.device, dtype=x.dtype)
        else:
            # if excluded, feed zeros so downstream stays happy
            skin_vec = torch.zeros((B, 12), device=x.device, dtype=x.dtype)

        # Feature maps
        feat = self.forward_features(x, skin_vec)
        assert feat.dim() == 4 and feat.size(1) == self._feat_dim, \
            f"[ResNet] Expected [B,{self._feat_dim},H,W], got {feat.shape}"

        # Global pool -> [B,C]
        feat = F.adaptive_avg_pool2d(feat, 1).view(B, -1)
        features = feat  # for TSNE/etc.

        # Skin -> [B,8]
        skin_feat = self.skin_mlp(skin_vec)

        # Triplet -> [B,512] (if enabled)
        if self.use_triplet_embedding:
            if triplet_embedding is None:
                triplet_embedding = torch.zeros((B, self.triplet_embedding_dim), device=x.device, dtype=feat.dtype)
            elif triplet_embedding.dim() == 1:
                triplet_embedding = triplet_embedding.unsqueeze(0)
            elif triplet_embedding.shape[0] != B:
                raise ValueError(f"[ResNet] Triplet batch mismatch {triplet_embedding.shape[0]} vs {B}")

        # ---- Fusion ----
        if self.fusion_mode == "concat":
            parts = [feat, skin_feat] + ([triplet_embedding] if self.use_triplet_embedding else [])
            final_feat = torch.cat(parts, dim=1)

        elif self.fusion_mode == "mlp":
            feat_proj = self.image_proj(feat)
            skin_proj = self.skin_proj(skin_feat)
            if self.use_triplet_embedding:
                triplet_proj = self.triplet_proj(triplet_embedding)
                final_feat = feat_proj + skin_proj + triplet_proj
            else:
                final_feat = feat_proj + skin_proj

        elif self.fusion_mode == "gated":
            gate_in = torch.cat([feat, skin_feat] + ([triplet_embedding] if self.use_triplet_embedding else []), dim=1)
            weights = self.gate(gate_in)  # [B,3]
            feat_proj = self.image_proj(feat)
            skin_proj = self.skin_proj(skin_feat)
            if self.use_triplet_embedding:
                triplet_proj = self.triplet_proj(triplet_embedding)
                final_feat = weights[:, 0:1] * feat_proj + weights[:, 1:2] * skin_proj + weights[:, 2:3] * triplet_proj
            else:
                final_feat = weights[:, 0:1] * feat_proj + weights[:, 1:2] * skin_proj
        else:
            raise ValueError(f"[ResNet] Unknown fusion mode: {self.fusion_mode}")

        #print(f"[ResNet] final_feat: {final_feat.shape}")

        # Guards before classifier
        assert final_feat.dim() == 2, f"[ResNet] final_feat must be 2D, got {final_feat.shape}"
        assert final_feat.size(1) == self.expected_final_dim, \
            f"[ResNet] expected {self.expected_final_dim} features, got {final_feat.size(1)}"

        if self.dropout_before_classifier:
            final_feat = self.dropout(final_feat)
            logits = self.classifier(final_feat)
        else:
            logits = self.classifier(final_feat)
            logits = self.dropout(logits)

        #print(f"[ResNet] logits: {logits.shape}")
        return (logits, features) if return_features else logits

class EfficientNetWithAttention(nn.Module):
    def __init__(
        self,
        num_classes,
        attention_type="none",
        pretrained=True,
        use_film=False,
        use_film_before=False,
        use_film_in_cbam=False,
        use_triplet_embedding=False,
        triplet_embedding_dim=512,
        include_skin_vec=True,
        efficientnet_variant="efficientnet_b0",
        dropout_rate=0.5
    ):
        super().__init__()
        self.use_film_before = use_film_before
        self.use_film_in_cbam = use_film_in_cbam
        self.use_triplet_embedding = use_triplet_embedding
        self.triplet_embedding_dim = triplet_embedding_dim
        self.include_skin_vec = include_skin_vec

        self.base = timm.create_model(efficientnet_variant, pretrained=pretrained, num_classes=0)
        C = self.base.num_features

        if self.use_film_before:
            self.film = FiLM(in_features=12, feature_map_channels=C)

        if attention_type == "self":
            self.attn = SelfAttentionBlock(C)
        elif attention_type == "cbam":
            self.attn = CBAM(C, use_film=self.use_film_in_cbam, film_in_dim=12)
        else:
            self.attn = nn.Identity()

        self.skin_mlp = Skin_Multi_Layer_Perceptron(input_dim=12)  # -> 8D

        self.expected_final_dim = C + 8 + (triplet_embedding_dim if use_triplet_embedding else 0)
        self.classifier = TwoLayerClassifierHead(input_dim=self.expected_final_dim, output_dim=num_classes)

    def forward_features(self, x, skin_vec):
        x = self.base.forward_features(x)
        if self.use_film_before:
            x = self.film(x, skin_vec)
        if isinstance(self.attn, CBAM):
            x = self.attn(x, skin_vec)
        else:
            x = self.attn(x)
        return x

    def forward(self, x, skin_vec=None, triplet_embedding=None, return_features=False):
        B = x.size(0)
        skin_vec = skin_vec if skin_vec is not None else torch.zeros((B, 12), device=x.device)

        x = self.forward_features(x, skin_vec)
        #print(f"[EffNet] pre-pool: {x.shape}")
        assert x.dim() == 4, f"[EffNet] expected 4D before pool, got {x.shape}"

        x = F.adaptive_avg_pool2d(x, 1).view(B, -1)  # [B,C]
        features = x

        skin_feat = self.skin_mlp(skin_vec)          # [B,8]

        # Triplet
        if self.use_triplet_embedding:
            if triplet_embedding is None:
                triplet_embedding = torch.zeros((B, self.triplet_embedding_dim), device=x.device)
            elif triplet_embedding.dim() == 1:
                triplet_embedding = triplet_embedding.unsqueeze(0)
            elif triplet_embedding.shape[0] != B:
                raise ValueError(f"[EffNet] Triplet batch mismatch {triplet_embedding.shape[0]} vs {B}")

        parts = [x, skin_feat] + ([triplet_embedding] if self.use_triplet_embedding else [])
        final_feat = torch.cat(parts, dim=1)
        #print(f"[EffNet] final_feat: {final_feat.shape}")

        # Hard guards
        assert final_feat.dim() == 2, f"[EffNet] final_feat must be 2D, got {final_feat.shape}"
        assert final_feat.size(1) == self.expected_final_dim, (
            f"[EffNet] expected {self.expected_final_dim} features, got {final_feat.size(1)}"
        )

        logits = self.classifier(final_feat)
        #print(f"[EffNet] logits: {logits.shape}")

        return (logits, features) if return_features else logits


