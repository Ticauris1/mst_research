from models.attention.self_attention import SelfAttentionBlock
from models.skin_mlp import Skin_Multi_Layer_Perceptron
from models.classifier import TwoLayerClassifierHead
import torch.nn.functional as F  # type: ignore
import timm # type: ignore
from models.attention.film import FiLM
from models.attention.cbam import CBAM
import torch # type: ignore
import torch.nn as nn # type: ignore

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
        dropout_rate=0.5  # ✅ New parameter
    ):
        super().__init__()
        self.use_film_before = use_film_before
        self.use_film_in_cbam = use_film_in_cbam
        self.use_triplet_embedding = use_triplet_embedding
        self.triplet_embedding_dim = triplet_embedding_dim
        self.include_skin_vec = include_skin_vec
        self.dropout_rate = dropout_rate

        self.base = timm.create_model(efficientnet_variant, pretrained=pretrained, num_classes=0)
        self.backbone = self.base
        C = self.base.num_features

        if self.use_film_before:
            self.film = FiLM(in_features=12, feature_map_channels=C)

        if attention_type == "self":
            self.attn = SelfAttentionBlock(C)
        elif attention_type == "cbam":
            self.attn = CBAM(C, use_film=self.use_film_in_cbam, film_in_dim=12)
        else:
            self.attn = nn.Identity()

        self.skin_mlp = Skin_Multi_Layer_Perceptron(input_dim=12)  # → 8D
        final_in_dim = C + 8 + (triplet_embedding_dim if use_triplet_embedding else 0)

        #self.dropout = nn.Dropout(p=dropout_rate)  # ✅ Dropout before classifier
        self.classifier = TwoLayerClassifierHead(input_dim=final_in_dim, output_dim=num_classes)

    def forward_features(self, x, skin_vec):
        x = self.base.forward_features(x)
        if self.use_film_before:
            x = self.film(x, skin_vec)
        x = self.attn(x, skin_vec) if isinstance(self.attn, CBAM) else self.attn(x)
        return x

    def forward(self, x, skin_vec, triplet_embedding=None, return_features=False):
        x = self.forward_features(x, skin_vec)

        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)

        features = x  # ✅ Save for t-SNE

        skin_feat = self.skin_mlp(skin_vec)

        # ✅ Ensure triplet_embedding shape is always valid if enabled
        if self.use_triplet_embedding:
            if triplet_embedding is None:
                triplet_embedding = torch.zeros((x.size(0), 512), device=x.device)
            elif triplet_embedding.dim() == 1:
                triplet_embedding = triplet_embedding.unsqueeze(0)
            elif triplet_embedding.shape[0] != x.shape[0]:
                raise ValueError("Triplet embedding batch mismatch")

        concat = [x, skin_feat]
        if self.use_triplet_embedding:
            concat.append(triplet_embedding)

        final_feat = torch.cat(concat, dim=1)
        logits = self.classifier(final_feat)

        if return_features:
            return logits, features  # ✅ Required for t-SNE extraction

        return logits

    def extract_features(self, x, skin_vec=None, triplet_embedding=None):
        x = self.forward_features(x, skin_vec)

        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)

        skin_feat = self.skin_mlp(skin_vec)

        if self.use_triplet_embedding:
            if triplet_embedding is None:
                triplet_embedding = torch.zeros((x.size(0), 512), device=x.device)
            elif triplet_embedding.dim() == 1:
                triplet_embedding = triplet_embedding.unsqueeze(0)
            elif triplet_embedding.shape[0] != x.shape[0]:
                raise ValueError("Triplet embedding batch mismatch")

        concat = [x, skin_feat]
        if self.use_triplet_embedding:
            concat.append(triplet_embedding)

        final_feat = torch.cat(concat, dim=1)
        return final_feat


    '''def forward(self, x, skin_vec, triplet_embedding=None, return_features=False):
        x = self.forward_features(x, skin_vec)
        
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        
        features = x  # ✅ Save for t-SNE

        skin_feat = self.skin_mlp(skin_vec)

        if self.use_triplet_embedding and triplet_embedding is not None:
            if triplet_embedding.dim() == 1:
                triplet_embedding = triplet_embedding.unsqueeze(0)
            elif triplet_embedding.shape[0] != x.shape[0]:
                raise ValueError("Triplet embedding batch mismatch")

        concat = [x, skin_feat]
        if self.use_triplet_embedding and triplet_embedding is not None:
            concat.append(triplet_embedding)

        final_feat = torch.cat(concat, dim=1)

        logits = self.classifier(final_feat)

        if return_features:
            return logits, features  # ✅ Required for t-SNE extraction

        return logits'''

    