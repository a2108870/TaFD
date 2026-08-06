# -*- coding: utf-8 -*-
"""
models/resnet_cifar_firstblock_FDConv_10_mix.py

鏁村悎鐗堟湰锛氭敮鎸?ResNet 鍜?MobileViT 涓ょ楠ㄦ灦锛岄€氳繃 create_encoder(backbone=...) 閫夋嫨銆?

浣跨敤鏂规硶锛?
  from models.resnet_cifar_firstblock_FDConv_10_mix import create_encoder
  model = create_encoder(backbone='resnet', num_classes=100, num_domains=4)
  model = create_encoder(backbone='mobilevit', num_classes=100, num_domains=4)

鏋舵瀯璇存槑锛?
  鈥?ResNetEncoder: ResNet-34 楠ㄦ灦 + FDConv (stem + 姣忛樁娈甸鍧?conv1)
  鈥?MobileViTEncoder: MobileViT 楠ㄦ灦 + FDConv (stem + 鎸囧畾 MV2Block depthwise)
  鈥?涓よ€呭潎鏀寔 BPDA (set_bpda 鏂规硶)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fdconv_woDomainUniformMix import FCConv

# 鍔ㄦ€佽仛绫?
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
#  鏁版嵁闆嗘爣鍑嗗寲鍙傛暟
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
DATASET_STATS = {
    'CIFAR10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
    'CIFAR100': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
    'TinyImageNet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
    'Imagenette': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
    'ImageNet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
}


class InputNormalize(nn.Module):
    """Input normalization layer with registered mean/std buffers."""
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
#  棰戣氨鐗瑰緛鎻愬彇鍣紙鐢ㄤ簬 WaveletDomainManager 鍩熷垎绫伙級
#  鎻愬彇鐗瑰緛锛氬熀纭€12缁达紙鑳介噺浠介+鑹插害瑙掞級+ 鑹插害涓诲鎬?缁?= 15缁?
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class SpectralFeatureExtractor:
    """
    棰戣氨鐗瑰緛鎻愬彇鍣紝鐢ㄤ簬濞佽儊鍩熷垎绫汇€?

    鎻愬彇鐨勭壒寰侊細
    - 9涓兘閲忎唤棰濓紙YCbCr 脳 浣?涓?楂橀锛?
    - 3涓壊搴﹁
    - 3涓壊搴︿富瀵兼€э紙浣?涓?楂橀鐨?Cr-Cb 宸紓锛?

    鎬荤淮搴︼細15
    """

    def __init__(self, eps=1e-6):
        self.eps = float(eps)
        self.band_edges = (0.2, 0.5)

        self._cached_shape = None
        self._cached_device = None
        self._cached_dtype = None
        self._bands_onehot = None

    def output_dim(self) -> int:
        return 15  # 12 (base) + 3 (chroma dominance)

    def _prepare_cache(self, H, Wc, device, dtype):
        need = (
            self._cached_shape != (H, Wc)
            or self._cached_device != device
            or self._cached_dtype != dtype
        )
        if not need:
            return

        fy = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype)
        fx = torch.fft.rfftfreq(Wc * 2 - 2 if Wc > 1 else 2, d=1.0, device=device, dtype=dtype)
        ky = fy.view(H, 1).expand(H, Wc)
        kx = fx.view(1, Wc).expand(H, Wc)
        radius = torch.sqrt(ky ** 2 + kx ** 2)
        radius = radius / (radius.max().clamp_min(1.0))
        r = radius.reshape(-1)

        low_th, mid_th = self.band_edges
        low = (r <= low_th)
        mid = (r > low_th) & (r <= mid_th)
        high = (r > mid_th)
        bands = torch.stack([low, mid, high], dim=1).to(dtype)

        self._cached_shape = (H, Wc)
        self._cached_device = device
        self._cached_dtype = dtype
        self._bands_onehot = bands

    @staticmethod
    def _rgb_to_ycbcr_fft(rgb_fft):
        R = rgb_fft[:, 0]; G = rgb_fft[:, 1]; B = rgb_fft[:, 2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.168736 * R - 0.331264 * G + 0.5 * B
        Cr = 0.5 * R - 0.418688 * G - 0.081312 * B
        return torch.stack([Y, Cb, Cr], dim=1)

    def features_from_images(self, img_pixel):
        """
        浠庡浘鍍忔彁鍙栭璋辩壒寰併€?

        Args:
            img_pixel: [B, 3, H, W] 杈撳叆鍥惧儚

        Returns:
            features: [B, 15] 棰戣氨鐗瑰緛鍚戦噺
        """
        B, C, H, W = img_pixel.shape
        img_fft = torch.fft.rfft2(img_pixel.float(), norm='ortho')
        self._prepare_cache(H, img_fft.size(-1), img_pixel.device, img_fft.real.dtype)

        ycbcr_fft = self._rgb_to_ycbcr_fft(img_fft)
        # energy = (ycbcr_fft.abs() ** 2)
        energy = ycbcr_fft.real.pow(2) + ycbcr_fft.imag.pow(2)

        # 鍩虹鐗瑰緛锛?涓兘閲忎唤棰?+ 3涓壊搴﹁
        _, _, H_, Wc_ = energy.shape
        N = H_ * Wc_
        e_flat = energy.reshape(B, 3, N)
        tot = e_flat.sum(dim=2).clamp_min(self.eps)
        band_e = torch.einsum('bcn,nk->bck', e_flat, self._bands_onehot)
        share = band_e / tot.unsqueeze(-1)

        cb_band = band_e[:, 1, :]
        cr_band = band_e[:, 2, :]
        cb_norm = torch.sqrt(cb_band + self.eps)
        cr_norm = torch.sqrt(cr_band + self.eps)
        chroma_angle = torch.atan2(cr_norm, cb_norm)

        feats_base = torch.cat([share.reshape(B, 9), chroma_angle], dim=1)  # [B, 12]

        # 鑹插害涓诲鎬х壒寰?
        cb_b = band_e[:, 1, :]
        cr_b = band_e[:, 2, :]
        chroma_dominance = (cr_b - cb_b) / (cr_b + cb_b + self.eps)  # [B, 3]

        return torch.cat([feats_base, chroma_dominance], dim=1)  # [B, 15]


# ------------------------------------------------------------------------------
#  Threat-Domain Classifier (Lightweight CNN for domain prediction)
# ------------------------------------------------------------------------------
class ThreatDomainClassifier(nn.Module):
    """Lightweight Domain Classifier G for threat-domain prediction."""

    def __init__(self, input_channels=3, num_domains=3):
        super(ThreatDomainClassifier, self).__init__()
 
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(64, num_domains)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        domain_logits = self.classifier(features)
        return domain_logits


# ------------------------------------------------------------------------------
#  Threat-Domain Diagnosis (Dynamic clustering and alignment)
# ------------------------------------------------------------------------------
class ThreatDomainDiagnosis:
    """Threat-Domain Diagnosis module for spectral prototype learning and clustering."""

    def __init__(self, num_sources=6, num_threat_domains=3, feature_ids=None, source_names=None):
        self.num_sources = num_sources
        self.num_threat_domains = num_threat_domains
        # 鏂板锛氭敮鎸佸姩鎬佹敾鍑诲悕绉?
        if source_names is not None:
            self.source_names = source_names
        else:
            self.source_names = ['PGD', 'ACE', 'Hue', 'ReColorAdv', 'Light', 'UAA']

        self.source_features = {}
        self.source_counts = {}
        self.source_to_domain = {i: 0 for i in range(num_sources)}

        self.momentum = 0.99
        self.is_mapping_fixed = False
        self.cluster_centers = None
        self.update_counter = 0
        self.feature_record_frequency = 1
        self.feature_record_count = 0
        self.previous_clustering = None
        self.cluster_count_warning_emitted = False

        self.last_update_epoch = -1
        self.mapping_ever_updated = False

        self.extractor = SpectralFeatureExtractor()

    def _compute_features(self, img_pixel_batch):
        return self.extractor.features_from_images(img_pixel_batch)

    def compute_wavelet_features(self, img_pixel_batch, domain_ids=None):
        if img_pixel_batch.size(0) == 0:
            return torch.empty(0, 0, device=img_pixel_batch.device)
        return self._compute_features(img_pixel_batch)

    def update_features(self, img_pixel_batch, domain_ids):
        with torch.no_grad():
            self.update_counter += 1
            self.feature_record_count += 1

            feats = self.compute_wavelet_features(img_pixel_batch, domain_ids)
            if feats.numel() == 0:
                return False

            for d in torch.unique(domain_ids):
                d_int = int(d.item())
                mask = (domain_ids == d)
                if not mask.any():
                    continue
                f_src = feats[mask].mean(dim=0)
                if d_int in self.source_features:
                    proto = self.source_features[d_int].to(f_src.device)
                    self.source_features[d_int] = self.momentum * proto + (1 - self.momentum) * f_src
                    self.source_counts[d_int] = self.source_counts.get(d_int, 0) + int(mask.sum().item())
                else:
                    self.source_features[d_int] = f_src
                    self.source_counts[d_int] = int(mask.sum().item())
            return True

    def apply_hungarian_matching(self, new_labels, valid_sources, num_clusters=None):
        if self.previous_clustering is None:
            self.previous_clustering = {source: lab for source, lab in zip(valid_sources, new_labels)}
            if 0 in valid_sources:
                pgd_idx = valid_sources.index(0)
                pgd_cluster = new_labels[pgd_idx]
                if pgd_cluster != 0:
                    mapping = {pgd_cluster: 0, 0: pgd_cluster}
                    adjusted = {}
                    for i, source in enumerate(valid_sources):
                        lab = new_labels[i]
                        adjusted[source] = mapping.get(lab, lab)
                    self.previous_clustering = adjusted
            return {source: self.previous_clustering[source] for source in valid_sources}

        if num_clusters is None:
            num_clusters = max(int(np.max(new_labels)) + 1, 1)
        cost = np.zeros((num_clusters, num_clusters))
        for i in range(num_clusters):
            for j in range(num_clusters):
                mismatch = 0
                for idx, source in enumerate(valid_sources):
                    if source in self.previous_clustering:
                        if new_labels[idx] == i and self.previous_clustering[source] != j:
                            mismatch += 1
                cost[i, j] = mismatch
        row_ind, col_ind = linear_sum_assignment(cost)
        label_mapping = {i: col_ind[i] for i in range(num_clusters)}
        adjusted = {source: label_mapping[new_labels[i]] for i, source in enumerate(valid_sources)}
        self.previous_clustering = adjusted.copy()
        return adjusted

    def update_domain_mapping(self, epoch=None, total_epochs=None):
        if epoch is not None and epoch != self.last_update_epoch:
            self.last_update_epoch = epoch
        if self.is_mapping_fixed:
            return self.source_to_domain
        if epoch is not None and total_epochs is not None and total_epochs > 0:
            if epoch / float(total_epochs) > 0.8:
                self.is_mapping_fixed = True
                return self.source_to_domain

        valid_sources = sorted(list(self.source_features.keys()))
        if len(valid_sources) < 2:
            return self.source_to_domain

        feats = np.stack([self.source_features[d].detach().cpu().numpy() for d in valid_sources], axis=0)

        scaler = StandardScaler()
        feats_scaled = scaler.fit_transform(feats)

        feats_red = PCA(n_components=4, random_state=42).fit_transform(feats_scaled) if feats_scaled.shape[1] > 4 else feats_scaled

        active_num_clusters = min(self.num_threat_domains, len(valid_sources))
        if active_num_clusters < self.num_threat_domains:
            if not self.cluster_count_warning_emitted:
                print(
                    f"[ThreatDomainDiagnosis] Active sources={len(valid_sources)} < "
                    f"requested domains={self.num_threat_domains}; using {active_num_clusters} clusters this round."
                )
                self.cluster_count_warning_emitted = True

        kmeans = KMeans(n_clusters=active_num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feats_red)

        self.cluster_centers = kmeans.cluster_centers_

        adjusted = self.apply_hungarian_matching(labels, valid_sources, num_clusters=active_num_clusters)
        for source, domain_id in adjusted.items():
            self.source_to_domain[source] = int(domain_id)

        self.mapping_ever_updated = True
        return self.source_to_domain

    def get_domain_assignments(self, source_ids):
        device = source_ids.device
        max_source_id = max(self.source_to_domain.keys()) if self.source_to_domain else 0
        mapping = torch.zeros(max_source_id + 1, dtype=torch.long, device=device)
        for s, d in self.source_to_domain.items():
            mapping[s] = max(0, int(d))
        return mapping[torch.clamp(source_ids, 0, max_source_id)]

    def get_mapping_status(self):
        return {self.source_names[s]: f"Domain-{self.source_to_domain.get(s, 0)}"
                for s in range(min(len(self.source_names), self.num_sources))}

    def get_state_dict(self):
        return {
            'source_to_domain': self.source_to_domain,
            'source_features': {k: v.cpu() for k, v in self.source_features.items()},
            'source_counts': self.source_counts,
            'cluster_centers': self.cluster_centers.tolist() if self.cluster_centers is not None else None,
            'is_mapping_fixed': self.is_mapping_fixed,
            'update_counter': self.update_counter,
            'feature_record_count': self.feature_record_count,
            'mapping_ever_updated': self.mapping_ever_updated,
            'previous_clustering': self.previous_clustering,
            'last_update_epoch': self.last_update_epoch
        }

    def load_state_dict(self, state_dict, device=None):
        self.source_to_domain = state_dict.get('source_to_domain', self.source_to_domain)
        self.source_counts = state_dict.get('source_counts', {})
        self.is_mapping_fixed = state_dict.get('is_mapping_fixed', False)
        self.update_counter = state_dict.get('update_counter', 0)
        self.feature_record_count = state_dict.get('feature_record_count', 0)
        self.mapping_ever_updated = state_dict.get('mapping_ever_updated', False)
        self.previous_clustering = state_dict.get('previous_clustering', None)
        self.last_update_epoch = state_dict.get('last_update_epoch', -1)
        self.cluster_centers = (np.array(state_dict['cluster_centers'])
                                if state_dict.get('cluster_centers') is not None else None)

        # 鈽?[淇敼] 鎭㈠鐗瑰緛鍘熷瀷锛屼繚鎸?EMA 杩炵画鎬?
        saved_features = state_dict.get('source_features', None)
        if saved_features is not None:
            self.source_features = {}
            for k, v in saved_features.items():
                # 纭繚鍔犺浇鍒版纭殑璁惧
                if device is not None:
                    self.source_features[int(k)] = v.to(device)
                else:
                    self.source_features[int(k)] = v
        else:
            self.source_features = {}


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
#  鍩虹妯″潡锛團DConv 浠呯敤浜庢寚瀹氫綅缃級
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def conv3x3(in_planes, out_planes, stride=1, layer_name="",
            use_fdconv=False, num_domains=3, num_experts=4):
    if use_fdconv:
        return FCConv(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            num_domains=num_domains,
            num_experts=num_experts
        )
    else:
        return nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None,
                 layer_name="", use_fdconv_conv1=False,
                 num_threat_domains=3, fd_num_experts=4):
        super(BasicBlock, self).__init__()
        # conv1: use FCConv only when use_fdconv_conv1=True
        if use_fdconv_conv1:
            self.conv1_is_fd = True
            self.conv1 = conv3x3(in_planes, planes, stride, layer_name + "_conv1",
                                 use_fdconv=True, num_domains=num_threat_domains,
                                 num_experts=fd_num_experts)
        else:
            self.conv1_is_fd = False
            self.conv1 = conv3x3(in_planes, planes, stride, layer_name + "_conv1",
                                 use_fdconv=False)

        # conv2锛氬缁堟櫘閫氬嵎绉?
        self.conv2 = conv3x3(planes, planes, 1, layer_name + "_conv2",
                             use_fdconv=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    # 鈽?鏀硅繖閲岋細澧炲姞 router_weights / bpda锛堥粯璁や笉褰卞搷鍘熼€昏緫锛?
    def forward(self, x, domain_assignments=None, router_weights=None, bpda=False, soft_forward=False):
        identity = x
        if self.conv1_is_fd:
            # 鈽?鎶?router_weights / bpda 浼犺繘 FDConv
            out = self.conv1(
                x, domain_assignments,
                router_weights=router_weights,
                bpda=bpda,
                soft_forward=soft_forward
            )[0]
        else:
            out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
#  ResNetEncoder (鍘?Encoder)
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class ResNetEncoder(nn.Module):
    def __init__(self, num_sources=6, num_domains=3, fd_num_experts=3,
                 enable_band_stats=False, domain_feature_ids=None, num_classes=10,
                 dataset='CIFAR100', source_names=None):
        super(ResNetEncoder, self).__init__()
        block = BasicBlock
        self.in_channels = 64
        self.num_sources = num_sources
        self.num_threat_domains = int(num_domains)

        # 杈撳叆鏍囧噯鍖栧眰锛堟牴鎹暟鎹泦鑷姩閫夋嫨锛?
        if dataset in DATASET_STATS:
            stats = DATASET_STATS[dataset]
        else:
            stats = DATASET_STATS['CIFAR100']  # 榛樿浣跨敤 CIFAR100
        self.input_normalize = InputNormalize(mean=stats['mean'], std=stats['std'])

        # 鈽?寮哄埗涓庡煙鏁颁竴鑷达紙閬垮厤澶栭儴浼犲叆涓嶄竴鑷达級
        if int(fd_num_experts) != self.num_threat_domains:
            print(f"[INFO] fd_num_experts={fd_num_experts} -> num_domains={self.num_threat_domains}")
        self.fd_num_experts = int(self.num_threat_domains)
        self.band_stats_enabled = enable_band_stats

        # ===== BPDA 寮€鍏筹紙榛樿鍏抽棴锛屼笉褰卞搷鍘熼€昏緫锛?====
        self.use_bpda = False

        # Domain鍒嗙被鍣紙杈撳嚭 = num_domains锛?
        self.threat_domain_classifier = ThreatDomainClassifier(input_channels=3, num_domains=self.num_threat_domains)

        if source_names is None:
            source_names = ['PGD', 'ACE', 'SUB', 'STADV'] if num_sources == 4 else ['PGD', 'ACE', 'Hue', 'ReColorAdv',
                                                                                    'Light', 'UAA']

        # 鍩熺鐞嗗櫒锛堣仛绫荤皣鏁?= num_domains锛?
        self.threat_domain_diagnosis = ThreatDomainDiagnosis(
            num_sources=num_sources,
            num_threat_domains=self.num_threat_domains,
            source_names=source_names  # 闇€瑕佷粠妯″瀷鏋勯€犲嚱鏁颁紶鍏?
        )

        # initial conv 浣跨敤 FDConv
        # 鈽?鏍规嵁鏁版嵁闆嗚嚜鍔ㄩ€夋嫨 stem(涓庝富 main 涓€鑷?:
        #   ImageNet-like(Imagenette/ImageNet) 鈫?7x7 conv stride=2 + MaxPool 3x3 stride=2 (224鈫?6)
        #   CIFAR-like 鈫?3x3 conv stride=1 (32鈫?2)
        _imagenet_like = dataset in ('Imagenette', 'ImageNet', 'ImageNet-100', 'Restricted_ImageNet')
        if _imagenet_like:
            self.initial_conv = nn.Sequential(
                FCConv(3, 64, kernel_size=7, stride=2, padding=3, bias=False,
                       num_domains=self.num_threat_domains,
                       num_experts=self.fd_num_experts),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            self.initial_conv = nn.Sequential(
                FCConv(3, 64, 3, 1, 1, bias=False,
                       num_domains=self.num_threat_domains,
                       num_experts=self.fd_num_experts),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )

        layers = [3, 4, 6, 3]
        self.conv_block1 = self._make_layer(block, 64,  layers[0], stride=1, layer_name="conv_block1",
                                            use_fdconv_first=True)
        self.conv_block2 = self._make_layer(block, 128, layers[1], stride=2, layer_name="conv_block2",
                                            use_fdconv_first=True)
        self.conv_block3 = self._make_layer(block, 256, layers[2], stride=2, layer_name="conv_block3",
                                            use_fdconv_first=True)
        self.conv_block4 = self._make_layer(block, 512, layers[3], stride=2, layer_name="conv_block4",
                                            use_fdconv_first=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        import math
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # 鈽?缁欓獙璇?鏀诲嚮鐢ㄧ殑鎺ュ彛锛氬紑鍚?鍏抽棴 BPDA
    def set_bpda(self, enabled: bool = True):
        self.use_bpda = bool(enabled)

    def _make_layer(self, block, out_channels, blocks, stride=1, layer_name="", use_fdconv_first=False):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample,
                            layer_name + "_block0",
                            use_fdconv_conv1=use_fdconv_first,
                            num_threat_domains=self.num_threat_domains,
                            fd_num_experts=self.fd_num_experts))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels,
                                layer_name=layer_name + f"_block{i}",
                                use_fdconv_conv1=False,
                                num_threat_domains=self.num_threat_domains,
                                fd_num_experts=self.fd_num_experts))
        return nn.ModuleList(layers)

    def enable_band_stats(self, enabled=True):
        self.band_stats_enabled = enabled
        for m in self.modules():
            if isinstance(m, FCConv) and hasattr(m, "enable_band_stats"):
                m.enable_band_stats(enabled)
            if isinstance(m, FCConv) and hasattr(m, "enable_gate_stats"):
                m.enable_gate_stats(enabled)

    def reset_band_stats(self):
        if not self.band_stats_enabled: return
        for m in self.modules():
            if isinstance(m, FCConv) and hasattr(m, "reset_band_stats"):
                m.reset_band_stats()
            if isinstance(m, FCConv) and hasattr(m, "_reset_stats_storage"):
                m._reset_stats_storage()

    def update_domain_mappings(self, epoch, total_epochs):
        self.threat_domain_diagnosis.update_domain_mapping(epoch, total_epochs)
        return self.threat_domain_diagnosis.get_mapping_status()

    def get_all_mapping_statuses(self):
        return {"global_mapping": self.threat_domain_diagnosis.get_mapping_status()}

    def get_updated_layers_count(self):
        return 1 if self.threat_domain_diagnosis.mapping_ever_updated else 0, 1

    def get_domain_labels(self, source_ids):
        return self.threat_domain_diagnosis.get_domain_assignments(source_ids)

    def set_forced_domain_route(self, route_id=None):
        self.forced_domain_route_id = None if route_id is None else int(route_id)

    def set_uniform_domain_mix(self, enabled=False):
        self.uniform_domain_mix = bool(enabled)

    def count_frequency_convolutions(self):
        """Count FDConv and standard convolution modules."""
        stages = [3, 4, 6, 3]
        fdconv_count = 1 + len(stages)    # initial(1) + 鍥涗釜stage棣栧潡conv1(4)
        standard_count = 0
        # initial 鍚庣画娌℃湁 conv 璁℃暟
        for n in stages:
            # 姣忎釜 block 2 涓?conv锛岄鍧楃殑 conv1 琚?FDConv 鍗犵敤锛堜笉绠楁櫘閫氬嵎绉級
            standard_count += (2 * n - 1)
        # 鍔犱笂 3 涓?downsample 鐨?1x1
        standard_count += 3
        total = fdconv_count + standard_count
        print("妯″瀷鍗风Н缁熻:")
        print(f"  FDConv 鏁伴噺: {fdconv_count}")
        print(f"  鏍囧噯鍗风Н鏁伴噺: {standard_count}")
        print(f"  鎬诲嵎绉暟閲? {total}")
        print(f"  FDConv 姣斾緥: {fdconv_count / total:.1%}")
        return {'fdconv_count': fdconv_count, 'standard_count': standard_count,
                'total_count': total, 'fdconv_ratio': fdconv_count / total}

    @staticmethod
    def _denorm_imagenet(x):
        device, dtype = x.device, x.dtype
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
        return x * std + mean

    def extract_wavelet_features(self, img_pixel, source_ids):
        # 浠呯敤浜庢洿鏂板煙鍘熷瀷锛堝儚绱犲煙杈撳叆鏇寸ǔ锛?
        return None, self.threat_domain_diagnosis.update_features(img_pixel, source_ids)

    def forward(self, img, branch_idx=None, criterion=None, attack_num=5, flag=0,
                domain_ids=None, track_expert_freqs=False, skip_normalize=False,
                forced_domain_assignments=None):

        # 杈撳叆鏍囧噯鍖栵紙妯″瀷鍐呴儴鑷姩澶勭悊锛?
        if not skip_normalize:
            img = self.input_normalize(img)

        # 1) 棰勬祴 domain
        domain_logits = self.threat_domain_classifier(img)              # [B, D]
        _, predicted_domains = torch.max(domain_logits, dim=1)   # hard argmax

        # 璺敱鍩燂細璁粌鏃跺彲鐢?get_domain_labels(domain_ids)锛岄獙璇佹椂 None -> predicted_domains
        if forced_domain_assignments is None:
            forced_route_id = getattr(self, "forced_domain_route_id", None)
            if forced_route_id is not None:
                forced_domain_assignments = torch.full(
                    (img.size(0),), int(forced_route_id),
                    device=img.device, dtype=torch.long
                )

        uniform_domain_mix = bool(getattr(self, "uniform_domain_mix", False))

        if uniform_domain_mix:
            domain_assignments = torch.zeros(
                img.size(0), device=img.device, dtype=torch.long
            )
        elif forced_domain_assignments is not None:
            domain_assignments = forced_domain_assignments.to(
                device=img.device, dtype=torch.long
            ).clamp(0, self.num_threat_domains - 1)
        elif domain_ids is not None:
            domain_assignments = self.get_domain_labels(domain_ids)
        else:
            domain_assignments = predicted_domains

        # 鈽?2) BPDA 鐨?soft 鏉冮噸锛堜粎鐢ㄤ簬 backward surrogate锛沠orward 涓嶅彉锛?        router_weights = None
        router_weights = None
        soft_forward = False
        if uniform_domain_mix:
            router_weights = domain_logits.new_full(
                (img.size(0), self.num_threat_domains),
                1.0 / float(self.num_threat_domains)
            )
            soft_forward = True
        elif self.use_bpda and forced_domain_assignments is None:
            router_weights = F.softmax(domain_logits, dim=1)

        # 3) 鍒濆 FDConv锛堟妸 router_weights / bpda 浼犺繘鍘伙級
        if self.use_bpda and forced_domain_assignments is not None:
            router_weights = F.one_hot(
                domain_assignments,
                num_classes=self.num_threat_domains
            ).to(domain_logits.dtype)

        x_ini_out = self.initial_conv[0](
            img, domain_assignments,
            router_weights=router_weights,
            bpda=self.use_bpda,
            soft_forward=soft_forward
        )
        x = self.initial_conv[1:](x_ini_out[0])

        # 4) 涓诲共锛堟墍鏈?FDConv 鐨?block0 conv1 涔熶紶 router_weights / bpda锛?
        for layer in self.conv_block1:
            x = layer(x, domain_assignments, router_weights=router_weights, bpda=self.use_bpda, soft_forward=soft_forward)
        for layer in self.conv_block2:
            x = layer(x, domain_assignments, router_weights=router_weights, bpda=self.use_bpda, soft_forward=soft_forward)
        for layer in self.conv_block3:
            x = layer(x, domain_assignments, router_weights=router_weights, bpda=self.use_bpda, soft_forward=soft_forward)
        for layer in self.conv_block4:
            x = layer(x, domain_assignments, router_weights=router_weights, bpda=self.use_bpda, soft_forward=soft_forward)

        # 5) 鍒嗙被
        out = self.fc(self.avgpool(x).view(x.size(0), -1))

        merged_expert_freqs = (x_ini_out[1] if isinstance(x_ini_out, tuple) and len(x_ini_out) > 1 else
                               {f'domain_{i}': torch.zeros(1, self.fd_num_experts, device=img.device)
                                for i in range(self.num_threat_domains)})

        return out, merged_expert_freqs, domain_logits, predicted_domains
    # 淇濆瓨/鍔犺浇锛堝惈鍩熺鐞嗗櫒鐘舵€侊級
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        sd[prefix + 'threat_domain_diagnosis_state'] = self.threat_domain_diagnosis.get_state_dict()
        return sd

    def load_state_dict(self, state_dict, strict=True):
        gdm = state_dict.pop('threat_domain_diagnosis_state', None)
        missing, unexpected = super().load_state_dict(state_dict, strict=False)
        if gdm is not None:
            device = next(self.parameters()).device
            self.threat_domain_diagnosis.load_state_dict(gdm, device)
        if strict and (len(missing) > 0 or len(unexpected) > 0):
            raise RuntimeError(f"閿欒鍔犺浇鐘舵€佸瓧鍏? 缂哄皯閿? {missing}, 鎰忓閿? {unexpected}")
        return missing, unexpected


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
#  MobileViT 涓撶敤缁勪欢
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
try:
    from einops import rearrange
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False
    rearrange = None
    print("[Warning] einops not installed. MobileViTEncoder will not be available.")

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU(),
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=1, dim_head=32, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4,
                 use_fdconv_dw=False, num_domains=3, fd_num_experts=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.use_fdconv_dw = bool(use_fdconv_dw)
        self.num_domains = int(num_domains)
        self.fd_num_experts = int(fd_num_experts)
        self.pw1 = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )
        if self.use_fdconv_dw:
            self.dw = FCConv(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                             groups=hidden_dim, bias=False,
                             num_experts=self.fd_num_experts, num_domains=self.num_domains)
            self.dw_bn = nn.BatchNorm2d(hidden_dim)
            self.dw_act = nn.SiLU()
        else:
            self.dw = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
            self.dw_bn = nn.BatchNorm2d(hidden_dim)
            self.dw_act = nn.SiLU()
        self.pw2 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.pw2_bn = nn.BatchNorm2d(oup)

    def forward(self, x, domain_assignments=None, router_weights=None, bpda=False, soft_forward=False):
        out = self.pw1(x)
        if self.use_fdconv_dw:
            out = self.dw(
                out, domain_assignments,
                router_weights=router_weights,
                bpda=bpda,
                soft_forward=soft_forward
            )[0]
            out = self.dw_bn(out); out = self.dw_act(out)
        else:
            out = self.dw(out); out = self.dw_bn(out); out = self.dw_act(out)
        out = self.pw2(out)
        out = self.pw2_bn(out)
        if self.use_res_connect:
            return x + out
        else:
            return out

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.0,
                 use_fdconv_conv1=False, num_domains=3, fd_num_experts=3):
        super().__init__()
        self.ph, self.pw = patch_size
        # 鈽?TaFD 涓撳瀹氳皟(涓?ResNet "stage 澶撮儴婊″嵎绉笓瀹躲€佷富浣撳叡浜? 涓€鑷?:
        #   鎶?transformer 闃舵鐨勫叆鍙ｅ眬閮ㄨ〃寰佸嵎绉?conv1 鎹㈡垚 FDConv(婊″嵎绉? 鎸夊煙璺敱);
        #   attention/FeedForward 涓讳綋淇濇寔鍏变韩(涓嶅姩 transformer)銆?
        self.use_fdconv_conv1 = bool(use_fdconv_conv1)
        self.num_domains = int(num_domains)
        self.fd_num_experts = int(fd_num_experts)
        if self.use_fdconv_conv1:
            self.conv1_fd = FCConv(channel, channel, kernel_size, stride=1, padding=1, groups=1, bias=False,
                                   num_experts=self.fd_num_experts, num_domains=self.num_domains)
            self.conv1_bn = nn.BatchNorm2d(channel)
            self.conv1_act = nn.SiLU()
        else:
            self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = Transformer(dim, depth, 1, 32, mlp_dim, dropout)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    def forward(self, x, domain_assignments=None, router_weights=None, bpda=False, soft_forward=False):
        y = x.clone()
        if self.use_fdconv_conv1:
            x = self.conv1_fd(
                x, domain_assignments,
                router_weights=router_weights,
                bpda=bpda,
                soft_forward=soft_forward
            )[0]
            x = self.conv1_bn(x); x = self.conv1_act(x)
        else:
            x = self.conv1(x)
        x = self.conv2(x)
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                      h=h // self.ph, w=w // self.pw, ph=self.ph, pw=self.pw)
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
#  MobileViTEncoder
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class MobileViTEncoder(nn.Module):
    def __init__(self, size=32, num_classes=10,
                 num_sources=6, num_domains=3, fd_num_experts=3,
                 expansion=3, kernel_size=3, patch_size=(2, 2),
                 fdconv_stage_indices=None, replace_mode='dw',
                 enable_band_stats=False, dataset='CIFAR100', source_names=None,
                 mvit_fdconv=False):
        super().__init__()
        self.mvit_fdconv = bool(mvit_fdconv)
        ih = iw = size
        assert ih % patch_size[0] == 0 and iw % patch_size[1] == 0

        self.num_sources = int(num_sources)
        self.num_threat_domains = int(num_domains)
        if int(fd_num_experts) != self.num_threat_domains:
            print(f"[INFO] fd_num_experts={fd_num_experts} -> num_domains={self.num_threat_domains}")
        self.fd_num_experts = int(self.num_threat_domains)

        # 杈撳叆鏍囧噯鍖栧眰锛堟牴鎹暟鎹泦鑷姩閫夋嫨锛?
        if dataset in DATASET_STATS:
            stats = DATASET_STATS[dataset]
        else:
            stats = DATASET_STATS['CIFAR100']  # 榛樿浣跨敤 CIFAR100
        self.input_normalize = InputNormalize(mean=stats['mean'], std=stats['std'])

        self.band_stats_enabled = enable_band_stats
        self.replace_mode = replace_mode

        # BPDA 寮€鍏?
        self.use_bpda = False

        self.threat_domain_classifier = ThreatDomainClassifier(input_channels=3, num_domains=self.num_threat_domains)

        # 涓?ResNetEncoder 淇濇寔涓€鑷达細鏍规嵁 num_sources 璁剧疆鏀诲嚮鍚嶇О
        if source_names is None:
            source_names = ['PGD', 'ACE', 'SUB', 'STADV'] if num_sources == 4 else ['PGD', 'ACE', 'Hue', 'ReColorAdv', 'Light', 'UAA']
        self.threat_domain_diagnosis = ThreatDomainDiagnosis(
            num_sources=num_sources,
            num_threat_domains=self.num_threat_domains,
            source_names=source_names
        )

        dims = [144, 192, 240]
        channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
        L = [2, 4, 3]

        self.use_fdconv_stem = True
        if self.use_fdconv_stem:
            self.conv1 = nn.Sequential(
                FCConv(3, channels[0], kernel_size=3, stride=2, padding=1, groups=1, bias=False,
                       num_experts=self.fd_num_experts, num_domains=self.num_threat_domains),
                nn.BatchNorm2d(channels[0]),
                nn.SiLU()
            )
        else:
            self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        if fdconv_stage_indices is None:
            fdconv_stage_indices = [0, 4, 5, 6]
        self.fdconv_stage_indices = set(fdconv_stage_indices)

        self.mv2 = nn.ModuleList([])
        def _use_fd_dw(idx):
            return (idx in self.fdconv_stage_indices) and (self.replace_mode == 'dw')

        # 鈽?224 杈撳叆(Imagenette/ImageNet) 閫傞厤(涓庝富 main 涓€鑷?: mv2[0] stride=2, 鏃╂湡澶氫竴娆′笅閲囨牱,
        #   璁╄繘 mvit transformer 鐨勭壒寰佸浘浠?56脳56 闄嶅埌 28脳28(attention 閲忕骇闄?16 鍊?; 32杈撳叆(CIFAR)琛屼负涓嶅彉銆?
        _imagenet_like = dataset in ('Imagenette', 'ImageNet', 'ImageNet-100', 'Restricted_ImageNet')
        _mv2_0_stride = 2 if _imagenet_like else 1

        self.mv2.append(MV2Block(channels[0], channels[1], _mv2_0_stride, expansion, use_fdconv_dw=_use_fd_dw(0),
                                 num_domains=self.num_threat_domains, fd_num_experts=self.fd_num_experts))
        self.mv2.append(MV2Block(channels[1], channels[2], 1, expansion, use_fdconv_dw=_use_fd_dw(1),
                                 num_domains=self.num_threat_domains, fd_num_experts=self.fd_num_experts))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion, use_fdconv_dw=_use_fd_dw(2),
                                 num_domains=self.num_threat_domains, fd_num_experts=self.fd_num_experts))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion, use_fdconv_dw=_use_fd_dw(3),
                                 num_domains=self.num_threat_domains, fd_num_experts=self.fd_num_experts))
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion, use_fdconv_dw=_use_fd_dw(4),
                                 num_domains=self.num_threat_domains, fd_num_experts=self.fd_num_experts))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion, use_fdconv_dw=_use_fd_dw(5),
                                 num_domains=self.num_threat_domains, fd_num_experts=self.fd_num_experts))
        self.mv2.append(MV2Block(channels[7], channels[8], 1, expansion, use_fdconv_dw=_use_fd_dw(6),
                                 num_domains=self.num_threat_domains, fd_num_experts=self.fd_num_experts))

        self.mvit = nn.ModuleList([])
        _mv_kw = dict(use_fdconv_conv1=self.mvit_fdconv,
                      num_domains=self.num_threat_domains, fd_num_experts=self.fd_num_experts)
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2), **_mv_kw))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4), **_mv_kw))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4), **_mv_kw))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])
        # 鈽?鑷€傚簲姹犲寲(涓庝富 main 涓€鑷?: 32杈撳叆涓嬬瓑浠?AvgPool2d(4,1); 224/Imagenette 涓嬭兘鎶婁换鎰忕┖闂村昂瀵糕啋1脳1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()

    def set_bpda(self, enabled: bool = True):
        self.use_bpda = bool(enabled)

    def enable_band_stats(self, enabled=True):
        self.band_stats_enabled = enabled
        for m in self.modules():
            if isinstance(m, FCConv) and hasattr(m, "enable_band_stats"):
                m.enable_band_stats(enabled)

    def reset_band_stats(self):
        if not self.band_stats_enabled: return
        for m in self.modules():
            if isinstance(m, FCConv) and hasattr(m, "reset_band_stats"):
                m.reset_band_stats()

    def update_domain_mappings(self, epoch, total_epochs):
        self.threat_domain_diagnosis.update_domain_mapping(epoch, total_epochs)
        return self.threat_domain_diagnosis.get_mapping_status()

    def get_all_mapping_statuses(self):
        return {"global_mapping": self.threat_domain_diagnosis.get_mapping_status()}

    def get_updated_layers_count(self):
        return 1 if self.threat_domain_diagnosis.mapping_ever_updated else 0, 1

    def get_domain_labels(self, source_ids):
        return self.threat_domain_diagnosis.get_domain_assignments(source_ids)

    def set_forced_domain_route(self, route_id=None):
        self.forced_domain_route_id = None if route_id is None else int(route_id)

    def set_uniform_domain_mix(self, enabled=False):
        self.uniform_domain_mix = bool(enabled)

    def count_frequency_convolutions(self):
        fdconv_count = 0
        conv2d_count = 0
        for name, m in self.named_modules():
            if isinstance(m, FCConv):
                fdconv_count += 1
            elif isinstance(m, nn.Conv2d):
                if ('expert_conv' in name) or ('residual_conv' in name):
                    continue
                conv2d_count += 1
        total = fdconv_count + conv2d_count
        print("妯″瀷鍗风Н缁熻 (鎺掗櫎 FDConv 鍐呴儴 expert/residual):")
        print(f"  FDConv 鏁伴噺: {fdconv_count}")
        print(f"  鏍囧噯 Conv2d 鏁伴噺: {conv2d_count}")
        print(f"  鎬诲嵎绉暟閲? {total}")
        print(f"  FDConv 姣斾緥: {fdconv_count / total:.1%}" if total > 0 else "  FDConv 姣斾緥: N/A")
        return {'fdconv_count': fdconv_count, 'standard_count': conv2d_count,
                'total_count': total, 'fdconv_ratio': (fdconv_count / total if total > 0 else 0.0)}

    @staticmethod
    def _denorm_imagenet(x):
        device, dtype = x.device, x.dtype
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
        return x * std + mean

    def extract_wavelet_features(self, img_pixel, source_ids):
        return None, self.threat_domain_diagnosis.update_features(img_pixel, source_ids)

    def forward(self, img, branch_idx=None, criterion=None, attack_num=5, flag=0,
                domain_ids=None, track_expert_freqs=False, skip_normalize=False,
                forced_domain_assignments=None):

        # 杈撳叆鏍囧噯鍖栵紙妯″瀷鍐呴儴鑷姩澶勭悊锛?
        if not skip_normalize:
            img = self.input_normalize(img)

        domain_logits = self.threat_domain_classifier(img)
        _, predicted_domains = torch.max(domain_logits, dim=1)

        # 璺敱鍩燂細璁粌鏃剁敤鑱氱被鏄犲皠锛岄獙璇佹椂鐢ㄩ娴嬪煙
        if forced_domain_assignments is None:
            forced_route_id = getattr(self, "forced_domain_route_id", None)
            if forced_route_id is not None:
                forced_domain_assignments = torch.full(
                    (img.size(0),), int(forced_route_id),
                    device=img.device, dtype=torch.long
                )

        uniform_domain_mix = bool(getattr(self, "uniform_domain_mix", False))

        if uniform_domain_mix:
            domain_assignments = torch.zeros(
                img.size(0), device=img.device, dtype=torch.long
            )
        elif forced_domain_assignments is not None:
            domain_assignments = forced_domain_assignments.to(
                device=img.device, dtype=torch.long
            ).clamp(0, self.num_threat_domains - 1)
        elif domain_ids is not None:
            domain_assignments = self.get_domain_labels(domain_ids)
        else:
            domain_assignments = predicted_domains

        router_weights = None
        router_weights = None
        soft_forward = False
        if uniform_domain_mix:
            router_weights = domain_logits.new_full(
                (img.size(0), self.num_threat_domains),
                1.0 / float(self.num_threat_domains)
            )
            soft_forward = True
        elif self.use_bpda and forced_domain_assignments is None:
            router_weights = F.softmax(domain_logits, dim=1)

        if self.use_bpda and forced_domain_assignments is not None:
            router_weights = F.one_hot(
                domain_assignments,
                num_classes=self.num_threat_domains
            ).to(domain_logits.dtype)

        if self.use_fdconv_stem:
            stem_out = self.conv1[0](img, domain_assignments,
                                      router_weights=router_weights, bpda=self.use_bpda,
                                      soft_forward=soft_forward)
            x = self.conv1[1:](stem_out[0])
        else:
            x = self.conv1(img)

        for i, blk in enumerate(self.mv2):
            x = blk(x, domain_assignments, router_weights=router_weights,
                    bpda=self.use_bpda, soft_forward=soft_forward)
            if i == 4:
                x = self.mvit[0](x, domain_assignments, router_weights=router_weights,
                                 bpda=self.use_bpda, soft_forward=soft_forward)
            if i == 5:
                x = self.mvit[1](x, domain_assignments, router_weights=router_weights,
                                 bpda=self.use_bpda, soft_forward=soft_forward)
            if i == 6:
                x = self.mvit[2](x, domain_assignments, router_weights=router_weights,
                                 bpda=self.use_bpda, soft_forward=soft_forward)

        x_feature = self.conv2(x)
        pooled = self.pool(x_feature).view(-1, x_feature.shape[1])
        out = self.fc(pooled)

        # 涓?ResNetEncoder 淇濇寔涓€鑷达細杩斿洖 expert_freqs锛堜粠 stem FDConv 鑾峰彇锛?
        merged_expert_freqs = (stem_out[1] if self.use_fdconv_stem and isinstance(stem_out, tuple) and len(stem_out) > 1 else
                               {f'domain_{i}': torch.zeros(1, self.fd_num_experts, device=img.device)
                                for i in range(self.num_threat_domains)})

        return out, merged_expert_freqs, domain_logits, predicted_domains

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        sd[prefix + 'threat_domain_diagnosis_state'] = self.threat_domain_diagnosis.get_state_dict()
        return sd

    def load_state_dict(self, state_dict, strict=True):
        gdm = state_dict.pop('threat_domain_diagnosis_state', None)
        missing, unexpected = super().load_state_dict(state_dict, strict=False)
        if gdm is not None:
            device = next(self.parameters()).device
            self.threat_domain_diagnosis.load_state_dict(gdm, device)
        if strict and (len(missing) > 0 or len(unexpected) > 0):
            raise RuntimeError(f"閿欒鍔犺浇鐘舵€佸瓧鍏? 缂哄皯閿? {missing}, 鎰忓閿? {unexpected}")
        return missing, unexpected


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
#  宸ュ巶鍑芥暟 & 鍏煎鍒悕
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
def create_encoder(backbone='resnet', dataset=None, **kwargs):
    """Create the requested encoder backbone."""
    kwargs['dataset'] = dataset
    if backbone == 'resnet':
        return ResNetEncoder(**kwargs)
    elif backbone == 'mobilevit':
        if not EINOPS_AVAILABLE:
            raise ImportError("MobileViTEncoder requires 'einops' package. Install with: pip install einops")
        return MobileViTEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose 'resnet' or 'mobilevit'.")


# 鍏煎鎬у埆鍚?
Encoder = ResNetEncoder

