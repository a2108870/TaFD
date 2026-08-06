# -*- coding: utf-8 -*-
"""TaFD ablation without frequency decoupling."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.frequency_conditional_convolution_without_frequency_decoupling import FrequencyConditionalConvolution


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler





DATASET_STATS = {
    'CIFAR10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
    'CIFAR100': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
    'TinyImageNet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
    'Imagenette': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
    'ImageNet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
}


class InputNormalize(nn.Module):
    """Normalize input tensors with registered dataset statistics."""
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std






class SpectralFeatureExtractor:
    """Extract the 15-dimensional spectral descriptor used for threat-domain diagnosis."""

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
        """Extract spectral descriptors from a batch of pixel-space images."""
        B, C, H, W = img_pixel.shape
        img_fft = torch.fft.rfft2(img_pixel.float(), norm='ortho')
        self._prepare_cache(H, img_fft.size(-1), img_pixel.device, img_fft.real.dtype)

        ycbcr_fft = self._rgb_to_ycbcr_fft(img_fft)
        # energy = (ycbcr_fft.abs() ** 2)
        energy = ycbcr_fft.real.pow(2) + ycbcr_fft.imag.pow(2)


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


        cb_b = band_e[:, 1, :]
        cr_b = band_e[:, 2, :]
        chroma_dominance = (cr_b - cb_b) / (cr_b + cb_b + self.eps)  # [B, 3]

        return torch.cat([feats_base, chroma_dominance], dim=1)  # [B, 15]


# ------------------------------------------------------------------------------
#  Threat-Domain Classifier (Lightweight CNN for domain prediction)
# ------------------------------------------------------------------------------
class ThreatDomainClassifier(nn.Module):
    """Lightweight threat-domain classifier G."""

    def __init__(self, input_channels=3, num_threat_domains=2):
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
            nn.Linear(64, num_threat_domains)
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
        diagnosis_logits = self.classifier(features)
        return diagnosis_logits


# ------------------------------------------------------------------------------
#  Threat-Domain Diagnosis (Dynamic clustering and alignment)
# ------------------------------------------------------------------------------
class ThreatDomainDiagnosis:
    """Threat-Domain Diagnosis module for spectral prototype learning and clustering."""

    def __init__(self, num_attack_sources=7, num_threat_domains=2, feature_ids=None, attack_names=None):
        self.num_attack_sources = num_attack_sources
        self.num_threat_domains = num_threat_domains

        if attack_names is not None:
            self.attack_names = attack_names
        else:
            self.attack_names = ['APGD_Linf', 'APGD_L2', 'ACE', 'HSVAdv', 'ReColorAdv', 'ALA', 'RetouchUAA']

        self.source_features = {}
        self.source_counts = {}
        self.source_to_domain = {i: 0 for i in range(num_attack_sources)}

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

    def compute_spectral_features(self, img_pixel_batch):
        if img_pixel_batch.size(0) == 0:
            return torch.empty(0, 0, device=img_pixel_batch.device)
        return self._compute_features(img_pixel_batch)

    def update_spectral_prototypes(self, img_pixel_batch, attack_source_ids):
        with torch.no_grad():
            self.update_counter += 1
            self.feature_record_count += 1

            feats = self.compute_spectral_features(img_pixel_batch)
            if feats.numel() == 0:
                return False

            for d in torch.unique(attack_source_ids):
                d_int = int(d.item())
                mask = (attack_source_ids == d)
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

    def update_threat_domain_assignments(self, epoch=None, total_epochs=None):
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

    def get_threat_domain_indices(self, attack_source_ids):
        device = attack_source_ids.device
        max_source_id = max(self.source_to_domain.keys()) if self.source_to_domain else 0
        mapping = torch.zeros(max_source_id + 1, dtype=torch.long, device=device)
        for s, d in self.source_to_domain.items():
            mapping[s] = max(0, int(d))
        return mapping[torch.clamp(attack_source_ids, 0, max_source_id)]

    def get_assignment_status(self):
        return {self.attack_names[s]: f"ThreatDomain-{self.source_to_domain.get(s, 0)}"
                for s in range(min(len(self.attack_names), self.num_attack_sources))}

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


        saved_features = state_dict.get('source_features', None)
        if saved_features is not None:
            self.source_features = {}
            for k, v in saved_features.items():

                if device is not None:
                    self.source_features[int(k)] = v.to(device)
                else:
                    self.source_features[int(k)] = v
        else:
            self.source_features = {}





def conv3x3(in_planes, out_planes, stride=1, layer_name="",
            use_fc_conv=False, num_threat_domains=2, num_frequency_experts=2):
    if use_fc_conv:
        return FrequencyConditionalConvolution(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            num_threat_domains=num_threat_domains,
            num_frequency_experts=num_frequency_experts
        )
    else:
        return nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None,
                 layer_name="", use_fc_conv_conv1=False,
                 num_threat_domains=2, num_frequency_experts=2):
        super(BasicBlock, self).__init__()
        # conv1: use FrequencyConditionalConvolution only when use_fc_conv_conv1=True
        if use_fc_conv_conv1:
            self.conv1_is_fd = True
            self.conv1 = conv3x3(in_planes, planes, stride, layer_name + "_conv1",
                                 use_fc_conv=True, num_threat_domains=num_threat_domains,
                                 num_frequency_experts=num_frequency_experts)
        else:
            self.conv1_is_fd = False
            self.conv1 = conv3x3(in_planes, planes, stride, layer_name + "_conv1",
                                 use_fc_conv=False)


        self.conv2 = conv3x3(planes, planes, 1, layer_name + "_conv2",
                             use_fc_conv=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x, threat_domain_indices=None, dispatch_weights=None, bpda=False):
        identity = x
        if self.conv1_is_fd:

            out = self.conv1(
                x, threat_domain_indices,
                dispatch_weights=dispatch_weights,
                bpda=bpda
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





class TaFDResNet34(nn.Module):
    def __init__(self, num_attack_sources=7, num_threat_domains=2, num_frequency_experts=2,
                 enable_band_stats=False, diagnosis_feature_ids=None, num_classes=10,
                 dataset='CIFAR100', attack_names=None):
        super(TaFDResNet34, self).__init__()
        block = BasicBlock
        self.in_channels = 64
        self.num_attack_sources = num_attack_sources
        self.num_threat_domains = int(num_threat_domains)


        if dataset in DATASET_STATS:
            stats = DATASET_STATS[dataset]
        else:
            stats = DATASET_STATS['CIFAR100']
        self.input_normalize = InputNormalize(mean=stats['mean'], std=stats['std'])


        if int(num_frequency_experts) != self.num_threat_domains:
            print(f"[INFO] num_frequency_experts={num_frequency_experts} -> num_threat_domains={self.num_threat_domains}")
        self.num_frequency_experts = int(self.num_threat_domains)
        self.band_stats_enabled = enable_band_stats


        self.use_bpda = False


        self.threat_domain_classifier = ThreatDomainClassifier(input_channels=3, num_threat_domains=self.num_threat_domains)

        if attack_names is None:
            attack_names = (
                ['APGD_Linf', 'APGD_L2', 'ACE', 'GPGD', 'StAdv']
                if num_attack_sources == 5
                else ['APGD_Linf', 'APGD_L2', 'ACE', 'HSVAdv', 'ReColorAdv', 'ALA', 'RetouchUAA']
            )


        self.threat_domain_diagnosis = ThreatDomainDiagnosis(
            num_attack_sources=num_attack_sources,
            num_threat_domains=self.num_threat_domains,
            attack_names=attack_names
        )


        self.initial_conv = nn.Sequential(
            FrequencyConditionalConvolution(3, 64, 3, 1, 1, bias=False,
                   num_threat_domains=self.num_threat_domains,
                   num_frequency_experts=self.num_frequency_experts),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        layers = [3, 4, 6, 3]
        self.conv_block1 = self._make_layer(block, 64,  layers[0], stride=1, layer_name="conv_block1",
                                            use_fc_conv_first=True)
        self.conv_block2 = self._make_layer(block, 128, layers[1], stride=2, layer_name="conv_block2",
                                            use_fc_conv_first=True)
        self.conv_block3 = self._make_layer(block, 256, layers[2], stride=2, layer_name="conv_block3",
                                            use_fc_conv_first=True)
        self.conv_block4 = self._make_layer(block, 512, layers[3], stride=2, layer_name="conv_block4",
                                            use_fc_conv_first=True)

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


    def set_bpda(self, enabled: bool = True):
        self.use_bpda = bool(enabled)

    def _make_layer(self, block, out_channels, blocks, stride=1, layer_name="", use_fc_conv_first=False):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample,
                            layer_name + "_block0",
                            use_fc_conv_conv1=use_fc_conv_first,
                            num_threat_domains=self.num_threat_domains,
                            num_frequency_experts=self.num_frequency_experts))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels,
                                layer_name=layer_name + f"_block{i}",
                                use_fc_conv_conv1=False,
                                num_threat_domains=self.num_threat_domains,
                                num_frequency_experts=self.num_frequency_experts))
        return nn.ModuleList(layers)

    def enable_band_stats(self, enabled=True):
        self.band_stats_enabled = enabled
        for m in self.modules():
            if isinstance(m, FrequencyConditionalConvolution) and hasattr(m, "enable_band_stats"):
                m.enable_band_stats(enabled)

    def reset_band_stats(self):
        if not self.band_stats_enabled: return
        for m in self.modules():
            if isinstance(m, FrequencyConditionalConvolution) and hasattr(m, "reset_band_stats"):
                m.reset_band_stats()
            if isinstance(m, FrequencyConditionalConvolution) and hasattr(m, "_reset_stats_storage"):
                m._reset_stats_storage()

    def update_threat_domain_assignments(self, epoch, total_epochs):
        self.threat_domain_diagnosis.update_threat_domain_assignments(epoch, total_epochs)
        return self.threat_domain_diagnosis.get_assignment_status()

    def get_threat_domain_assignment_status(self):
        return self.threat_domain_diagnosis.get_assignment_status()

    def get_updated_layers_count(self):
        return 1 if self.threat_domain_diagnosis.mapping_ever_updated else 0, 1

    def get_threat_domain_indices(self, attack_source_ids):
        return self.threat_domain_diagnosis.get_threat_domain_indices(attack_source_ids)

    def count_frequency_convolutions(self):
        """Report FC-Conv and standard-convolution counts for the ResNet topology."""
        stages = [3, 4, 6, 3]
        fc_conv_count = 1 + len(stages)
        standard_count = 0

        for n in stages:

            standard_count += (2 * n - 1)

        standard_count += 3
        total = fc_conv_count + standard_count
        print("Convolution summary:")
        print(f"  FC-Conv layers: {fc_conv_count}")
        print(f"  Standard convolution layers: {standard_count}")
        print(f"  Total convolution layers: {total}")
        print(f"  FC-Conv ratio: {fc_conv_count / total:.1%}")
        return {'fc_conv_count': fc_conv_count, 'standard_count': standard_count,
                'total_count': total, 'fc_conv_ratio': fc_conv_count / total}

    @staticmethod
    def _denorm_imagenet(x):
        device, dtype = x.device, x.dtype
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
        return x * std + mean

    def update_spectral_prototypes(self, img_pixel, attack_source_ids):

        return None, self.threat_domain_diagnosis.update_spectral_prototypes(img_pixel, attack_source_ids)

    def forward(self, img, attack_source_ids=None, skip_normalize=False):


        if not skip_normalize:
            img = self.input_normalize(img)


        diagnosis_logits = self.threat_domain_classifier(img)              # [B, D]
        _, predicted_threat_domains = torch.max(diagnosis_logits, dim=1)   # hard argmax


        if attack_source_ids is not None:
            threat_domain_indices = self.get_threat_domain_indices(attack_source_ids)
        else:
            threat_domain_indices = predicted_threat_domains


        dispatch_weights = None
        if self.use_bpda:
            dispatch_weights = F.softmax(diagnosis_logits, dim=1)


        x_ini_out = self.initial_conv[0](
            img, threat_domain_indices,
            dispatch_weights=dispatch_weights,
            bpda=self.use_bpda
        )
        x = self.initial_conv[1:](x_ini_out[0])


        for layer in self.conv_block1:
            x = layer(x, threat_domain_indices, dispatch_weights=dispatch_weights, bpda=self.use_bpda)
        for layer in self.conv_block2:
            x = layer(x, threat_domain_indices, dispatch_weights=dispatch_weights, bpda=self.use_bpda)
        for layer in self.conv_block3:
            x = layer(x, threat_domain_indices, dispatch_weights=dispatch_weights, bpda=self.use_bpda)
        for layer in self.conv_block4:
            x = layer(x, threat_domain_indices, dispatch_weights=dispatch_weights, bpda=self.use_bpda)


        out = self.fc(self.avgpool(x).view(x.size(0), -1))

        merged_expert_freqs = (x_ini_out[1] if isinstance(x_ini_out, tuple) and len(x_ini_out) > 1 else
                               {f'domain_{i}': torch.zeros(1, self.num_frequency_experts, device=img.device)
                                for i in range(self.num_threat_domains)})

        return out, merged_expert_freqs, diagnosis_logits, predicted_threat_domains

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = super().state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )
        sd[prefix + 'threat_domain_diagnosis_state'] = self.threat_domain_diagnosis.get_state_dict()
        return sd

    def load_state_dict(self, state_dict, strict=True):
        state_dict = state_dict.copy()
        diagnosis_state = state_dict.pop('threat_domain_diagnosis_state', None)
        missing, unexpected = super().load_state_dict(state_dict, strict=False)
        if diagnosis_state is not None:
            device = next(self.parameters()).device
            self.threat_domain_diagnosis.load_state_dict(diagnosis_state, device)
        if strict and (len(missing) > 0 or len(unexpected) > 0):
            raise RuntimeError(f"State-dict mismatch: missing keys={missing}, unexpected keys={unexpected}")
        return missing, unexpected





try:
    from einops import rearrange
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False
    rearrange = None
    print("[Warning] einops not installed. TaFDMobileViT will not be available.")

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
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
                 use_fc_conv_dw=False, num_threat_domains=2, num_frequency_experts=2):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.use_fc_conv_dw = bool(use_fc_conv_dw)
        self.num_threat_domains = int(num_threat_domains)
        self.num_frequency_experts = int(num_frequency_experts)
        self.pw1 = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )
        if self.use_fc_conv_dw:
            self.dw = FrequencyConditionalConvolution(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                             groups=hidden_dim, bias=False,
                             num_frequency_experts=self.num_frequency_experts, num_threat_domains=self.num_threat_domains)
            self.dw_bn = nn.BatchNorm2d(hidden_dim)
            self.dw_act = nn.SiLU()
        else:
            self.dw = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
            self.dw_bn = nn.BatchNorm2d(hidden_dim)
            self.dw_act = nn.SiLU()
        self.pw2 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.pw2_bn = nn.BatchNorm2d(oup)

    def forward(self, x, threat_domain_indices=None, dispatch_weights=None, bpda=False):
        out = self.pw1(x)
        if self.use_fc_conv_dw:
            out = self.dw(out, threat_domain_indices, dispatch_weights=dispatch_weights, bpda=bpda)[0]
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
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.0):
        super().__init__()
        self.ph, self.pw = patch_size
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = Transformer(dim, depth, 1, 32, mlp_dim, dropout)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    def forward(self, x):
        y = x.clone()
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



#  TaFDMobileViT

class TaFDMobileViT(nn.Module):
    def __init__(self, size=32, num_classes=10,
                 num_attack_sources=7, num_threat_domains=2, num_frequency_experts=2,
                 expansion=3, kernel_size=3, patch_size=(2, 2),
                 fc_conv_stage_indices=None, replace_mode='dw',
                 enable_band_stats=False, dataset='CIFAR100', attack_names=None):
        super().__init__()
        ih = iw = size
        assert ih % patch_size[0] == 0 and iw % patch_size[1] == 0

        self.num_attack_sources = int(num_attack_sources)
        self.num_threat_domains = int(num_threat_domains)
        if int(num_frequency_experts) != self.num_threat_domains:
            print(f"[INFO] num_frequency_experts={num_frequency_experts} -> num_threat_domains={self.num_threat_domains}")
        self.num_frequency_experts = int(self.num_threat_domains)


        if dataset in DATASET_STATS:
            stats = DATASET_STATS[dataset]
        else:
            stats = DATASET_STATS['CIFAR100']
        self.input_normalize = InputNormalize(mean=stats['mean'], std=stats['std'])

        self.band_stats_enabled = enable_band_stats
        self.replace_mode = replace_mode


        self.use_bpda = False

        self.threat_domain_classifier = ThreatDomainClassifier(input_channels=3, num_threat_domains=self.num_threat_domains)


        if attack_names is None:
            attack_names = (
                ['APGD_Linf', 'APGD_L2', 'ACE', 'GPGD', 'StAdv']
                if num_attack_sources == 5
                else ['APGD_Linf', 'APGD_L2', 'ACE', 'HSVAdv', 'ReColorAdv', 'ALA', 'RetouchUAA']
            )
        self.threat_domain_diagnosis = ThreatDomainDiagnosis(
            num_attack_sources=num_attack_sources,
            num_threat_domains=self.num_threat_domains,
            attack_names=attack_names
        )

        dims = [144, 192, 240]
        channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
        L = [2, 4, 3]

        self.use_fc_conv_stem = True
        if self.use_fc_conv_stem:
            self.conv1 = nn.Sequential(
                FrequencyConditionalConvolution(3, channels[0], kernel_size=3, stride=2, padding=1, groups=1, bias=False,
                       num_frequency_experts=self.num_frequency_experts, num_threat_domains=self.num_threat_domains),
                nn.BatchNorm2d(channels[0]),
                nn.SiLU()
            )
        else:
            self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        if fc_conv_stage_indices is None:
            fc_conv_stage_indices = [0, 4, 5, 6]
        self.fc_conv_stage_indices = set(fc_conv_stage_indices)

        self.mv2 = nn.ModuleList([])
        def _use_fd_dw(idx):
            return (idx in self.fc_conv_stage_indices) and (self.replace_mode == 'dw')

        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion, use_fc_conv_dw=_use_fd_dw(0),
                                 num_threat_domains=self.num_threat_domains, num_frequency_experts=self.num_frequency_experts))
        self.mv2.append(MV2Block(channels[1], channels[2], 1, expansion, use_fc_conv_dw=_use_fd_dw(1),
                                 num_threat_domains=self.num_threat_domains, num_frequency_experts=self.num_frequency_experts))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion, use_fc_conv_dw=_use_fd_dw(2),
                                 num_threat_domains=self.num_threat_domains, num_frequency_experts=self.num_frequency_experts))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion, use_fc_conv_dw=_use_fd_dw(3),
                                 num_threat_domains=self.num_threat_domains, num_frequency_experts=self.num_frequency_experts))
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion, use_fc_conv_dw=_use_fd_dw(4),
                                 num_threat_domains=self.num_threat_domains, num_frequency_experts=self.num_frequency_experts))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion, use_fc_conv_dw=_use_fd_dw(5),
                                 num_threat_domains=self.num_threat_domains, num_frequency_experts=self.num_frequency_experts))
        self.mv2.append(MV2Block(channels[7], channels[8], 1, expansion, use_fc_conv_dw=_use_fd_dw(6),
                                 num_threat_domains=self.num_threat_domains, num_frequency_experts=self.num_frequency_experts))

        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])
        self.pool = nn.AvgPool2d(ih // 8, 1)
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
            if isinstance(m, FrequencyConditionalConvolution) and hasattr(m, "enable_band_stats"):
                m.enable_band_stats(enabled)

    def reset_band_stats(self):
        if not self.band_stats_enabled: return
        for m in self.modules():
            if isinstance(m, FrequencyConditionalConvolution) and hasattr(m, "reset_band_stats"):
                m.reset_band_stats()

    def update_threat_domain_assignments(self, epoch, total_epochs):
        self.threat_domain_diagnosis.update_threat_domain_assignments(epoch, total_epochs)
        return self.threat_domain_diagnosis.get_assignment_status()

    def get_threat_domain_assignment_status(self):
        return self.threat_domain_diagnosis.get_assignment_status()

    def get_updated_layers_count(self):
        return 1 if self.threat_domain_diagnosis.mapping_ever_updated else 0, 1

    def get_threat_domain_indices(self, attack_source_ids):
        return self.threat_domain_diagnosis.get_threat_domain_indices(attack_source_ids)

    def count_frequency_convolutions(self):
        fc_conv_count = 0
        conv2d_count = 0
        for name, m in self.named_modules():
            if isinstance(m, FrequencyConditionalConvolution):
                fc_conv_count += 1
            elif isinstance(m, nn.Conv2d):
                if ('expert_conv' in name) or ('residual_conv' in name):
                    continue
                conv2d_count += 1
        total = fc_conv_count + conv2d_count
        print("Convolution summary (excluding FC-Conv expert and residual branches):")
        print(f"  FC-Conv layers: {fc_conv_count}")
        print(f"  Standard Conv2d layers: {conv2d_count}")
        print(f"  Total convolution layers: {total}")
        print(f"  FC-Conv ratio: {fc_conv_count / total:.1%}" if total > 0 else "  FC-Conv ratio: N/A")
        return {'fc_conv_count': fc_conv_count, 'standard_count': conv2d_count,
                'total_count': total, 'fc_conv_ratio': (fc_conv_count / total if total > 0 else 0.0)}

    @staticmethod
    def _denorm_imagenet(x):
        device, dtype = x.device, x.dtype
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
        return x * std + mean

    def update_spectral_prototypes(self, img_pixel, attack_source_ids):
        return None, self.threat_domain_diagnosis.update_spectral_prototypes(img_pixel, attack_source_ids)

    def forward(self, img, attack_source_ids=None, skip_normalize=False):


        if not skip_normalize:
            img = self.input_normalize(img)

        diagnosis_logits = self.threat_domain_classifier(img)
        _, predicted_threat_domains = torch.max(diagnosis_logits, dim=1)


        if attack_source_ids is not None:
            threat_domain_indices = self.get_threat_domain_indices(attack_source_ids)
        else:
            threat_domain_indices = predicted_threat_domains

        dispatch_weights = None
        if self.use_bpda:
            dispatch_weights = F.softmax(diagnosis_logits, dim=1)

        if self.use_fc_conv_stem:
            stem_out = self.conv1[0](img, threat_domain_indices,
                                      dispatch_weights=dispatch_weights, bpda=self.use_bpda)
            x = self.conv1[1:](stem_out[0])
        else:
            x = self.conv1(img)

        for i, blk in enumerate(self.mv2):
            x = blk(x, threat_domain_indices, dispatch_weights=dispatch_weights, bpda=self.use_bpda)
            if i == 4:
                x = self.mvit[0](x)
            if i == 5:
                x = self.mvit[1](x)
            if i == 6:
                x = self.mvit[2](x)

        x_feature = self.conv2(x)
        pooled = self.pool(x_feature).view(-1, x_feature.shape[1])
        out = self.fc(pooled)


        merged_expert_freqs = (stem_out[1] if self.use_fc_conv_stem and isinstance(stem_out, tuple) and len(stem_out) > 1 else
                               {f'domain_{i}': torch.zeros(1, self.num_frequency_experts, device=img.device)
                                for i in range(self.num_threat_domains)})

        return out, merged_expert_freqs, diagnosis_logits, predicted_threat_domains

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = super().state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )
        sd[prefix + 'threat_domain_diagnosis_state'] = self.threat_domain_diagnosis.get_state_dict()
        return sd

    def load_state_dict(self, state_dict, strict=True):
        state_dict = state_dict.copy()
        diagnosis_state = state_dict.pop('threat_domain_diagnosis_state', None)
        missing, unexpected = super().load_state_dict(state_dict, strict=False)
        if diagnosis_state is not None:
            device = next(self.parameters()).device
            self.threat_domain_diagnosis.load_state_dict(diagnosis_state, device)
        if strict and (len(missing) > 0 or len(unexpected) > 0):
            raise RuntimeError(f"State-dict mismatch: missing keys={missing}, unexpected keys={unexpected}")
        return missing, unexpected





def build_tafd_model(backbone='resnet', dataset=None, **kwargs):
    """Build a TaFD model for the requested backbone and dataset."""
    kwargs['dataset'] = dataset
    if backbone == 'resnet':
        return TaFDResNet34(**kwargs)
    elif backbone == 'mobilevit':
        if not EINOPS_AVAILABLE:
            raise ImportError("TaFDMobileViT requires 'einops' package. Install with: pip install einops")
        return TaFDMobileViT(**kwargs)
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose 'resnet' or 'mobilevit'.")
