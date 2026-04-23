import torch
from recoloradv.utils import get_attack_from_name

class IdentityNormalizer:
    """Identity normalizer that does nothing - model handles normalization internally."""
    def __call__(self, x):
        return x

    def forward(self, x):
        return x


def attack_image(image, label, model, attack):
    adv_inputs = attack.attack(
        image,
        label,
    )[0]
    with torch.no_grad():
        # Model handles normalization internally
        adv_logits = model(adv_inputs)[0]
        succ = (adv_logits.argmax(1) != label).detach()
    return adv_inputs, succ


def ReColorAdv(inputs, labels, model, device, max_iterations=10, lr=0.01, ncls=10,
               norm_mean=None, norm_std=None):
    """
    ReColorAdv attack function.
    Note: Model handles normalization internally, so input should be in [0,1] pixel domain.
    norm_mean and norm_std parameters are kept for API compatibility but not used.
    """
    # Use identity normalizer since model handles normalization internally
    normalizer = IdentityNormalizer()

    attack_name = 'recoloradv+delta'
    attack = get_attack_from_name(lr, attack_name, model, normalizer, max_iterations)
    adv_img, succ = attack_image(inputs, labels, model, attack)
    return adv_img
