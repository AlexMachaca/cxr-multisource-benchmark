"""Grad-CAM para los 5 modelos de la app (módulo reutilizable; la app lo necesitará para calcular los mapas de calor).

Grad-CAM (Selvaraju et al., 2017): se toman las activaciones A_k de una capa convolucional/de tokens cerca de la salida,
se calcula el gradiente del puntaje de la clase objetivo respecto de ellas, se promedia ese gradiente espacialmente para obtener un
peso por canal (w_k) y el mapa es ReLU(sum_k w_k A_k), escalado a la imagen. Indica QUÉ REGIONES empujaron la decisión hacia la clase.

Cada familia guarda sus características de forma distinta, por eso hay una capa objetivo y un "layout" por modelo:
    CNN y CoAtNet : (B, C, H, W)                       -> se usa tal cual
    Swin          : (B, H, W, C)  (canales al final)   -> se permuta a (B, C, H, W)
    ViT           : (B, 1 + 14*14, C) (cls + parches)  -> se descarta el cls y se redimensiona a 14x14
Para ViT NO se usa la salida del último bloque: de ella solo el token cls llega a la cabeza, así que el gradiente sobre los
parches sería cero. Se usa la entrada de la atención del último bloque (`blocks[-1].norm1`), por donde sí circula el gradiente.

LIMITACIÓN: Grad-CAM en transformers y en CoAtNet es una explicación aproximada (la atención global mezcla los parches); los mapas
deben leerse como "dónde miró", no como delimitación exacta de la lesión.
"""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn.functional as F

# modelo -> (función que devuelve la capa objetivo, layout)
TARGETS = {
    "vit_base_patch16_224": (lambda m: m.blocks[-1].norm1, "tokens"),
    "coatnet_0_rw_224": (lambda m: m.stages[-1], "nchw"),
    "swin_tiny_patch4_window7_224": (lambda m: m.layers[-1], "nhwc"),
    "vgg16_bn": (lambda m: m.features[-2], "nchw"),        # ReLU previa al último max-pool: mapa 14x14
    "mobilenetv3_small_100": (lambda m: m.blocks[-1], "nchw"),
}


def load_model(model_name: str, n_classes: int, ckpt: Path, device) -> tuple[torch.nn.Module, dict]:
    model = timm.create_model(model_name, pretrained=False, num_classes=n_classes)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)          # solo se necesita el gradiente respecto de las activaciones, no de los pesos
    return model, timm.data.resolve_data_config({}, model=model)


class GradCAM:
    def __init__(self, model: torch.nn.Module, model_name: str):
        fn, self.layout = TARGETS[model_name]
        self.model, self.act = model, None
        self.handle = fn(model).register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        # corta el grafo en la capa objetivo: solo se guarda (y se deriva) lo que viene DESPUÉS de ella -> poca memoria
        self.act = out.detach().requires_grad_(True)
        return self.act

    def remove(self):
        self.handle.remove()

    def _nchw(self, t: torch.Tensor) -> torch.Tensor:
        if self.layout == "nhwc":
            return t.permute(0, 3, 1, 2)
        if self.layout == "tokens":
            b, n, c = t.shape
            s = int(round((n - 1) ** 0.5))
            return t[:, 1:, :].transpose(1, 2).reshape(b, c, s, s)
        return t

    def __call__(self, x: torch.Tensor, target: torch.Tensor | None = None, size: int = 224):
        """Devuelve (logits, clase_usada, cam (B,size,size) en [0,1])."""
        with torch.enable_grad():
            logits = self.model(x)
            cls = logits.argmax(1) if target is None else target
            score = logits.gather(1, cls[:, None]).sum()
            grad = torch.autograd.grad(score, self.act)[0]
        act, grad = self._nchw(self.act.detach()), self._nchw(grad)
        w = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False).squeeze(1)
        mx = cam.flatten(1).max(1)[0].clamp_min(1e-8).view(-1, 1, 1)
        return logits.detach(), cls.detach(), (cam / mx).detach()
