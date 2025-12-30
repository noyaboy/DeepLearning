import torch
import torch.nn as nn
import torch.nn.functional as F

### Write your model architecture

###

def load_model(MODEL_PATH, base_channels=9):
    # Load state dict to CPU
    state_dict = torch.load(MODEL_PATH, map_location="cpu")

    # Check if this is a quantized model by looking for quantized tensors
    is_quantized = any(torch.is_tensor(v) and v.is_quantized for v in state_dict.values())

    if is_quantized:
        # Load as quantized INT8 model
        print(f"[load_model] Detected INT8 quantized model")

        # Create FP32 model structure first
        model_fp32 = EfficientUNet(n_channels=3, n_classes=8, base_channels=base_channels)
        model_fp32.train()

        # Prepare for QAT and convert to quantized
        torch.backends.quantized.engine = 'qnnpack'
        model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        model_prepared = torch.quantization.prepare_qat(model_fp32, inplace=False)
        model_prepared.eval()

        # Convert to quantized model
        model = torch.quantization.convert(model_prepared, inplace=False)

        # Load the quantized state dict
        state_dict_clean = {
            k: v for k, v in state_dict.items()
            if not (k.endswith("total_ops") or k.endswith("total_params"))
        }

        missing_keys, unexpected_keys = model.load_state_dict(state_dict_clean, strict=False)

        if missing_keys:
            print(f"[load_model] Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"[load_model] Unexpected keys: {len(unexpected_keys)}")

        print(f"[load_model] Loaded INT8 quantized model (CPU-only)")

        # Wrap the quantized model to handle .to() calls gracefully
        return QuantizedModelWrapper(model)

    else:
        # Load as FP32 model
        print(f"[load_model] Detected FP32 model")

        model = EfficientUNet(n_channels=3, n_classes=8, base_channels=base_channels)

        # Clean up state dict
        state_dict_clean = {
            k: v for k, v in state_dict.items()
            if not (k.endswith("total_ops") or k.endswith("total_params"))
        }

        missing_keys, unexpected_keys = model.load_state_dict(state_dict_clean, strict=False)

        if missing_keys:
            print(f"[load_model] Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"[load_model] Unexpected keys: {len(unexpected_keys)}")

        print(f"[load_model] Loaded FP32 model")
        return model

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        # Adaptive kernel size calculation
        t = int(abs((torch.log2(torch.tensor(float(channels))) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.q_mul = nn.quantized.FloatFunctional()

    def forward(self, x):
        # Global average pooling: (B, C, H, W) -> (B, C, 1, 1)
        y = self.avg_pool(x)
        # Squeeze and transpose: (B, C, 1, 1) -> (B, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)
        # 1D convolution across channels: (B, 1, C) -> (B, 1, C)
        y = self.conv(y)
        # Transpose back and unsqueeze: (B, 1, C) -> (B, C, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)
        # Apply sigmoid and broadcast
        y = self.sigmoid(y)
        return self.q_mul.mul(x, y)


class EfficientDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_eca=True):
        super(EfficientDoubleConv, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.use_eca = use_eca
        if use_eca:
            self.eca = ECABlock(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_eca:
            x = self.eca(x)
        return x


class EfficientDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EfficientDown, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            EfficientDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class EfficientUp(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(EfficientUp, self).__init__()
        # Use bilinear upsampling instead of transpose conv (fewer parameters)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 1x1 projections to match both inputs to output channels
        self.up_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False)
        # Conv operates on out_channels (constant width) instead of concatenated channels
        self.conv = EfficientDoubleConv(out_channels, out_channels)
        self.q_add = nn.quantized.FloatFunctional()
    def forward(self, x1, x2):
        # x1: upsampled from lower layer
        # x2: skip connection from encoder
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Project both inputs to out_channels and add element-wise
        x1_proj = self.up_proj(x1)
        x2_proj = self.skip_proj(x2)
        x = self.q_add.add(x1_proj, x2_proj)
        return self.conv(x)


class EfficientUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=8, base_channels=9):
        super(EfficientUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # Encoder - reduced depth and channels
        self.inc = EfficientDoubleConv(n_channels, base_channels)
        self.down1 = EfficientDown(base_channels, base_channels * 2)
        self.down2 = EfficientDown(base_channels * 2, base_channels * 4)
        self.down3 = EfficientDown(base_channels * 4, base_channels * 8)

        # Decoder with constant width (no channel doubling from concatenation)
        # EfficientUp(in_channels, out_channels, skip_channels)
        self.up1 = EfficientUp(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = EfficientUp(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up3 = EfficientUp(base_channels * 2, base_channels, base_channels)

        # Final output layer
        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):
        # Quantization stub (no-op in FP32 mode)
        x = self.quant(x)

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.outc(x)

        # Dequantization stub (no-op in FP32 mode)
        logits = self.dequant(logits)
        return logits



# QUANTIZATION-AWARE TRAINING (QAT) UTILITIES

def prepare_qat_model(model, backend='qnnpack'):
    # Set quantization backend
    torch.backends.quantized.engine = backend

    # Configure model for QAT
    model.train()

    # Set qconfig for quantization (per-tensor symmetric for weights and activations)
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)

    # Prepare model for QAT (adds fake quantization modules)
    qat_model = torch.quantization.prepare_qat(model, inplace=False)

    print(f"Model prepared for QAT with backend={backend}")
    print(f"Fake quantization modules added for training")

    return qat_model


def convert_to_quantized(qat_model):
    # Set model to eval mode (required for conversion)
    qat_model.eval()

    # Move model to CPU (qnnpack/fbgemm backends require CPU)
    qat_model_cpu = qat_model.cpu()

    # Convert to quantized model
    quantized_model = torch.quantization.convert(qat_model_cpu, inplace=False)

    print("Model converted to INT8 quantized format")
    print("All weights and activations are now 8-bit integers")
    print("Note: Quantized model is on CPU (qnnpack/fbgemm requirement)")

    return quantized_model


def calculate_quantized_model_size(model):
    total_size = 0

    # Count quantized parameters (INT8: 1 byte each)
    for name, param in model.named_parameters():
        if param.dtype == torch.qint8 or param.dtype == torch.quint8:
            # Quantized parameters: 1 byte per element
            total_size += param.numel() * 1
        else:
            # FP32 parameters (scale, zero_point, etc.): 4 bytes per element
            total_size += param.numel() * param.element_size()

    # Count buffers (running_mean, running_var, scale, zero_point, etc.)
    for name, buffer in model.named_buffers():
        if buffer.dtype == torch.qint8 or buffer.dtype == torch.quint8:
            total_size += buffer.numel() * 1
        else:
            total_size += buffer.numel() * buffer.element_size()

    size_kb = total_size / 1024
    return size_kb


class QuantizedModelWrapper(nn.Module):
    def __init__(self, quantized_model):
        super().__init__()
        self.model = quantized_model
        self.is_quantized = True

    def forward(self, x):
        # Ensure input is on CPU for quantized model
        if x.device.type != 'cpu':
            x = x.cpu()
        return self.model(x)

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def parameters(self):
        return self.model.parameters()

    def buffers(self):
        return self.model.buffers()

    def state_dict(self):
        return self.model.state_dict()
