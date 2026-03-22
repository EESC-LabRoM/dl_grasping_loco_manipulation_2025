import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import *

import torch
from torch import nn 
from torchvision.models import mobilenet_v2
from ultralytics import YOLO

print("PyTorch version: ",torch.__version__)
print("CUDA Available:", torch.cuda.is_available()) # checks for GPU access with PyTorch
print("PyTorch-Cuda version: ",torch.version.cuda) 
print("Device name: ", torch.cuda.get_device_name(0))
device = "cuda" if torch.cuda.is_available() else "cpu"

class UpBlock(nn.Module):
    def __init__(self, in_up, in_skip, out_c, groups=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_up, out_c, 2, stride=2, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(out_c + in_skip, out_c, 3, padding=1, groups=groups, bias=False),
            nn.GroupNorm(8, out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x_up, x_skip):
        x = self.up(x_up)
        x = torch.cat([x_skip, x], 1)
        return self.conv(x)

class GraspNN_V3(nn.Module):
    def __init__(self, in_channels=8, base=16):
        super().__init__()
        mb = mobilenet_v2(weights=None)
        # adapta 8 canais para 32
        mb.features[0][0] = nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False)
        self.backbone = mb.features

        self.enc1 = self.backbone[:3]     # 24 ch
        self.enc2 = self.backbone[3:6]    # 32 ch
        self.enc3 = self.backbone[6:10]   # 64 ch   (skip)
        self.enc4 = self.backbone[10:]    # 1280 ch (bottleneck)

        self.up3 = UpBlock(1280, 64,  base*8)  # <-- canais certos
        self.up2 = UpBlock(base*8, 32, base*4)
        self.up1 = UpBlock(base*4, 24, base*2)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(base*2, base, 2, stride=2, bias=False),
            nn.GroupNorm(8, base), nn.ReLU(inplace=True))
        self.outc = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x  = self.enc4(s3)

        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x)
        return self.outc(x)

def load_rgb_d_images(rgb_path, depth_path):
    """Load pre-saved RGB and Depth images."""
    if not os.path.exists(rgb_path):
        print(f"Erro: RGB file {rgb_path} not found.")
        return None, None
    if not os.path.exists(depth_path):
        print(f"Erro: Depth file {depth_path} not found.")
        return None, None

    rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR) # expects .png
    depth_img = np.load(depth_path) # expects .npy
    
    print(
        f"RGB Source: {rgb_path}, "
        f"Resolution: {rgb_img.shape[0]}x{rgb_img.shape[1]}, "
        f"Shape: {rgb_img.shape}"
    )
    print(
        f"Depth Source: {depth_path}, "
        f"Resolution: {depth_img.shape[0]}x{depth_img.shape[1]}, "
        f"Shape: {depth_img.shape}"
    )

    return rgb_img, depth_img

def segment_bottle(model, rgb):
    """Segment a bottle using YOLO and extract mask."""
    results = model(rgb_img)[0]
    # Inicializar uma máscara em branco
    mask = np.zeros_like(rgb[:, :, 0], dtype=np.uint8)

    # Iterar sobre os resultados de segmentação
    for result in results:
        # Verificar se as máscaras estão presentes
        if result.masks is not None:
            # Obter as máscaras de segmentação
            for i, mask_data in enumerate(result.masks.xy):
                # Verificar se a classe detectada é 'garrafa' (classe 39 no COCO)
                class_id = int(result.boxes.cls[i])
                if (result.names[class_id] == "vase") | (result.names[class_id] == "bottle"):  # Verificar pela classe 'garrafa'
                    # Criar a máscara binária a partir do segmento
                    mask_polygon = np.array(mask_data, dtype=np.int32)
                    # Preencher a máscara com a área correspondente à garrafa
                    cv2.fillPoly(mask, [mask_polygon], 1)

    # Exibir a imagem original
    cv2.imshow("original image",cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

    # Sobrepor a máscara da garrafa
    cv2.imshow("mask", mask)

    return mask

def d2nt(depth, fx, fy, u0, v0):
    """normal maping version implemented: d2nt_v3"""
    h, w = depth.shape
    depth = depth.astype(np.float64) / 1000.0
    u_map = np.ones((h, 1)) * np.arange(1, w + 1) - u0
    v_map = np.arange(1, h + 1).reshape(h, 1) * np.ones((1, w)) - v0
    Gu, Gv = get_DAG_filter(depth)
    est_nx = Gu * fx
    est_ny = Gv * fy
    est_nz = -(depth + v_map * Gv + u_map * Gu)
    est_normal = cv2.merge((est_nx, est_ny, est_nz))
    est_normal = vector_normalization(est_normal)
    est_normal = MRF_optim(depth, est_normal)
    return est_normal

def preprocess_data(rgb, depth, normal, mask, target):
    
    mask = cv2.inRange(mask, 1, 1).astype(bool).astype(int) # Seleciona apenas o objeto, retirando o gripper ou outros elementos
    
    depth[mask == 0] = 2 # Coloca tudo que não é o objeto a 2 metros de distância
    depth = depth/depth.max() # Normaliza a profundidade
    
    normal[mask == 0] = np.array([0.49803922, 0.49803922, 1. ])*255 # Retira normais indesejadas
            
    #grasp = (cv2.inRange(grasp, 0, 0).astype(bool) * mask).astype(int) + grasp # Preenche os espaços vazios da garrafa
    
    kernel = np.ones((2, 2), np.uint8)  # Definir o tamanho do kernel (ajuste conforme necessário)
    target = cv2.dilate(target, kernel, iterations=2)
    
    return rgb, depth, normal, mask, target

def nn_grasping_model(loaded_model, rgb_img, depth_img, mask, normal_map):
    """ Here the data will be passed through the loaded_model """
    view_rgb_t = torch.from_numpy(np.copy(rgb_img)).permute(2,0,1).float()
    view_depth_t = torch.from_numpy(np.copy(depth_img)).unsqueeze(dim=2).permute(2,0,1).float()
    view_normal_t = torch.from_numpy(np.copy(normal_map)).permute(2,0,1).float()
    view_seg_t = torch.from_numpy(np.copy(mask)).unsqueeze(dim=2).permute(2,0,1).float()
    data = torch.cat([
        view_rgb_t,
        view_depth_t,
        view_normal_t,
        view_seg_t,
        ],dim=0).unsqueeze(dim=0).to(device)
    
    with torch.inference_mode():
        output = loaded_model(data).squeeze().cpu()

    grasp_y, grasp_x = np.unravel_index(np.argmax(output*mask), output.shape)
    return grasp_x, grasp_y

def calculate_params(rgb_img, depth_img, grasp_y, grasp_x, normal_map):
    """Calculate grasp parameters using d2nt and grasp pixel. 
       Params f[i] needs to be checked!!!!!!!!!!!!!!!!!!!!!!"""
    # FOV needs to be checked !!!!!!!!!!!!
    x_rgb, y_rgb = grasp_y, grasp_x # TODO: analise this invertion from abscisses and ordinates
    depth_height, depth_width = depth_img.shape
    rgb_height, rgb_width = rgb_img.shape[:2]

    if rgb_width == depth_width and rgb_height == depth_height:
        x = x_rgb
        y = y_rgb
    else:
        y = int(y_rgb * depth_height / rgb_height)
        x = int(x_rgb * depth_width / rgb_width)

    x = min(max(x, 0), depth_width - 1)
    y = min(max(y, 0), depth_height - 1)
    depth_value = depth_img[y, x]
    print(f"Grasp pixel (x, y): ({x_rgb}, {y_rgb}), Depth at ({x}, {y}): {depth_value} mm")

    Z = depth_value / 1000.0
    if Z <= 0:
        print("Invalid depth (0 mm).")
        return None, None, None, None

    X = (x - u0) * Z / fx
    Y = (y - v0) * Z / fy
    distance = np.sqrt(X**2 + Y**2 + Z**2)
    normal_vector = normal_map[y, x]
    position = (X, Y, Z)

    print(f"d2nt parameters: u0={u0}, v0={v0}, fx={fx}, fy={fy}")
    print(f"Position: ({X:.3f}, {Y:.3f}, {Z:.3f}) m, Distance: {distance:.2f} m, Normal: {normal_vector}")

    return (x_rgb, y_rgb), normal_vector, position, distance

# Carregando o modelo salvo
MODEL_PATH = Path("simulations_Arroz/spot_validate/evaluation/models/nn-2_25-04-28_GraspNN_V3.pth")
loaded_model = GraspNN_V3() # cria um objeto com o mesmo tipo dos dados anteriores
loaded_model.load_state_dict(torch.load(f=MODEL_PATH)) 
loaded_model.to(device)

# Load YOLO model
model = YOLO("yolo11n-seg.pt")
print("YOLOv11 segmentation model loaded.")
path_data = Path("path/to/data")
rgb_path = "simulations_Arroz/spot_validate/images/rgbd-3/rgb/rgb_20250428153646.png"#path_data / "rgb.png"
depth_path = "simulations_Arroz/spot_validate/images/rgbd-3/depth/depth_20250428153646.npy"#path_data / "depth.npy"
# Load images
rgb_img, depth_img = load_rgb_d_images(rgb_path, depth_path)
if rgb_img is None or depth_img is None:
    print("Failed to laod images, exiting . . .")
    sys.exit(1)

# Segment bottle
seg = segment_bottle(model, rgb_img)
if seg is None:
    print("No bottle to grasp, exiting . . .")
    cv2.imshow("Gripper RGB", rgb_img)
    depth_display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
    cv2.imshow("Gripper Depth", depth_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(1)

fov = 30
fx, fy, u0, v0 = 640 / (2 * np.tan(np.deg2rad(fov) / 2)), 480 / (2 * np.tan(np.deg2rad(fov) / 2)), 640, 480
normal_map = d2nt(depth_img, fx, fy, u0, v0)

# Data pre-processment
rgb_input, depth_input, normal_map_input, mask, _= preprocess_data(rgb_img, depth_img, normal_map, seg, target=np.zeros(rgb_img.shape))

# NN grasping model
grasp_y, grasp_x = nn_grasping_model(loaded_model, rgb_input, depth_input, mask, normal_map_input)
grasp_pixel = grasp_y, grasp_x
if grasp_pixel is None:
    print("No grasp pixel selected, exiting . . .")
    cv2.imshow("Gripper RGB", rgb_img)
    depth_display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
    cv2.imshow("Gripper Depth", depth_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(1)

# Calculate grasp parameters
pixel, normal_vector, position, distance = calculate_params(rgb_input, depth_img, grasp_y, grasp_x, normal_map_input)
if position is None:
    print("No valid position or normal, exiting . . .")
    cv2.imshow("Gripper RGB", rgb_img)
    depth_display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
    cv2.imshow("Gripper Depth", depth_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(1)

# Visualize results
rgb_display = rgb_img.copy()
if pixel:
    x, y = pixel
    cv2.circle(rgb_display, (x, y), 5, (0, 255, 0), -1)  # Mark grasp pixel
cv2.imshow("Gripper RGB with Grasp", rgb_display)
depth_display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
cv2.imshow("Gripper Depth", depth_display)  
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Evaluation completed.")
