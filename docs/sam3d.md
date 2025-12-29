# SAM 3 - Guia Completo de Funções

## 📋 Índice
1. [Configuração Inicial](#configuração-inicial)
2. [Funções para Imagens](#funções-para-imagens)
3. [Funções para Vídeos](#funções-para-vídeos)
4. [Funções Auxiliares](#funções-auxiliares)

---

## Configuração Inicial

### `build_sam3_image_model()`
Constrói o modelo SAM 3 para segmentação de imagens.

**Entradas:**
- `checkpoint` (str, opcional): Caminho para o arquivo de checkpoint
- `device` (str, opcional): Dispositivo para executar o modelo ('cuda' ou 'cpu')
- `config` (dict, opcional): Configurações customizadas do modelo

**Saídas:**
- `model` (Sam3ImageModel): Instância do modelo SAM 3 para imagens

**Exemplo:**
```python
model = build_sam3_image_model(
    checkpoint="sam3_hiera_large.pt",
    device="cuda"
)
```

---

### `build_sam3_video_predictor()`
Constrói o preditor SAM 3 para segmentação e rastreamento em vídeos.

**Entradas:**
- `checkpoint` (str, opcional): Caminho para o arquivo de checkpoint
- `device` (str, opcional): Dispositivo para executar o modelo
- `max_inference_state_frames` (int, opcional): Máximo de frames a processar simultaneamente

**Saídas:**
- `predictor` (Sam3VideoPredictor): Instância do preditor para vídeos

**Exemplo:**
```python
predictor = build_sam3_video_predictor(
    checkpoint="sam3_hiera_large.pt"
)
```

---

## Funções para Imagens

### `Sam3Processor(model)`
Classe processadora para segmentação de imagens.

**Entradas (construtor):**
- `model` (Sam3ImageModel): Modelo SAM 3 construído

**Saídas:**
- `processor` (Sam3Processor): Instância do processador

---

### `processor.set_image()`
Define a imagem a ser processada.

**Entradas:**
- `image` (PIL.Image ou np.ndarray): Imagem a segmentar
- `image_format` (str, opcional): Formato da imagem ('RGB' ou 'BGR')

**Saídas:**
- `inference_state` (dict): Estado de inferência contendo a imagem processada

**Exemplo:**
```python
from PIL import Image
image = Image.open("foto.jpg")
state = processor.set_image(image)
```

---

### `processor.set_text_prompt()`
Segmenta usando prompt textual.

**Entradas:**
- `state` (dict): Estado de inferência da imagem
- `prompt` (str): Descrição textual do objeto ("carro vermelho", "pessoa com chapéu")
- `box_threshold` (float, opcional): Limiar de confiança para detecção (padrão: 0.3)
- `text_threshold` (float, opcional): Limiar de similaridade textual (padrão: 0.25)

**Saídas:**
- `output` (dict): Dicionário contendo:
  - `masks` (torch.Tensor): Máscaras binárias, shape (N, H, W)
  - `boxes` (torch.Tensor): Bounding boxes, shape (N, 4) formato [x1, y1, x2, y2]
  - `scores` (torch.Tensor): Scores de confiança, shape (N,)
  - `logits` (torch.Tensor): Logits das máscaras

**Exemplo:**
```python
output = processor.set_text_prompt(
    state=state,
    prompt="gato laranja dormindo",
    box_threshold=0.35
)
masks = output["masks"]  # Tensor com as máscaras
```

---

### `processor.set_point_prompt()`
Segmenta usando pontos como prompt.

**Entradas:**
- `state` (dict): Estado de inferência
- `point_coords` (np.ndarray): Coordenadas dos pontos, shape (N, 2) formato [x, y]
- `point_labels` (np.ndarray): Labels dos pontos, shape (N,) - 1 para foreground, 0 para background
- `multimask_output` (bool, opcional): Se True, retorna 3 máscaras candidatas

**Saídas:**
- `output` (dict): Dicionário contendo:
  - `masks` (np.ndarray): Máscaras, shape (N, H, W) ou (3, H, W) se multimask
  - `scores` (np.ndarray): Scores de qualidade das máscaras
  - `logits` (np.ndarray): Logits das máscaras

**Exemplo:**
```python
import numpy as np
# Ponto positivo na posição (100, 150)
points = np.array([[100, 150]])
labels = np.array([1])

output = processor.set_point_prompt(
    state=state,
    point_coords=points,
    point_labels=labels,
    multimask_output=True
)
```

---

### `processor.set_box_prompt()`
Segmenta usando bounding box como prompt.

**Entradas:**
- `state` (dict): Estado de inferência
- `box` (np.ndarray): Coordenadas da caixa, shape (4,) formato [x1, y1, x2, y2]
- `multimask_output` (bool, opcional): Se True, retorna múltiplas máscaras

**Saídas:**
- `output` (dict): Mesmo formato do set_point_prompt

**Exemplo:**
```python
# Box ao redor de um objeto
box = np.array([50, 50, 200, 200])
output = processor.set_box_prompt(state=state, box=box)
```

---

### `processor.set_mask_prompt()`
Refina segmentação usando máscara existente.

**Entradas:**
- `state` (dict): Estado de inferência
- `mask_input` (np.ndarray): Máscara inicial, shape (H, W) ou (1, H, W)
- `multimask_output` (bool, opcional): Se True, retorna múltiplas refinamentos

**Saídas:**
- `output` (dict): Máscara refinada no mesmo formato

**Exemplo:**
```python
# Usar máscara anterior como input
refined = processor.set_mask_prompt(
    state=state,
    mask_input=output["masks"][0]
)
```

---

### `Sam3AutomaticMaskGenerator(model)`
Gera máscaras automaticamente sem prompts.

**Entradas (construtor):**
- `model` (Sam3ImageModel): Modelo SAM 3
- `points_per_side` (int, opcional): Número de pontos por lado da grade (padrão: 32)
- `pred_iou_thresh` (float, opcional): Limiar de IoU para filtrar máscaras (padrão: 0.88)
- `stability_score_thresh` (float, opcional): Limiar de estabilidade (padrão: 0.95)
- `crop_n_layers` (int, opcional): Número de camadas de crop para processar (padrão: 0)
- `min_mask_region_area` (int, opcional): Área mínima da máscara em pixels (padrão: 0)

**Método `generate()`:**

**Entradas:**
- `image` (np.ndarray): Imagem RGB, shape (H, W, 3)

**Saídas:**
- `masks` (list[dict]): Lista de dicionários, cada um contendo:
  - `segmentation` (np.ndarray): Máscara binária, shape (H, W)
  - `bbox` (list): Bounding box [x, y, width, height]
  - `area` (int): Área da máscara em pixels
  - `predicted_iou` (float): IoU predito
  - `stability_score` (float): Score de estabilidade

**Exemplo:**
```python
from sam3 import Sam3AutomaticMaskGenerator

generator = Sam3AutomaticMaskGenerator(
    model=model,
    points_per_side=32,
    pred_iou_thresh=0.9
)

masks = generator.generate(image)
print(f"Encontradas {len(masks)} máscaras")
```

---

## Funções para Vídeos

### `predictor.handle_request()`
Interface unificada para operações em vídeo.

**Tipos de Requisição:**

#### 1. **start_session** - Iniciar nova sessão

**Entradas:**
```python
request = {
    "type": "start_session",
    "resource_path": "video.mp4",  # Caminho do vídeo
    "resource_type": "video"        # Tipo de recurso
}
```

**Saídas:**
```python
response = {
    "session_id": "abc123",         # ID único da sessão
    "num_frames": 250,              # Total de frames
    "video_height": 1080,           # Altura do vídeo
    "video_width": 1920             # Largura do vídeo
}
```

---

#### 2. **add_prompt** - Adicionar prompt em frame específico

**Entradas:**
```python
request = {
    "type": "add_prompt",
    "session_id": "abc123",
    "frame_index": 0,               # Frame onde adicionar prompt
    "object_id": 1,                 # ID do objeto a rastrear
    
    # Opção 1: Prompt textual
    "text": "pessoa com camisa vermelha",
    
    # Opção 2: Pontos
    "point_coords": [[100, 150], [200, 250]],
    "point_labels": [1, 1],         # 1=foreground, 0=background
    
    # Opção 3: Box
    "box": [50, 50, 300, 400],      # [x1, y1, x2, y2]
    
    # Opção 4: Máscara
    "mask": np.array(...)            # Máscara binária
}
```

**Saídas:**
```python
response = {
    "object_id": 1,
    "frame_index": 0,
    "mask": np.array(...),          # Máscara gerada, shape (H, W)
    "score": 0.95                   # Confiança
}
```

---

#### 3. **propagate_in_video** - Propagar segmentação

**Entradas:**
```python
request = {
    "type": "propagate_in_video",
    "session_id": "abc123",
    "start_frame": 0,               # Frame inicial (opcional)
    "max_frame": 100                # Frame final (opcional)
}
```

**Saídas:**
```python
response = {
    "masks": {                      # Máscaras por objeto e frame
        1: {                        # object_id = 1
            0: np.array(...),       # frame 0
            1: np.array(...),       # frame 1
            ...
        },
        2: {                        # object_id = 2
            ...
        }
    },
    "scores": {                     # Scores de confiança
        1: {0: 0.95, 1: 0.93, ...},
        2: {...}
    }
}
```

---

#### 4. **get_frame** - Obter frame específico

**Entradas:**
```python
request = {
    "type": "get_frame",
    "session_id": "abc123",
    "frame_index": 42
}
```

**Saídas:**
```python
response = {
    "frame": np.array(...),         # Frame RGB, shape (H, W, 3)
    "frame_index": 42
}
```

---

#### 5. **remove_object** - Remover objeto rastreado

**Entradas:**
```python
request = {
    "type": "remove_object",
    "session_id": "abc123",
    "object_id": 1
}
```

**Saídas:**
```python
response = {
    "success": True,
    "object_id": 1
}
```

---

#### 6. **end_session** - Finalizar sessão

**Entradas:**
```python
request = {
    "type": "end_session",
    "session_id": "abc123"
}
```

**Saídas:**
```python
response = {
    "success": True
}
```

---

## Funções Auxiliares

### `show_mask()`
Visualiza máscara sobre imagem.

**Entradas:**
- `mask` (np.ndarray): Máscara binária, shape (H, W)
- `ax` (matplotlib.axes.Axes): Eixo do matplotlib
- `random_color` (bool, opcional): Usar cor aleatória (padrão: False)
- `color` (tuple, opcional): Cor RGB customizada

**Saídas:**
- Nenhuma (plota diretamente no eixo)

**Exemplo:**
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.imshow(image)
show_mask(masks[0], ax, random_color=True)
plt.show()
```

---

### `show_points()`
Visualiza pontos sobre imagem.

**Entradas:**
- `coords` (np.ndarray): Coordenadas dos pontos, shape (N, 2)
- `labels` (np.ndarray): Labels dos pontos, shape (N,)
- `ax` (matplotlib.axes.Axes): Eixo do matplotlib
- `marker_size` (int, opcional): Tamanho dos marcadores

**Saídas:**
- Nenhuma (plota diretamente)

---

### `show_box()`
Visualiza bounding box sobre imagem.

**Entradas:**
- `box` (np.ndarray): Coordenadas [x1, y1, x2, y2]
- `ax` (matplotlib.axes.Axes): Eixo do matplotlib
- `edgecolor` (str, opcional): Cor da borda (padrão: 'green')
- `linewidth` (int, opcional): Espessura da linha

**Saídas:**
- Nenhuma (plota diretamente)

---

## 📊 Exemplo Completo - Workflow de Imagem

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
import matplotlib.pyplot as plt

# 1. Carregar modelo
model = build_sam3_image_model(checkpoint="sam3_hiera_large.pt")
processor = Sam3Processor(model)

# 2. Processar imagem
image = Image.open("foto.jpg")
state = processor.set_image(image)

# 3. Segmentar com texto
output = processor.set_text_prompt(
    state=state,
    prompt="cachorro marrom"
)

# 4. Refinar com ponto adicional
refined = processor.set_point_prompt(
    state=state,
    point_coords=np.array([[150, 200]]),
    point_labels=np.array([1])
)

# 5. Visualizar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(image)
ax1.imshow(output["masks"][0], alpha=0.5)
ax1.set_title("Texto apenas")

ax2.imshow(image)
ax2.imshow(refined["masks"][0], alpha=0.5)
ax2.set_title("Texto + Ponto")
plt.show()
```

---

## 🎬 Exemplo Completo - Workflow de Vídeo

```python
from sam3.model_builder import build_sam3_video_predictor

# 1. Criar preditor
predictor = build_sam3_video_predictor()

# 2. Iniciar sessão
resp = predictor.handle_request({
    "type": "start_session",
    "resource_path": "video.mp4"
})
session_id = resp["session_id"]

# 3. Adicionar prompt no frame 0
predictor.handle_request({
    "type": "add_prompt",
    "session_id": session_id,
    "frame_index": 0,
    "text": "bola de futebol",
    "object_id": 1
})

# 4. Adicionar segundo objeto
predictor.handle_request({
    "type": "add_prompt",
    "session_id": session_id,
    "frame_index": 0,
    "text": "jogador camisa 10",
    "object_id": 2
})

# 5. Propagar para todo o vídeo
resp = predictor.handle_request({
    "type": "propagate_in_video",
    "session_id": session_id
})

# 6. Acessar resultados
masks_obj1 = resp["masks"][1]  # Máscaras da bola
masks_obj2 = resp["masks"][2]  # Máscaras do jogador

# 7. Finalizar
predictor.handle_request({
    "type": "end_session",
    "session_id": session_id
})
```

---

## 🔧 Configurações Avançadas

### Otimização de Performance

```python
# Processar em batch
images = [Image.open(f"img{i}.jpg") for i in range(10)]
outputs = []

for img in images:
    state = processor.set_image(img)
    out = processor.set_text_prompt(state, prompt="gato")
    outputs.append(out)
```

### Ajuste de Limiares

```python
# Mais sensível (mais detecções)
output = processor.set_text_prompt(
    state=state,
    prompt="pessoa",
    box_threshold=0.2,      # Menor = mais detecções
    text_threshold=0.2
)

# Mais conservador (menos falsos positivos)
output = processor.set_text_prompt(
    state=state,
    prompt="pessoa",
    box_threshold=0.5,      # Maior = menos detecções
    text_threshold=0.4
)
```

---

## 📚 Referências

- **Repositório:** https://github.com/facebookresearch/sam3
- **Paper:** [Segment Anything 3](https://ai.meta.com/research/publications/sam-3/)
- **Checkpoints:** Disponíveis no Hugging Face após solicitação
- **Exemplos:** https://github.com/facebookresearch/sam3/tree/main/examples