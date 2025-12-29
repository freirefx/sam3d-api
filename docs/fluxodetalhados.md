# SAM 3 - Fluxogramas Explicativos

## 1. Fluxo Geral - Decisão de Uso

```mermaid
graph TD
    Start([Usuário com Tarefa]) --> Question{O que segmentar?}
    
    Question -->|Imagem única| ImageFlow[Workflow de Imagem]
    Question -->|Sequência/Vídeo| VideoFlow[Workflow de Vídeo]
    
    ImageFlow --> KnowObject{Sabe o que<br/>quer segmentar?}
    VideoFlow --> NeedTrack{Precisa<br/>rastrear?}
    
    KnowObject -->|Sim| UsePrompt[Usar Prompts<br/>texto/pontos/box]
    KnowObject -->|Não| UseAuto[Usar Gerador<br/>Automático]
    
    NeedTrack -->|Sim| UseVideo[Segmentação +<br/>Rastreamento]
    NeedTrack -->|Não| ProcessFrames[Processar frames<br/>individualmente]
    
    UsePrompt --> ResultImg[Máscaras Precisas]
    UseAuto --> ResultAuto[Todas Máscaras<br/>Possíveis]
    UseVideo --> ResultVid[Máscaras em<br/>Todos Frames]
    ProcessFrames --> ResultImg
    
    ResultImg --> End([Fim])
    ResultAuto --> End
    ResultVid --> End
    
    style Start fill:#e1f5ff
    style End fill:#d4edda
    style Question fill:#fff3cd
    style KnowObject fill:#fff3cd
    style NeedTrack fill:#fff3cd
```

---

## 2. Workflow Detalhado - Segmentação de Imagem

```mermaid
flowchart TD
    A([Início]) --> B[Importar bibliotecas<br/>sam3, PIL, numpy]
    B --> C[build_sam3_image_model<br/>checkpoint path]
    
    C --> D[Carregar checkpoint<br/>em memória GPU/CPU]
    D --> E[Sam3Processor model]
    
    E --> F[Abrir imagem<br/>PIL.Image.open]
    F --> G[processor.set_image image]
    
    G --> H{Escolher<br/>Tipo de Prompt}
    
    H -->|Descrição Textual| I1[set_text_prompt<br/>state, prompt, thresholds]
    H -->|Clique em Ponto| I2[set_point_prompt<br/>state, coords, labels]
    H -->|Desenhar Caixa| I3[set_box_prompt<br/>state, box]
    H -->|Máscara Prévia| I4[set_mask_prompt<br/>state, mask_input]
    
    I1 --> J[Processamento Interno:<br/>Encoder + Decoder]
    I2 --> J
    I3 --> J
    I4 --> J
    
    J --> K[Retorno: dict com<br/>masks, boxes, scores, logits]
    
    K --> L{Resultado<br/>Satisfatório?}
    
    L -->|Não| M[Combinar prompts ou<br/>ajustar thresholds]
    M --> H
    
    L -->|Sim| N[Extrair máscaras:<br/>output'masks']
    N --> O[Visualizar com<br/>matplotlib/opencv]
    
    O --> P[Salvar resultados:<br/>máscaras, coordenadas]
    P --> Q([Fim])
    
    style A fill:#e1f5ff
    style Q fill:#d4edda
    style H fill:#fff3cd
    style L fill:#fff3cd
    style J fill:#ffe6e6
```

---

## 3. Workflow - Geração Automática de Máscaras

```mermaid
sequenceDiagram
    participant User
    participant Generator as Sam3AutomaticMaskGenerator
    participant Model as Sam3ImageModel
    participant Output
    
    User->>Generator: Criar instância<br/>Sam3AutomaticMaskGenerator(model, params)
    Note over Generator: points_per_side=32<br/>pred_iou_thresh=0.88<br/>stability_score_thresh=0.95
    
    User->>Generator: generate(image)
    
    Generator->>Generator: Criar grade de pontos<br/>32x32 = 1024 pontos
    
    loop Para cada ponto na grade
        Generator->>Model: Prever máscara<br/>no ponto (x, y)
        Model-->>Generator: Retorna máscara candidata
    end
    
    Generator->>Generator: Filtrar por IoU threshold<br/>Remover duplicatas
    
    Generator->>Generator: Filtrar por stability score<br/>Remover instáveis
    
    Generator->>Generator: Non-Maximum Suppression<br/>Remover sobreposições
    
    Generator->>Generator: Ordenar por área/qualidade
    
    Generator-->>Output: Lista de máscaras<br/>cada com: segmentation,<br/>bbox, area, scores
    
    Output-->>User: Retornar todas máscaras
    
    Note over User: Pode ter 10-100+<br/>máscaras dependendo<br/>da imagem
```

---

## 4. Workflow Detalhado - Segmentação em Vídeo

```mermaid
stateDiagram-v2
    [*] --> Inicialização
    
    Inicialização --> CarregarModelo: build_sam3_video_predictor()
    CarregarModelo --> IniciarSessão: handle_request<br/>type='start_session'
    
    IniciarSessão --> AguardandoPrompts: Recebe session_id,<br/>num_frames, dimensões
    
    state AguardandoPrompts {
        [*] --> EscolherFrame
        EscolherFrame --> DefinirObjeto: Escolhe frame_index
        DefinirObjeto --> AdicionarPrompt: Define object_id
        
        state AdicionarPrompt {
            [*] --> TipoPrompt
            TipoPrompt --> Texto: text='...'
            TipoPrompt --> Pontos: point_coords, labels
            TipoPrompt --> Box: box=[x1,y1,x2,y2]
            TipoPrompt --> Máscara: mask=array
            
            Texto --> [*]
            Pontos --> [*]
            Box --> [*]
            Máscara --> [*]
        }
        
        AdicionarPrompt --> VerificarMais: handle_request<br/>type='add_prompt'
        VerificarMais --> EscolherFrame: Mais objetos?
        VerificarMais --> [*]: Pronto
    }
    
    AguardandoPrompts --> Propagação: handle_request<br/>type='propagate_in_video'
    
    state Propagação {
        [*] --> ProcessarFrames
        ProcessarFrames --> Forward: Frames 0→N
        ProcessarFrames --> Backward: Frames 0←N
        
        Forward --> GerarMáscaras: Rastreamento temporal
        Backward --> GerarMáscaras
        
        GerarMáscaras --> [*]: Máscaras para<br/>todos frames
    }
    
    Propagação --> ResultadosProntos: Retorna dict<br/>masks, scores
    
    state ResultadosProntos {
        [*] --> PodeConsultar
        PodeConsultar --> GetFrame: type='get_frame'
        PodeConsultar --> RemoverObjeto: type='remove_object'
        PodeConsultar --> AdicionarMais: Voltar para prompts
        
        GetFrame --> PodeConsultar
        RemoverObjeto --> PodeConsultar
    }
    
    ResultadosProntos --> FinalizarSessão: handle_request<br/>type='end_session'
    FinalizarSessão --> [*]: Libera memória
```

---

## 5. Arquitetura Interna - Processamento

```mermaid
graph LR
    subgraph Input
        A1[Imagem RGB] 
        A2[Prompt texto/ponto/box]
    end
    
    subgraph "Image Encoder"
        B1[Hiera Backbone]
        B2[Multi-scale Features]
    end
    
    subgraph "Prompt Encoder"
        C1[Text Encoder CLIP]
        C2[Point/Box Embeddings]
        C3[Mask Embeddings]
    end
    
    subgraph "Mask Decoder"
        D1[Cross-Attention]
        D2[Self-Attention]
        D3[MLP Heads]
    end
    
    subgraph Output
        E1[Máscaras Logits]
        E2[Bounding Boxes]
        E3[Confidence Scores]
    end
    
    A1 --> B1
    B1 --> B2
    
    A2 --> C1
    A2 --> C2
    A2 --> C3
    
    B2 --> D1
    C1 --> D1
    C2 --> D1
    C3 --> D1
    
    D1 --> D2
    D2 --> D3
    
    D3 --> E1
    D3 --> E2
    D3 --> E3
    
    style Input fill:#e1f5ff
    style Output fill:#d4edda
```

---

## 6. Fluxo de Decisão - Escolha de Prompt

```mermaid
graph TD
    Start([Preciso Segmentar]) --> Q1{Consigo descrever<br/>em palavras?}
    
    Q1 -->|Sim| Q2{É um conceito<br/>comum?}
    Q1 -->|Não| Q3{Posso clicar<br/>no objeto?}
    
    Q2 -->|Sim| UseText[✅ Usar Prompt Textual<br/>Rápido e preciso]
    Q2 -->|Não| Q4{Tenho imagem<br/>de exemplo?}
    
    Q3 -->|Sim| Q5{Objeto tem<br/>forma clara?}
    Q3 -->|Não| Q6{Tenho máscara<br/>aproximada?}
    
    Q4 -->|Sim| UseFewShot[✅ Few-Shot Learning<br/>Fornecer exemplos]
    Q4 -->|Não| UsePoints[✅ Prompt de Pontos<br/>Clicar em regiões]
    
    Q5 -->|Sim, retangular| UseBox[✅ Prompt de Box<br/>Desenhar retângulo]
    Q5 -->|Não| UsePoints
    
    Q6 -->|Sim| UseMask[✅ Prompt de Máscara<br/>Refinar segmentação]
    Q6 -->|Não| UseAuto[✅ Geração Automática<br/>Explorar todas opções]
    
    UseText --> Combine{Resultado OK?}
    UseBox --> Combine
    UsePoints --> Combine
    UseMask --> Combine
    UseFewShot --> Combine
    UseAuto --> Combine
    
    Combine -->|Não| Refine[Combinar múltiplos<br/>prompts]
    Combine -->|Sim| Success([✅ Segmentação Completa])
    
    Refine --> Example1[Exemplo: Texto + Pontos<br/>para maior precisão]
    Example1 --> Combine
    
    style Start fill:#e1f5ff
    style Success fill:#d4edda
    style UseText fill:#d1ecf1
    style UseBox fill:#d1ecf1
    style UsePoints fill:#d1ecf1
    style UseMask fill:#d1ecf1
    style UseFewShot fill:#d1ecf1
    style UseAuto fill:#d1ecf1
```

---

## 7. Pipeline de Vídeo - Rastreamento Temporal

```mermaid
gantt
    title Pipeline de Processamento de Vídeo
    dateFormat X
    axisFormat %L
    
    section Inicialização
    Carregar Modelo           :0, 100
    Iniciar Sessão           :100, 150
    Carregar Vídeo           :150, 250
    
    section Frame 0 (Anotação)
    Adicionar Prompt Obj 1   :250, 300
    Adicionar Prompt Obj 2   :300, 350
    Gerar Máscaras Iniciais  :350, 450
    
    section Propagação Forward
    Frame 1                  :450, 470
    Frame 2                  :470, 490
    Frame 3-10               :490, 650
    Frame 11-20              :650, 800
    
    section Propagação Backward  
    Verificação Frame 0      :800, 850
    Refinar se necessário    :850, 900
    
    section Pós-Processamento
    Suavização Temporal      :900, 1000
    Consistência de Máscaras :1000, 1100
    
    section Finalização
    Salvar Resultados        :1100, 1200
    Liberar Memória          :1200, 1250
```

---

## 8. Comparação de Métodos

```mermaid
graph TD
    subgraph "Prompt Textual"
        T1[Input: 'gato laranja']
        T2[Vantagens:<br/>✓ Rápido<br/>✓ Intuitivo<br/>✓ Vocabulário aberto]
        T3[Desvantagens:<br/>✗ Ambíguo às vezes<br/>✗ Depende de descrição]
        T1 --- T2
        T2 --- T3
    end
    
    subgraph "Prompt de Pontos"
        P1[Input: coords, labels]
        P2[Vantagens:<br/>✓ Muito preciso<br/>✓ Controle fino<br/>✓ Múltiplas regiões]
        P3[Desvantagens:<br/>✗ Manual<br/>✗ Trabalhoso em vídeo]
        P1 --- P2
        P2 --- P3
    end
    
    subgraph "Prompt de Box"
        B1[Input: x1,y1,x2,y2]
        B2[Vantagens:<br/>✓ Simples<br/>✓ Bom para objetos<br/>retangulares]
        B3[Desvantagens:<br/>✗ Menos preciso<br/>✗ Formas irregulares]
        B1 --- B2
        B2 --- B3
    end
    
    subgraph "Geração Automática"
        A1[Input: imagem]
        A2[Vantagens:<br/>✓ Zero esforço<br/>✓ Descobre tudo<br/>✓ Exploratório]
        A3[Desvantagens:<br/>✗ Muitos resultados<br/>✗ Mais lento<br/>✗ Precisa filtrar]
        A1 --- A2
        A2 --- A3
    end
    
    style T2 fill:#d4edda
    style P2 fill:#d4edda
    style B2 fill:#d4edda
    style A2 fill:#d4edda
    style T3 fill:#f8d7da
    style P3 fill:#f8d7da
    style B3 fill:#f8d7da
    style A3 fill:#f8d7da
```

---

## 9. Ciclo de Refinamento Iterativo

```mermaid
graph LR
    A([Primeira Tentativa]) --> B[set_text_prompt<br/>'pessoa']
    B --> C{Muitas detecções?}
    
    C -->|Sim| D[Refinar texto:<br/>'pessoa camisa azul']
    C -->|Não| E{Faltou alguma<br/>região?}
    
    D --> F[set_text_prompt<br/>mais específico]
    F --> E
    
    E -->|Sim| G[Adicionar ponto positivo<br/>na região faltante]
    E -->|Não| H{Incluiu regiões<br/>erradas?}
    
    G --> I[set_point_prompt<br/>coords foreground]
    I --> H
    
    H -->|Sim| J[Adicionar ponto negativo<br/>na região errada]
    H -->|Não| K{Bordas<br/>imprecisas?}
    
    J --> L[set_point_prompt<br/>coords background]
    L --> K
    
    K -->|Sim| M[set_mask_prompt<br/>para refinar]
    K -->|Não| N([✅ Resultado Final])
    
    M --> O{Melhorou?}
    O -->|Não| P[Ajustar thresholds<br/>box_threshold, text_threshold]
    O -->|Sim| N
    
    P --> B
    
    style A fill:#e1f5ff
    style N fill:#d4edda
    style C fill:#fff3cd
    style E fill:#fff3cd
    style H fill:#fff3cd
    style K fill:#fff3cd
    style O fill:#fff3cd
```

---

## 10. Gestão de Memória - Vídeo Longo

```mermaid
flowchart TD
    A[Vídeo com 1000 frames] --> B{Memória<br/>suficiente?}
    
    B -->|Sim| C[Processar<br/>tudo de uma vez]
    B -->|Não| D[Dividir em chunks]
    
    D --> E[Chunk 1: frames 0-249]
    E --> F[start_session<br/>add_prompts<br/>propagate]
    F --> G[Salvar máscaras<br/>chunk 1]
    
    G --> H[end_session<br/>liberar memória]
    
    H --> I[Chunk 2: frames 250-499]
    I --> J[start_session<br/>usar máscara final<br/>do chunk 1]
    J --> K[propagate]
    K --> L[Salvar máscaras<br/>chunk 2]
    
    L --> M[Repetir para<br/>chunks restantes]
    
    M --> N[Concatenar<br/>todos resultados]
    
    C --> O[propagate_in_video<br/>start_frame=0<br/>max_frame=999]
    
    O --> P[Obter todas<br/>máscaras]
    N --> P
    
    P --> Q([Processamento<br/>Completo])
    
    style A fill:#e1f5ff
    style Q fill:#d4edda
    style B fill:#fff3cd
```

## Legenda de Cores

- 🔵 **Azul claro**: Início/Input
- 🟢 **Verde**: Fim/Sucesso
- 🟡 **Amarelo**: Decisões/Pontos de escolha
- 🔴 **Vermelho claro**: Processamento interno
- 🟦 **Azul**: Vantagens
- 🟥 **Vermelho**: Desvantagens