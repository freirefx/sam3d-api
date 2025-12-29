graph TD
    Start([Início]) --> Choice{Tipo de<br/>Entrada?}
    
    Choice -->|Imagem| ImgPath[Fluxo de Imagem]
    Choice -->|Vídeo| VidPath[Fluxo de Vídeo]
    
    ImgPath --> LoadModel[build_sam3_image_model]
    LoadModel --> CreateProc[Sam3Processor model]
    CreateProc --> SetImg[processor.set_image image]
    
    SetImg --> PromptChoice{Tipo de<br/>Prompt?}
    
    PromptChoice -->|Texto| TextPrompt[set_text_prompt<br/>prompt='objeto']
    PromptChoice -->|Pontos| PointPrompt[set_point_prompt<br/>coords, labels]
    PromptChoice -->|Box| BoxPrompt[set_box_prompt<br/>box coords]
    PromptChoice -->|Máscara| MaskPrompt[set_mask_prompt<br/>mask_input]
    PromptChoice -->|Automático| AutoMask[AutomaticMaskGenerator<br/>generate]
    
    TextPrompt --> GetResults[Obter Resultados:<br/>masks, boxes, scores]
    PointPrompt --> GetResults
    BoxPrompt --> GetResults
    MaskPrompt --> GetResults
    AutoMask --> GetResults
    
    GetResults --> Refine{Precisa<br/>Refinar?}
    Refine -->|Sim| PromptChoice
    Refine -->|Não| Visualize[Visualizar com<br/>matplotlib]
    
    Visualize --> EndImg([Fim])
    
    VidPath --> LoadVidModel[build_sam3_video_predictor]
    LoadVidModel --> StartSession[handle_request<br/>type='start_session']
    StartSession --> GetSessionID[Receber session_id]
    
    GetSessionID --> AddPrompts[handle_request<br/>type='add_prompt'<br/>frame_index, object_id]
    
    AddPrompts --> MoreObjects{Mais<br/>Objetos?}
    MoreObjects -->|Sim| AddPrompts
    MoreObjects -->|Não| Propagate[handle_request<br/>type='propagate_in_video']
    
    Propagate --> GetMasks[Receber masks<br/>para todos frames]
    GetMasks --> Process[Processar/Salvar<br/>Resultados]
    
    Process --> EndSession[handle_request<br/>type='end_session']
    EndSession --> EndVid([Fim])
    
    style Start fill:#e1f5ff
    style EndImg fill:#d4edda
    style EndVid fill:#d4edda
    style Choice fill:#fff3cd
    style PromptChoice fill:#fff3cd
    style Refine fill:#fff3cd
    style MoreObjects fill:#fff3cd