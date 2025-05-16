

A **NVIDIA Triton Inference Server** (anteriormente conhecido como **TensorRT Inference Server**) é uma plataforma de inferência de código aberto desenvolvida pela NVIDIA para implantar modelos de machine learning (ML) e deep learning (DL) em produção de forma escalável e eficiente. Ele é projetado para funcionar em ambientes **GPU**, **CPU** ou **GPUs multi-nó**, suportando uma variedade de frameworks e formatos de modelos.

### **Principais recursos do NVIDIA Triton:**
1. **Suporte a Múltiplos Frameworks**  
   - TensorRT, TensorFlow, PyTorch, ONNX Runtime, OpenVINO, Python (scripts personalizados) e mais.  
   - Permite que diferentes modelos, treinados em diferentes frameworks, sejam executados no mesmo servidor.

2. **Modelos Ensemble**  
   - Combina vários modelos em um pipeline de inferência (útil para pré/pós-processamento ou modelos em cascata).

3. **Escalabilidade e Concorrência**  
   - Gerencia automaticamente solicitações em lotes (**dynamic batching**) para melhorar a utilização da GPU.  
   - Suporta implantação em **Kubernetes** e orquestração em clusters.

4. **Backends Personalizáveis**  
   - Permite adicionar backends em C++ ou Python para modelos personalizados.

5. **APIs Padronizadas**  
   - Suporta **HTTP/REST** e **gRPC** para integração com aplicações.  
   - Compatível com a **API KServe** (padrão do Kubernetes para inferência).

6. **Monitoramento e Métricas**  
   - Fornece métricas via **Prometheus** para monitoramento de desempenho.

7. **Multi-GPU e Multi-Nó**  
   - Distribui inferência em várias GPUs ou nós para alta disponibilidade e desempenho.

---

### **Casos de Uso Comuns**
- **Serviços de Inferência em Nuvem**: Implantação de modelos de visão computacional, NLP e recomendação.  
- **Edge Computing**: Execução eficiente em dispositivos NVIDIA Jetson.  
- **Pipelines de IA**: Integração com ferramentas como **NVIDIA TAO**, **RAPIDS** e **CUDA**.  

---

### **Como Usar o Triton?**
1. **Instalação**: Disponível como contêiner Docker via [NVIDIA NGC](https://catalog.ngc.nvidia.com/containers).  
   ```bash
   docker pull nvcr.io/nvidia/tritonserver:<version>
   ```
2. **Organização do Modelo**:  
   - Os modelos devem seguir uma estrutura de diretórios específica (ex: `model_repository/`).  
   - Cada modelo precisa de um arquivo `config.pbtxt` para configuração.

3. **Inicialização do Servidor**:  
   ```bash
   docker run --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
   -v /path/to/model_repository:/models \
   nvcr.io/nvidia/tritonserver:<version> \
   tritonserver --model-repository=/models
   ```

4. **Cliente de Inferência**:  
   - Use bibliotecas como `tritonclient` (Python) para enviar solicitações.  
   - Exemplo em Python:
     ```python
     import tritonclient.http as httpclient

     client = httpclient.InferenceServerClient(url="localhost:8000")
     inputs = [httpclient.InferInput("input_name", shape, "TYPE")]
     outputs = [httpclient.InferRequestedOutput("output_name")]
     results = client.infer(model_name, inputs, outputs=outputs)
     ```

---

### **Vantagens**
- **Alta Performance**: Otimizado para GPUs NVIDIA com TensorRT.  
- **Flexibilidade**: Suporte a múltiplos frameworks e modelos heterogêneos.  
- **Fácil Integração**: Compatível com Kubernetes, Prometheus e APIs padrão.  

