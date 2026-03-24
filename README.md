# edge-phone-usage-detector

Aplicação de visão computacional em tempo real para detecção de uso de celular via webcam, com foco em cenários de segurança no trânsito e monitoramento de comportamento do condutor.

O projeto utiliza um modelo treinado no Edge Impulse no formato **`.eim` (Linux x86_64)**.

## 📸 Demo

Exemplos de detecção em tempo real:

![Demo 1](assets/examples/foto1.png)
![Demo 2](assets/examples/foto2.png)
![Demo 3](assets/examples/foto3.png)
![Demo 4](assets/examples/foto4.png)

## Problema que o projeto resolve

Uso de celular ao volante é um fator recorrente em acidentes e quase-acidentes, principalmente por reduzir tempo de reação e atenção situacional do condutor. Este projeto demonstra uma pipeline de visão computacional em tempo real para detecção desse comportamento, servindo como base técnica para protótipos de monitoramento em cabine, validação de regras de segurança e estudos de assistência ao motorista.

## Tecnologias

- Python 3.10+
- OpenCV (`opencv-python`)
- NumPy
- Edge Impulse Linux Python SDK / runtime compatível com `.eim` (instalação pode variar por ambiente)

## Estrutura do projeto

```text
.
├── assets/examples/          # Mídias para demonstração (prints/gifs)
├── docs/ARCHITECTURE.md      # Arquitetura técnica e fluxo
├── model/                    # Pasta do modelo Edge Impulse .eim
├── scripts/run_dev.sh        # Script de execução local
├── src/
│   ├── main.py               # Entrypoint
│   └── app/
│       ├── config/           # Configurações
│       ├── domain/           # Entidades de domínio
│       ├── services/         # Câmera, inferência e pipeline
│       └── utils/            # Logging e paths
├── tests/                    # Testes básicos
├── requirements.txt          # Dependências Python base (sem runtime Edge Impulse fixo)
└── .gitignore
```

## Instalação

### 1) Criar ambiente virtual

```bash
python3 -m venv .venv
```

### 2) Ativar ambiente virtual

```bash
source .venv/bin/activate
```

### 3) Instalar dependências base

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Instalar runtime Edge Impulse compatível com `.eim`

A parte de inferência `.eim` depende de runtime/SDK Edge Impulse disponível no seu Linux. O nome exato do pacote e versão podem variar.

Exemplo comum (quando disponível no seu ambiente):

```bash
pip install edge_impulse_linux
```

Se esse pacote não estiver disponível para sua plataforma, consulte a documentação oficial do Edge Impulse para Linux e instale o runner Python equivalente.

## Modelo `.eim`

Coloque o modelo em:

```bash
model/cellphone_detector.eim
```

Observações:

- Este projeto **não** trata o modelo como TFLite/ONNX/PyTorch.
- A inferência é feita por runtime Edge Impulse compatível com `.eim`.

## Como rodar

### Opção 1: script

```bash
bash scripts/run_dev.sh
```

### Opção 2: comando direto

```bash
python -m src.main \
  --model-path model/cellphone_detector.eim \
  --camera-index 0 \
  --camera-width 1280 \
  --camera-height 720 \
  --min-confidence 0.5
```

Controles:

- `q` para encerrar.

## Integração `.eim` no projeto

`EimInferenceService` tenta usar `ImageImpulseRunner` via SDK Python do Edge Impulse para:

1. Inicializar o modelo `.eim` no Linux.
2. Converter frame de BGR para RGB.
3. Executar inferência por frame.
4. Ler `result.bounding_boxes` e normalizar saída para o pipeline.

A aplicação faz tratamento defensivo para falhas de inicialização, ausência de runtime e resposta inesperada do runner.

## Tratamento de erros implementado

- Webcam indisponível.
- Arquivo `.eim` ausente ou inválido.
- Falha ao inicializar runtime Edge Impulse.
- Erro durante inferência.
- Dependências ausentes.

## Limitações técnicas reais

- O runtime `.eim` depende de suporte da plataforma Linux e bibliotecas nativas.
- Metadados de modelo (`model_info`) podem variar por versão do SDK.
- O desempenho (FPS/latência) depende de CPU, webcam e resolução.
- O modelo atual foi treinado com entrada reduzida (ex.: 96x96), o que limita detalhe espacial e pode degradar a precisão das bounding boxes.
- A capacidade de generalização ainda é limitada: variações de iluminação, ângulo da câmera, oclusões parciais e contexto de fundo podem alterar significativamente a confiança da detecção.
- A qualidade final depende fortemente do dataset (diversidade de cenários, balanceamento de classes e qualidade de anotação).
- Bounding boxes podem ficar imprecisas em objetos pequenos, movimentos rápidos e enquadramentos fora do padrão de treino.

## ⚠️ Status atual do modelo

O modelo `.eim` atual deve ser considerado um **protótipo experimental**.

- A pipeline de software (captura de webcam, inferência, renderização e tratamento de erros) está funcional.
- As principais limitações observadas estão concentradas na etapa de treinamento do modelo e na cobertura do dataset, e não na arquitetura geral da aplicação.
- Dependendo da posição do celular, postura do condutor e condições da cena, a detecção pode ocorrer com baixa confiança ou gerar classificações incorretas.

## 🧠 Roadmap de melhoria

- Expandir o dataset com maior diversidade de iluminação, câmera, ângulos e perfis de condutor.
- Melhorar a qualidade das anotações para tornar as bounding boxes mais consistentes.
- Incluir mais exemplos negativos (mão no rosto, objetos parecidos, gestos sem celular) para reduzir falsos positivos.
- Avaliar outras arquiteturas de object detection e diferentes configurações de treino/export no Edge Impulse.
- Adotar versionamento de modelos `.eim` com rastreabilidade de métricas por versão (ex.: precisão, recall, FPS).

## Melhorias futuras

- Filtro temporal para reduzir falsos positivos.
- Export de eventos (timestamp + confiança) em CSV/JSON.
- Modo headless (sem UI) para edge devices.
- Testes de integração com mock do runner e testes em hardware real.

## Arquitetura detalhada

Consulte `docs/ARCHITECTURE.md`.

## 🎓 Objetivo do projeto

Este projeto foi construído para demonstrar, de forma prática e profissional:

- Integração entre ML aplicado e engenharia de software.
- Implementação de pipeline de inferência em tempo real com webcam.
- Organização arquitetural modular e manutenção de código em camadas.
- Aplicação de visão computacional em um problema real de segurança no trânsito.
