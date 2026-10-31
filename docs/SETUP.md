# Instalação e uso

Detalhes para rodar o projeto localmente. A visão geral está no
[README](../README.md) e a arquitetura em [ARCHITECTURE.md](ARCHITECTURE.md).

## Estrutura do projeto

```text
.
├── assets/examples/          # Mídias para demonstração (gif/prints)
├── docs/ARCHITECTURE.md      # Arquitetura técnica e fluxo
├── docs/SETUP.md             # Instalação, execução e flags
├── model/                    # Modelo Edge Impulse .eim
├── scripts/run_dev.sh        # Script de execução local
├── src/
│   ├── main.py               # Entrypoint
│   └── app/
│       ├── config/           # Configurações
│       ├── domain/           # Entidades de domínio
│       ├── services/         # Câmera, fontes de vídeo, inferência e pipeline
│       └── utils/            # Logging, paths e profiling
├── tests/                    # Testes
├── requirements.txt          # Dependências Python base (sem runtime Edge Impulse fixo)
└── requirements-dev.txt      # Dependências de desenvolvimento (pytest)
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

Para rodar os testes, instale também as dependências de desenvolvimento:

```bash
pip install -r requirements-dev.txt
python -m pytest
```

### 4) Instalar runtime Edge Impulse compatível com `.eim`

A parte de inferência `.eim` depende de runtime/SDK Edge Impulse disponível no seu Linux. O nome exato do pacote e versão podem variar.

Exemplo comum (quando disponível no seu ambiente):

```bash
pip install edge_impulse_linux
```

Se esse pacote não estiver disponível para sua plataforma, consulte a documentação oficial do Edge Impulse para Linux e instale o runner Python equivalente.

## Modelo `.eim`

O modelo já vem no repositório, em:

```bash
model/cellphone_detector.eim
```

Ele precisa de permissão de execução, porque o SDK o executa como subprocesso e
conversa por socket:

```bash
chmod +x model/cellphone_detector.eim
```

Observações:

- Este projeto **não** trata o modelo como TFLite/ONNX/PyTorch.
- A inferência é feita por runtime Edge Impulse compatível com `.eim`.
- O modelo é compilado para Linux x86_64.

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
  --camera-width 640 \
  --camera-height 480 \
  --min-confidence 0.5
```

Controles:

- `q` para encerrar.

> O modelo recebe 96x96. Capturar acima de 640x480 não acrescenta precisão e
> consome banda USB, que é justamente o gargalo do laço — veja a investigação
> de desempenho no [README](../README.md#desafios-e-o-que-aprendi).

### Opção 3: a partir de um arquivo de vídeo

Útil onde não há webcam exposta — WSL2, por exemplo, não traz V4L2/UVC no
kernel — e para validar a pipeline sobre material gravado externamente.

```bash
python -m src.main --source entrada.mp4 --record anotado.mp4
```

Nesse modo as opções `--camera-*` são ignoradas (com aviso no log), e a
diferença de comportamento em relação à webcam é intencional:

| | Webcam ao vivo | Arquivo |
| --- | --- | --- |
| Quadros atrasados | Descartados (só o mais recente importa) | Nenhum descarte: todos processados em ordem |
| Fim da fonte | Não existe; encerra com `q` | Encerra sozinho, sem erro |
| FPS da gravação | Taxa de processamento medida | Herdado do arquivo de entrada |
| Etapa no profiling | `capture` | `decode` |

O FPS de saída herda o da entrada de propósito: gravar no ritmo de
processamento faria a demonstração sair acelerada ou lenta. Se o container
reportar um FPS inválido, o fallback é 30 e fica registrado no log.

### Gravar vídeo de demonstração

Grava os quadros já anotados direto em mp4, no FPS real medido. Sem screen
recording: não há barra de título, cursor nem borda de janela no resultado.

```bash
python -m src.main --record demo.mp4
```

Para gerar material em modo headless (também útil para medir quanto o `imshow`
está custando):

```bash
python -m src.main --record demo.mp4 --no-display
```

### Medir o desempenho

Processa exatamente N quadros e imprime média, mediana e p95 de cada etapa do
laço, mais o FPS efetivo:

```bash
python -m src.main --profile-frames 200
```

```text
stage              n   mean ms   median ms    p95 ms    share
-------------------------------------------------------------
capture          200     20.20       20.65     23.40    60.5%
preprocess       200      1.49        1.46      2.18     4.5%
inference        200      2.83        2.61      3.73     8.5%
draw             200      0.31        0.27      0.51     0.9%
display          200      5.11        4.89      6.60    15.3%
-------------------------------------------------------------
frame total      200     33.39       33.31     36.61   100.0%
```

O overlay da janela mostra apenas números medidos por este laço: FPS por média
móvel dos últimos 30 quadros (não instantâneo), tempo total do quadro em ms e a
decomposição captura / inferência / render.

## Flags adicionais

| Flag | Efeito |
| --- | --- |
| `--source CAMINHO` | Lê quadros de um arquivo de vídeo em vez da webcam. |
| `--record CAMINHO.mp4` | Grava os quadros anotados. FPS medido para webcam, FPS da entrada para arquivo. |
| `--no-display` | Roda sem janela OpenCV (headless). Exige `--record` ou `--profile-frames`. |
| `--profile-frames N` | Processa N quadros, imprime o relatório de etapas e sai. |
| `--camera-fourcc MJPG` | Formato de captura. Útil quando o driver escolhe YUYV e limita o frame rate. |
| `--fps-window N` | Quadros usados na média móvel de FPS do overlay (padrão 30). |

As flags principais (`--model-path`, `--camera-index`, `--camera-width`,
`--camera-height`, `--min-confidence`) aparecem nos exemplos acima.
`python -m src.main --help` lista todas.

## Notas de execução

- `CAP_PROP_BUFFERSIZE` é uma sugestão: vários backends a ignoram
  silenciosamente. A thread produtora com descarte cobre esse caso.
- Se a sua câmera suportar MJPG, `--camera-fourcc MJPG` remove o limite de
  banda do YUYV em USB 2.0, às custas de decodificação; nem todo driver aceita.
- A gravação buffeiriza os primeiros 30 quadros para descobrir o FPS real antes
  de abrir o `VideoWriter`. Numa cena com frame rate muito instável, o arquivo
  fica com um FPS fixo que representa a média, não a variação.
- `imshow` custa ~5 ms por quadro (medido sob WSLg/X11). `--no-display`
  elimina esse custo para gravação e medição.
