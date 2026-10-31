# Model Folder

O modelo Edge Impulse (`cellphone_detector.eim`) já vem neste repositório.
Depois de clonar, só é preciso garantir a permissão de execução:

```bash
chmod +x model/cellphone_detector.eim
```

O SDK executa o `.eim` como subprocesso e conversa com ele por socket, por isso
o bit de execução é necessário.

Observações:

- O modelo é compilado para Linux x86_64.
- Não usar formato TFLite/PyTorch/ONNX neste projeto.
