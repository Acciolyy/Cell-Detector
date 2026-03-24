# Architecture - edge-phone-usage-detector

## Goal

Detect cellphone usage in real time from webcam frames using an Edge Impulse `.eim` model on Linux.

## End-to-end flow

1. `src/main.py`
- Parses runtime arguments (`model-path`, `camera-index`, optional resolution, threshold, log level).
- Validates `.eim` path and configuration.
- Creates service instances and starts the pipeline.

2. `CameraService`
- Opens webcam with OpenCV.
- Optionally applies target capture resolution (`camera_width`, `camera_height`).
- Raises explicit exceptions when device is unavailable or frame read fails.

3. `EimInferenceService`
- Integrates with an Edge Impulse Linux Python SDK/runtime compatible with `.eim`.
- Attempts to initialize `ImageImpulseRunner` and load model in `start()`.
- Converts frames from BGR to RGB before inference.
- Parses `result.bounding_boxes` into normalized entities.
- Reads `model_info` defensively to discover expected input size (when available).

4. `DetectionPipeline`
- Orchestrates loop: capture -> inference -> render -> display.
- Renders bounding boxes, class label/confidence, FPS and latency.
- Rescales bounding boxes to webcam frame dimensions when model input size differs.
- Stops on `q` and always performs cleanup.

## `.eim` integration strategy

- The model is handled strictly as Edge Impulse `.eim` (not TFLite/ONNX/PyTorch).
- Runtime specifics may vary by Linux environment and SDK version.
- The implementation is defensive:
  - checks model file before startup;
  - provides explicit initialization/inference errors;
  - tolerates metadata schema variations in `model_info`.

## Key modules

- `src/app/config`: runtime settings and validation.
- `src/app/services`: camera, `.eim` inference, and orchestration.
- `src/app/domain`: normalized data entities (`BoundingBox`, `InferenceResult`).
- `src/app/utils`: path and logging helpers.

## Error handling strategy

- Missing `.eim` file -> startup failure with clear error.
- Webcam unavailable -> `CameraUnavailableError`.
- Runtime not installed or incompatible -> `InferenceInitializationError`.
- Runtime output mismatch or classify failure -> `InferenceExecutionError`.

## Testing strategy (current)

- Smoke tests validate parser/settings and project bootstrap assumptions.
- Integration validation must be done on Linux machine with webcam + `.eim` + compatible runtime.
