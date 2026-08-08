# Architecture - edge-phone-usage-detector

## Goal

Detect cellphone usage in real time from webcam frames using an Edge Impulse
`.eim` model on Linux, or from a recorded video file when no camera is
available.

## End-to-end flow

```text
FrameSource --> EimInferenceService --> DetectionPipeline --> display
 (camera |        preprocess +            draw boxes +          and/or
  file)           classify               HUD overlay         VideoRecorder
```

Every arrow is timed independently by `FrameProfiler`. That is deliberate: the
`.eim` runner reports only its own native inference time, which is a small
fraction of the frame and cannot be used to reason about frame rate.

### 1. `src/main.py`

- Parses runtime arguments: model path, frame source, capture options,
  threshold, recording, display, profiling budget, log level.
- Warns when `--camera-*` options are passed together with `--source`, since
  they are ignored in file mode. Detection inspects `argv` rather than
  comparing against parser defaults, because `--camera-index 0` is
  indistinguishable from the default.
- Validates the `.eim` path and the configuration, then selects the frame
  source implementation and starts the pipeline.
- Names the missing module when a runtime dependency fails to import.

### 2. `CameraService`

- Opens the webcam with OpenCV.
- Applies, in this order, FOURCC then frame size: changing the pixel format can
  reset a negotiated resolution on V4L2 backends.
- Requests `CAP_PROP_BUFFERSIZE = 1`. A deeper driver queue hands back frames
  captured several periods ago, adding latency without adding throughput. Some
  backends ignore the request, which is why the producer below also drops.
- Reports the resolution actually negotiated and the driver frame rate.
- Raises explicit exceptions when the device is unavailable or a read fails.

### 3. `FrameSource` implementations

`FrameSource` is the abstraction the pipeline consumes through. There are two
implementations rather than one class with mode flags, because they differ in
the one place that matters: what to do with a frame the consumer was too slow
to take.

#### `ThreadedFrameSource` — live webcam

- Runs the blocking `cap.read()` in a producer thread. Inline, that call makes
  the whole pipeline advance at the camera's period no matter how fast
  inference is.
- Keeps exactly one frame, the newest. Older frames are **dropped, not
  queued**: in real time a stale frame is worthless, and queueing it would only
  accumulate latency.
- Tolerates transient read failures (USB re-enumeration); only a sustained run
  of them stops the producer and surfaces the error to the consumer.
- `output_fps` is `None`: a camera has no inherent rate to record at.

#### `VideoFileFrameSource` — recorded file

- Decodes in a producer thread too, but through a bounded **blocking** queue:
  nothing is dropped. A dropped frame from a file is lost content, not stale
  content. If the consumer falls behind, the decoder waits.
- Probes the first frame during `start()`, so an unreadable file fails at
  startup with a clear message instead of mid-run. The probed frame is kept and
  handed to the consumer first, so the check consumes nothing.
- Raises `EndOfStream` when the file runs out. That is not an error: it is the
  normal way a file-backed run finishes.
- `output_fps` is the file's own rate, read from `CAP_PROP_FPS`, falling back
  to 30 with a warning when the container reports something unusable.
- Reports its stage as `decode`, so disk and codec cost is never averaged into
  camera capture cost.

### 4. `EimInferenceService`

- Integrates with the Edge Impulse Linux Python SDK via `ImageImpulseRunner`.
- Splits work into `preprocess()` and `classify()` so the pipeline can time
  feature extraction and native inference independently. `infer()` remains as a
  thin wrapper over both.
- Downscales to model input size **before** converting colour, so no per-pixel
  work happens at full resolution. For `fit-shortest` the crop-and-scale is
  reproduced exactly, which also yields the `ModelViewport` needed to map
  detections back onto the frame. Any other resize mode is left to the SDK
  rather than guessed at.
- Uses the vectorized `get_features_from_image_auto_studio_settings()`. The
  legacy `get_features_from_image()` encodes features in a per-pixel Python
  loop, which measured ~5 ms per frame on its own.
- Parses `result.bounding_boxes` into normalized entities.
- Reads `model_info` defensively to discover the input size and the Studio
  resize mode, both of which vary by SDK version.

### 5. `DetectionPipeline`

- Orchestrates the loop: source -> preprocess -> inference -> draw ->
  record/display.
- Times every stage via `FrameProfiler`. The iteration that discovers end of
  stream is discarded from the statistics: it costs time but is not a frame,
  and counting it would understate every stage's share.
- Renders bounding boxes plus a compact HUD: FPS as a moving average over the
  last 30 frames, total frame time, and the capture/inference/render
  breakdown. Instantaneous `1/dt` is too noisy to read.
- Maps boxes through `ModelViewport`, which carries both the scale and the
  centre-crop offset the model actually saw. Scaling by input size alone
  stretches boxes across the full frame.
- Stops on `q`, on end of stream, on a frame budget (`--profile-frames`), or on
  exception, and always releases the producer thread, the recorder and the
  runner.

### 6. `VideoRecorder`

- Writes annotated frames to mp4 with `cv2.VideoWriter`, so demo material
  carries no window chrome, cursor or screen-recorder artefacts.
- The writer needs its frame rate up front. For a camera the honest rate is
  only known after the loop runs, so warm-up frames are buffered and flushed
  once it is measured — nothing from the start of the run is lost.
- For a file-backed source the rate is known up front and inherited from the
  input, with no warm-up. Recording a file at the machine's processing rate
  would play the demo back fast or slow.

## Behavioural differences by source

| | Live webcam | Video file |
| --- | --- | --- |
| Late frames | Dropped; only the newest is kept | None dropped; all processed in order |
| End of source | Does not exist; ends on `q` | Ends on its own, cleanly |
| Recording rate | Measured processing rate | Inherited from the input file |
| Profiler stage | `capture` | `decode` |

## `.eim` integration strategy

- The model is handled strictly as Edge Impulse `.eim` (not TFLite/ONNX/PyTorch).
- The `.eim` file must be executable: the SDK runs it as a subprocess and talks
  to it over a Unix socket.
- Runtime specifics vary by Linux environment and SDK version, so the
  implementation is defensive: it checks the model file before startup, raises
  explicit initialization and inference errors, and tolerates metadata schema
  variations in `model_info`.

## Key modules

- `src/app/config`: runtime settings, defaults and validation.
- `src/app/services`: camera, frame sources, `.eim` inference, recording and
  orchestration.
- `src/app/domain`: normalized entities (`BoundingBox`, `ModelViewport`,
  `InferenceResult`).
- `src/app/utils`: path, logging and profiling helpers (`FrameProfiler`,
  `StageStats`, `MovingAverageFps`).

## Error handling strategy

- Missing `.eim` file -> startup failure with a clear error.
- Webcam unavailable -> `CameraUnavailableError`.
- Frame read failure in the producer thread -> `CameraReadError` at the
  consumer. Transient failures are tolerated; only a sustained run stops it.
- Video file missing, a directory, unsupported by the codec, or containing no
  readable frames -> `VideoSourceError`, each with a message naming the actual
  problem.
- End of a video file -> `EndOfStream`, handled as a normal termination.
- Runtime not installed or incompatible -> `InferenceInitializationError`.
- Runtime output mismatch or classify failure -> `InferenceExecutionError`.
- Output video file cannot be opened -> `VideoRecordingError`.
- Missing Python dependency -> startup failure naming the missing module.

Cleanup runs from a `finally` and must survive being called while an exception
is already propagating, so each step is guarded independently: one failing
release never masks the original cause or skips the remaining releases.

## Testing strategy (current)

- `tests/test_smoke.py`: parser and settings bootstrap assumptions.
- `tests/test_profiling.py`: percentile ranking, rolling windows, stage
  aggregation, frame discarding, and FPS smoothing driven by an injected clock
  so assertions are deterministic.
- `tests/test_frame_source.py`: capture with `cv2.VideoCapture` mocked — that
  the driver buffer is set to 1, that old frames are dropped rather than
  queued, that the newest frame is the one delivered, and that the thread joins
  and the device is released on stop, on exception, and on repeated stops.
- `tests/test_video_file_source.py`: a synthetic clip written with
  `cv2.VideoWriter` inside the test — that every frame is delivered in order
  with no drops, that end of file terminates cleanly, that the input frame rate
  is reported, and that each invalid-path case raises its own message.
- `tests/test_video_recorder.py`: warm-up buffering, forced rate inheritance,
  rate clamping, and unwritable destinations.
- `tests/test_cli.py`: source flag parsing, camera-flag warning detection, and
  the missing-dependency message.

Real `.eim` inference is not unit tested. Integration validation must be done
on a Linux machine with a webcam, the `.eim` file and a compatible runtime; a
video file via `--source` covers the pipeline where no camera is exposed.
