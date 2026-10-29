# AI-Assisted Skin Lesion Screening Instrument

**MobileNetV2 deep features + four machine-learning classifiers + Arduino/NeoPixel user interface**

A GitHub-ready **research prototype** for 3D-printed, USB webcam-based, button-triggered skin-lesion image classification. A computer performs all ML inference; Arduino provides the physical pushbutton, haptic signal and colored NeoPixel status.

> **Not a medical device or diagnostic tool.** This research demonstrator must not be used to diagnose cancer, make clinical treatment decisions, or reassure someone that a lesion is harmless. Webcam images differ from dermoscopy images used for algorithm evaluation. No prospective clinical validation is claimed.

## What is included

| Path | Purpose |
|---|---|
| `app.py` | Desktop GUI (Windows/VS Code), camera preview and COM-port selection |
| `skin_screening/core.py` | Local `.keras` + `.pkl` inference, ROI preprocessing, 10-frame averaging |
| `skin_screening/serial_interface.py` | Serial port enumeration and Arduino message protocol |
| `arduino/SkinLesionScreening/SkinLesionScreening.ino` | Button, NeoPixel, vibration motor and serial firmware |
| `tools/create_feature_extractor.py` | **One-time** pretrained MobileNetV2 download and local `.keras` export |
| `tools/check_artifacts.py` | Check three required legacy model files |
| `tools/test_webcam.py` | Standalone USB webcam and ROI test |
| `tools/test_arduino_serial.py` | Manually send LED commands and monitor button messages |
| `tools/predict_single_image.py` | One-image research inference utility |
| `arduino/HardwareSelfTest/HardwareSelfTest.ino` | Arduino-only test of button, motor, and NeoPixel colors |
| `training/train_models.py` | Train Fine KNN, Fine Tree, Logistic Regression and Neural Network |
| `training/evaluate_models.py` | Train/test confusion matrices, ROC/PR, PCA and metric CSV |
| `training/shap_fine_knn.py` | Approximate test-set SHAP importance, summary and dependence |
| `tools/export_results_zip.py` | ZIP all plots and metrics from `results/` |
| `notebooks/Skin_Lesion_Training_Colab.ipynb` | Colab walkthrough and result download |
| `requirements-runtime.txt`, `requirements-research.txt` | Separate inference and research dependencies |

**This is a reconstructed, runnable repository based on the established project workflow, not a recovered byte-for-byte copy of every earlier script.** Existing trained `.pkl` and `.keras` artifacts are **not included** and should be supplied by the project owner.

## Hardware connection

| Device | Arduino pin / interface |
|---|---|
| Momentary pushbutton (other leg to GND) | D4 (INPUT_PULLUP) |
| Vibration motor control **through a MOSFET/transistor** | D2 |
| NeoPixel ring data, 16 addressable LEDs | D6 |
| Arduino USB | Host PC, 115200 baud |
| USB webcam | Host PC (separate USB connection) |

A motor cannot be driven directly from Arduino GPIO. Use a properly rated switching transistor, flyback diode, power source, and common ground. Confirm NeoPixel current and voltage compatibility; do not power a high-current full-brightness ring from a GPIO pin.

### Serial protocol

- Arduino → Python: `ARDUINO_READY` at startup, `CAPTURE` on debounced button press.
- Python → Arduino: `BENIGN` (green), `MALIGNANT` (red), `UNCERTAIN` (yellow), `RESET` (blue).
- Default NeoPixel ring = **blue**. Every physical button press triggers a **300 ms vibration pulse**.
- An inference error is reported as **UNCERTAIN**, not a benign or malignant prediction.

## Install in VS Code or Visual Studio on Windows

Use **Python 3.11** and a **dedicated virtual environment**, especially if a separate MediaPipe project is installed; TensorFlow and MediaPipe have had incompatible `protobuf` requirements.

```powershell
cd Skin-Lesion-AI-Screening-GitHub
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-runtime.txt
```

**Arduino IDE**: install the **Adafruit NeoPixel** library through **Tools → Manage Libraries**. This is an Arduino library, **not** a `pip` dependency.

Before loading the model, test the camera and controller separately:

```powershell
python tools/test_webcam.py --camera 0
python tools/test_arduino_serial.py --port COM4 --color GREEN
```

For a quick Arduino-only test, upload `arduino/HardwareSelfTest/HardwareSelfTest.ino` before replacing it with the final `SkinLesionScreening.ino` firmware.

## Reuse your original pretrained models (preferred)

Copy these existing files into `models/`:

```
models/
  mobilenet_feature_extractor.keras
  benign_malignant_model_fine_knn.pkl
  benign_malignant_model_fine_knn_metadata.pkl
```

Check files:

```powershell
python tools/check_artifacts.py
```

If only the feature-extractor file is missing, **run once**:

```powershell
python tools/create_feature_extractor.py
```

This **one-time creation** is the only step that can download ImageNet weights. The GUI **never downloads MobileNetV2**. If Windows downloads are unreliable, create the extractor in Colab and copy the `.keras` file into `models/`.

**Security:** do not open `.pkl` or `.keras` models from unknown sources. The scikit-learn version used to save the pickle should be matched as closely as possible when loading it.

## Start the webcam + Arduino application

1. Upload the `.ino` sketch to Arduino using Arduino IDE.
2. Connect the webcam and Arduino to the PC.
3. Close Arduino Serial Monitor (otherwise the COM port may be busy).
4. Start:

```powershell
python app.py --camera 0
```

Select the Arduino COM port in the GUI, click **Connect**, and press the physical pushbutton. Each press initiates acquisition of **10 NEW ROI frames** (default ROI `[150, 100, 450, 400]`) after the trigger, extracts MobileNetV2 features, averages per-frame classifier scores, and sends the displayed result to Arduino.

Other camera/ROI choices:

```powershell
python app.py --camera 1 --roi 100 50 500 450 --frames 10 --threshold 0.60
```

The **0.60 uncertainty cutoff is a heuristic**, not a clinically calibrated threshold; k=1 Fine KNN can produce very sharp per-frame probabilities. It must not be presented as a validated cancer risk estimate.

## Train models and produce figures in Colab

Optional: start with the included `notebooks/Skin_Lesion_Training_Colab.ipynb` notebook. The research scripts use four pre-defined algorithms but do **not** assert that retraining will match previously reported accuracy/AUC.

Arrange your files as shown in [`data/README.md`](data/README.md), with `train/benign`, `train/malignant`, `test/benign`, `test/malignant` folders. Train/test must be independent by **patient/lesion**, including across augmented versions. A simple per-image split is inadequate for clinical-grade research.

```bash
python -m pip install -r requirements-research.txt
python tools/create_feature_extractor.py
python training/train_models.py --data data
python training/evaluate_models.py
python training/shap_fine_knn.py --background 8 --explain 6 --nsamples 256
python tools/export_results_zip.py
```

All actual newly calculated values are saved in `results/figures/performance_metrics_train_and_test.csv`. The curves/SHAP figures are generated from the trained models and data at run time. **They are not synthetic reproductions of published plots.**

## Notes for scientific reproducibility

- HAM10000 has **10,015 original images**. Any derived dataset with more samples requires an explicit augmentation and provenance account.
- The original HAM10000 diagnosis categories do not map automatically to benign/malignant; document clinical labels and how ambiguous/premalignant groups were treated.
- Keep images of the same lesion/patient, including augmented versions, entirely in **one data partition**.
- Verify label semantics, feature scaling, normalization, original model versions, and test-set independence before comparing numbers across studies.
- Test-set SHAP relates to 1,280 MobileNetV2 learned features; it does not directly explain skin structures or offer clinical causality.
- Dataset results do not establish USB-webcam diagnostic effectiveness. Validate the prototype separately, including lighting, camera quality, reliability, timing, ergonomics and clinical oversight.
- The included application does not save patient images to disk. Obtain consent and follow ethics approval conditions for any new image acquisition.

## Suggested GitHub topics

`skin-lesion-classification`, `mobilenetv2`, `arduino`, `neopixel`, `medical-imaging-research`, `explainable-ai`, `python`, `opencv`

## Dataset and foundational references

- Tschandl et al. (2018), HAM10000, DOI: `10.1038/sdata.2018.161`.
- Sandler et al. (2018), MobileNetV2, DOI: `10.1109/CVPR.2018.00474`.
- Lundberg & Lee (2017), SHAP / interpretable prediction methods (cite the original publication when submitting research).

**License:** No reuse license is granted by default; add an appropriate `LICENSE` after checking authorship and institutional intellectual-property requirements.
