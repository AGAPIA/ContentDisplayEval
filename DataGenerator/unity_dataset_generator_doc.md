
# Unity Dataset Generator for Visual Anomalies

This tool allows you to automatically apply visual anomalies to selected GameObjects in a Unity scene and capture screenshots and metadata for dataset generation.

## Overview

The system includes:

- **Anomaly generators** (e.g., `BadTexturesGenerator`, `MeshAnomaliesGenerator`) that apply specific types of visual errors to objects.
- **A unified interface (`IAnomalyGenerator`)** to standardize how generators apply anomalies, capture screenshots, and log metadata.
- **A batch runner (`AnomalyBatchRunner`)** that executes one or more anomaly generators in sequence.

## Folder Structure

```
Assets/
└── DatasetGenerator/
    ├── bad_textures_generator.cs
    ├── mesh_anomalies_generator.cs
    ├── anomaly_batch_runner.cs
```

---

## 1. How to Use

### Step 1: Add the `AnomalyBatchRunner` to a GameObject

1. In your Unity scene, create an empty GameObject (e.g., `DatasetManager`).
2. Attach the `AnomalyBatchRunner` script to it.

When the scene starts, this component will automatically run all attached generators.

### Step 2: Attach Anomaly Generators

1. Create additional GameObjects for each generator (e.g., `BadTexturesGenerator`, `MeshAnomaliesGenerator`).
2. Attach the appropriate generator script.
3. Configure its settings via the Unity Inspector.

Each generator implements the `IAnomalyGenerator` interface and will be called automatically.

If no generators are manually assigned in the `AnomalyBatchRunner`, the script will attempt to auto-fill the list using `FindObjectsOfType<MonoBehaviour>()`.

---

## 2. Configuring a Generator

### BadTexturesGenerator

#### Settings

- **Target Objects**: List of GameObjects to apply texture anomalies to.
- **Anomaly Type**: Choose one of:
  - `Stretched`
  - `Discolored`
  - `Placeholder`
  - `Clipping`
  - `GammaShift`
- **Randomize Anomaly**: If enabled, selects a random anomaly type for each object.
- **Log Path**: Path to save a `.csv` log of applied anomalies.

#### Output

- Annotated screenshots saved as `.png` in the project folder.
- Log file (`anomaly_log.csv`) listing object name, anomaly type, and timestamp.

---

## 3. Adding Your Own Generator

To integrate a new anomaly type:

1. Create a new script implementing the `IAnomalyGenerator` interface:
```csharp
public interface IAnomalyGenerator
{
    void ApplyAnomaly(GameObject obj, System.Enum anomalyType);
    void CaptureScreenshot(GameObject obj, System.Enum anomalyType);
    void LogAnomaly(GameObject obj, System.Enum anomalyType);
}
```

2. Implement anomaly logic, image capture, and logging.

3. Attach the script to a GameObject in the scene.

---

## 4. Screenshot Capture

Screenshots are taken using `ScreenCapture.CaptureScreenshot(...)`. They are named according to:

```
tex_screenshot_<ObjectName>_<AnomalyType>_<Index>.png
```

Ensure your game view is correctly configured (e.g., resolution and camera angle) when capturing.

---

## 5. Logging Format

Each anomaly is logged to a CSV file with the following structure:

```
<timestamp>,<object_name>,<anomaly_type>
```

Example:
```
2025-06-10 14:32:01,Object_01,Stretched
```

---

## 6. Running the Dataset Generator

You can start the dataset generation by entering Play mode. The `Start()` method in `AnomalyBatchRunner` will automatically call `ApplyAnomalies()` for each generator, producing images and logs.

---

## Notes

- All generators assume objects are properly placed and visible in the scene view.
- Clipping and geometry shifts may require camera angle adjustments for full visibility.
- Screenshot capture timing may need to be tuned (e.g., via `WaitForEndOfFrame`) for precise frames.
