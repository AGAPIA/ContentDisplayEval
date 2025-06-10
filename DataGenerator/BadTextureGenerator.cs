using UnityEngine;
using System.Collections.Generic;
using System.IO;

namespace DatasetGenerator
{
    public interface IAnomalyGenerator
    {
        void ApplyAnomaly(GameObject obj, System.Enum anomalyType);
        void CaptureScreenshot(GameObject obj, System.Enum anomalyType);
        void LogAnomaly(GameObject obj, System.Enum anomalyType);
    }

    public class BadTexturesGenerator : MonoBehaviour, IAnomalyGenerator
    {
        public enum AnomalyType { Stretched, Discolored, Placeholder, Clipping, GammaShift }

        [Header("Target Settings")]
        public List<GameObject> targetObjects;

        [Header("Anomaly Settings")]
        public AnomalyType anomalyToApply;
        public bool randomizeAnomaly = false;

        [Header("Logging & Output")]
        public string logPath = "anomaly_log.csv";

        private int screenshotCount = 0;

        void Start()
        {
            ApplyAnomalies();
        }

        private void ApplyAnomalies()
        {
            foreach (var obj in targetObjects)
            {
                AnomalyType typeToApply = randomizeAnomaly ? GetRandomAnomaly() : anomalyToApply;
                ApplyAnomaly(obj, typeToApply);
                LogAnomaly(obj, typeToApply);
                CaptureScreenshot(obj, typeToApply);
            }
        }

        private AnomalyType GetRandomAnomaly()
        {
            int count = System.Enum.GetNames(typeof(AnomalyType)).Length;
            return (AnomalyType)Random.Range(0, count);
        }

        public void ApplyAnomaly(GameObject target, System.Enum anomaly)
        {
            AnomalyType type = (AnomalyType)anomaly;
            Renderer renderer = target.GetComponent<Renderer>();
            if (renderer == null || renderer.material == null)
                return;

            Material mat = renderer.material;

            switch (type)
            {
                case AnomalyType.Stretched:
                    mat.mainTextureScale = new Vector2(10, 0.1f);
                    break;
                case AnomalyType.Discolored:
                    mat.color = Color.magenta;
                    break;
                case AnomalyType.Placeholder:
                    mat.mainTexture = Texture2D.blackTexture;
                    break;
                case AnomalyType.Clipping:
                    target.transform.position += new Vector3(0, -1000, 0);
                    break;
                case AnomalyType.GammaShift:
                    mat.color *= 0.2f;
                    break;
            }
        }

        public void LogAnomaly(GameObject obj, System.Enum anomaly)
        {
            AnomalyType type = (AnomalyType)anomaly;
            string line = $"{System.DateTime.Now},{obj.name},{type}";
            File.AppendAllText(logPath, line + "\n");
        }

        public void CaptureScreenshot(GameObject obj, System.Enum anomaly)
        {
            AnomalyType type = (AnomalyType)anomaly;
            string safeName = obj.name.Replace(" ", "_");
            string filename = $"tex_screenshot_{safeName}_{type}_{screenshotCount}.png";
            ScreenCapture.CaptureScreenshot(filename);
            screenshotCount++;
        }
    }
}
