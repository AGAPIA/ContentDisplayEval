using UnityEngine;
using System.Collections.Generic;
using System.IO;

namespace DatasetGenerator
{
    public class MeshAnomaliesGenerator : MonoBehaviour, IAnomalyGenerator
    {
        public enum MeshAnomalyType { Occlusion, Deformation, Floating }

        [Header("Target Settings")]
        public List<GameObject> targetObjects;

        [Header("Anomaly Settings")]
        public MeshAnomalyType anomalyToApply;
        public bool randomizeAnomaly = false;

        [Header("Logging & Output")]
        public string logPath = "mesh_anomaly_log.csv";

        private int screenshotCount = 0;

        void Start()
        {
            ApplyAnomalies();
        }

        public void ApplyAnomalies()
        {
            foreach (var obj in targetObjects)
            {
                MeshAnomalyType typeToApply = randomizeAnomaly ? GetRandomAnomaly() : anomalyToApply;
                ApplyAnomalyToObject(obj, typeToApply);
                LogAnomaly(obj, typeToApply);
                CaptureScreenshot(obj, typeToApply);
            }
        }

        private MeshAnomalyType GetRandomAnomaly()
        {
            int count = System.Enum.GetNames(typeof(MeshAnomalyType)).Length;
            return (MeshAnomalyType)Random.Range(0, count);
        }

        private void ApplyAnomalyToObject(GameObject target, MeshAnomalyType type)
        {
            switch (type)
            {
                case MeshAnomalyType.Occlusion:
                    target.transform.localScale *= 5f;
                    break;
                case MeshAnomalyType.Deformation:
                    target.transform.localScale = new Vector3(0.1f, 3f, 0.1f);
                    break;
                case MeshAnomalyType.Floating:
                    target.transform.position += new Vector3(0, 5, 0);
                    break;
            }
        }

        private void LogAnomaly(GameObject obj, MeshAnomalyType type)
        {
            string line = $"{System.DateTime.Now},{obj.name},{type}";
            File.AppendAllText(logPath, line + "\n");
        }

        private void CaptureScreenshot(GameObject obj, MeshAnomalyType type)
        {
            string safeName = obj.name.Replace(" ", "_");
            string filename = $"mesh_screenshot_{safeName}_{type}_{screenshotCount}.png";
            ScreenCapture.CaptureScreenshot(filename);
            screenshotCount++;
        }
    }
}
