using UnityEngine;
using System.Collections.Generic;
using System.IO;

namespace DatasetGenerator
{
    public class AnomalyBatchRunner : MonoBehaviour
    {
        [Header("Generators to Run")]
        public List<MonoBehaviour> generators;

        void Start()
        {
            // Auto-populate only if the list is empty
            if (generators == null || generators.Count == 0)
            {
                generators = new List<MonoBehaviour>();
                foreach (var comp in FindObjectsOfType<MonoBehaviour>())
                {
                    if (comp is IAnomalyGenerator)
                    {
                        generators.Add(comp);
                    }
                }
                Debug.Log($"[AnomalyBatchRunner] Auto-filled {generators.Count} generators from scene.");
            }

            RunAllGenerators();
        }

        private void RunAllGenerators()
        {
            foreach (var gen in generators)
            {
                if (gen is IAnomalyGenerator anomalyGen)
                {
                    anomalyGen.ApplyAnomalies();
                }
                else
                {
                    Debug.LogWarning($"Generator {gen.name} does not implement IAnomalyGenerator interface.");
                }
            }
        }
    }

    public interface IAnomalyGenerator
    {
        void ApplyAnomalies();
    }
}
