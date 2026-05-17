# Analysis Pipelines

.. mermaid::
   flowchart TD
      QRNG_source[QRNG source] --> Preprocessing
      Preprocessing --> EntropyAnalysis[Entropy/Chaos/Consciousness Analysis]
      EntropyAnalysis --> AnomalyDetection[Anomaly Detection]
      AnomalyDetection --> Inference

Cross-references to testbed pipelines:
   - Preprocessing: :ref:`testbed.preprocessing_pipeline`
   - Entropy/Chaos/Consciousness Analysis: :ref:`testbed.entropy_analysis_pipeline`
   - Anomaly Detection: :ref:`testbed.anomaly_detection_pipeline`
   - Inference: :ref:`testbed.inference_pipeline`