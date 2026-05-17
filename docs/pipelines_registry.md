# Pipeline Registry

| Name | Input Types | Entropy Estimator | Output Signature |
|------|--------------|-------------------|-------------------|
| pipeline1 | ["pd.DataFrame"] | shannon | {"entropy": "float"} |
| pipeline2 | ["List[Dict]"] | kullback_leibler | {"entropy": "float", "confidence": "float"} |