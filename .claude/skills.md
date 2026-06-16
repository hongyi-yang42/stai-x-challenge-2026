{
  "skills": {
    "baseline-first": {
      "description": "Quickly produce a valid baseline submission from group means or global mean. Use as first step before any modeling.",
      "instructions": [
        "Read all data files in data/",
        "Identify target column from sample_submission.csv or train file",
        "Compute target mean by the strongest categorical grouping (or global mean if no grouping)",
        "Map predictions to test rows",
        "Write submission.csv matching sample format exactly",
        "Print confirmation with shape and head"
      ]
    }
  },
  "hooks": {
    "post-tool-use": []
  }
}
