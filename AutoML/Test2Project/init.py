def main(config):
    pass
# 多分类
# 多生成几个

if __name__ == "__main__":

    search_space = {
        "feature_extraction_method": {
            "_type": "choice",
            "_value": ["MorganFingerprint", "MACCSKeys", "RDKitDescriptors"],
        },
        "model_type": {
            "_type": "choice",
            "_value": ["RandomForest", "SVM", "DecisionTree"],
        },
        "RandomForest": {
            "n_estimators": {"_type": "randint", "_value": [50, 200]},
            "max_depth": {"_type": "choice", "_value": [10, 20, 30, null]},
            "min_samples_split": {"_type": "uniform", "_value": [0.01, 0.1]},
        },
        "SVM": {
            "C": {"_type": "loguniform", "_value": [1e-3, 1e3]},
            "kernel": {"_type": "choice", "_value": ["linear", "rbf", "poly"]},
        },
        "DecisionTree": {
            "max_depth": {"_type": "choice", "_value": [10, 20, 30, null]},
            "min_samples_split": {"_type": "uniform", "_value": [0.01, 0.1]},
        },
    }

    config = {
        "experimentName": "test_nas",
        "trialCommand": "python trial.py",
        "trialGpuNumber": 1,
        "trialConcurrency": 2,
        "maxTrialNumber": 100,
        "searchSpace": search_space,
        "tuner": {"name": "Random", "classArgs": {"optimize_mode": "maximize"}},
    }

    main(config)
