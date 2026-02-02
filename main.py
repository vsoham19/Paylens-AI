from pipeline.data_pipeline import DataPipeline


def main():
    pipeline = DataPipeline("config/config.yaml")
    pipeline.run()
    model_path = "artifacts/models/linear_regression.pkl"


if __name__ == "__main__":
    main()
