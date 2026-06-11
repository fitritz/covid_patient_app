from ucimlrepo import fetch_ucirepo
import pandas as pd

def fetch_and_save_dataset():
    # fetch dataset
    maternal_health_risk = fetch_ucirepo(id=863)

    # data (as pandas dataframes)
    X = maternal_health_risk.data.features
    y = maternal_health_risk.data.targets

    # concatenate features and targets
    df = pd.concat([X, y], axis=1)

    # save to csv
    df.to_csv('maternal_health_risk.csv', index=False)
    print("Dataset saved to maternal_health_risk.csv")

if __name__ == '__main__':
    fetch_and_save_dataset()
