import pandas as pd

def main():
    df = pd.read_csv("scaler_pairs.csv")
    print(df["mbyy"].loc[0])


if __name__ == "__main__":
    main()