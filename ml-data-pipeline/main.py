from data_loader import load_csv, validator


data_path = "data/sample.csv"

def main():
    df  = load_csv(data_path)
    print("data loaded successfully")
    print(df.head())

if __name__ == '__main__':
    main()