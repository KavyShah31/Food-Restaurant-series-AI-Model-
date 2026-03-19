import pandas as pd

def load_data(path):
    """Load processed time-series dataset"""
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def basic_info(df):
    """Print basic dataset info"""
    print("Shape:", df.shape)
    print("\nDate Range:", df['date'].min(), "to", df['date'].max())
    print("\nSummary:\n", df.describe())

def main():
    print("🚀 Food Demand Forecasting Project\n")

    # Load processed data
    df = load_data('data/processed/daily_demand.csv')

    # Show info
    basic_info(df)

    print("\n✅ Data loaded successfully. Ready for modeling!")

if __name__ == "__main__":
    main()