# This is a sample Python script.
import pandas as pd
from lifetimes import GammaGammaFitter
from lifetimes import BetaGeoFitter
from lifetimes.plotting import plot_frequency_recency_matrix
from lifetimes.utils import summary_data_from_transaction_data
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    df = pd.read_csv("data/data.csv")
    print(df.columns)

    print(df.head())
    # Get number of rows
    number_of_rows = df.shape[0]
    print(number_of_rows)

    # preprocess data
    df = df[df['CustomerID'].notna()]  # remove rows with missing customer IDs
    df = df[df['PurchaseValue'] > 0]  # remove rows with negative quantities
    df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.date

    #df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])  # convert invoice date to datetime

    number_of_rows = df.shape[0]
    print(number_of_rows)
    print(df.head())

    # create summary data from transaction data
    summary = summary_data_from_transaction_data(df,
                                                 customer_id_col='CustomerID',
                                                 datetime_col='Timestamp',
                                                 monetary_value_col='PurchaseValue',
                                                 observation_period_end=max(df["Timestamp"]))

    summary = summary[summary["monetary_value"] > 0]

    # fit the BG/NBD model
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(summary['frequency'], summary['recency'], summary['T'])

    # fit the Gamma-Gamma submodel
    ggf = GammaGammaFitter(penalizer_coef=0.0)
    ggf.fit(summary['frequency'], summary['monetary_value'])

    # predict customer lifetime value
    summary['predicted_purchases'] = bgf.predict(30, summary['frequency'], summary['recency'], summary['T'])
    summary['predicted_clv'] = ggf.customer_lifetime_value(bgf,
                                                           summary['frequency'],
                                                           summary['recency'],
                                                           summary['T'],
                                                           summary['monetary_value'],
                                                           time=1,  # the lifetime expected for the user in months
                                                           freq='D',
                                                           discount_rate=0.01)
    summary["estimated_monetary_value"] = ggf.conditional_expected_average_profit(
        summary['frequency'],
        summary['monetary_value']
    )

    print(summary.head())

# calculate days between two dates


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
