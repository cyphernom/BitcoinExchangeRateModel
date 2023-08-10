#(c)2020-2023 Nick Btconometrics
import argparse
import datetime as dt
from data_preprocessing import DataPreprocessing
from model import Model
from visualisation import Visualisation
from analysis import Analysis

def print_descriptions():
    print("USAGE: python baerm.py -o {1,2,3}")
    print(" [1] - Plot BAERM chart")
    print(" 2 - Plot rolling R-squared")
    print(" 3 - Plot coefficient evolution")

def main():
    parser = argparse.ArgumentParser(description="Choose the option to run", add_help=False)
    parser.add_argument("-o","--option", type=int, choices=[1, 2, 3], default=1, help="Choose option 1, 2, or 3")
    parser.add_argument('-h', '--help', action='store_true', help="Show this help message and exit")
    args = parser.parse_args()

    if args.help:
        print_descriptions()
        exit()

    url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
    data_preprocessing = DataPreprocessing(url)
    data_preprocessing.load_and_preprocess_data()

    model = Model(data_preprocessing.df)
    coefs, ols_summary = model.run_regression()

    print(coefs)
    
    model.calculate_YHAT()
    model.calculate_YHATs2()

    analysis = Analysis(data_preprocessing.df)
    R2_oos, R2_oos2 = analysis.calculate_oos_r_squared_values()
    print(f"Out-of-sample R-squared\n Base Model: {R2_oos} \n Damped Model: {R2_oos2}")

    analysis.calculate_rolling_r_squared()

    visualisation = Visualisation(data_preprocessing.df)
    plot_df = data_preprocessing.df[(data_preprocessing.df['date'] > dt.datetime.strptime('2011-01-01', '%Y-%m-%d')) & (data_preprocessing.df['date'] < dt.datetime.strptime('2026-01-01', '%Y-%m-%d'))]

    if args.option == 1:
        visualisation.plot_charts(plot_df)
    elif args.option == 2:
        visualisation.plot_rolling_r_squared(data_preprocessing.df)
    elif args.option == 3:
        coefficients, dates = model.study_coefficient_evolution()
        visualisation.plot_coefficient_evolution(coefficients, dates)
    
if __name__ == "__main__":
    main()
