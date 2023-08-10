#(c)2020-2023 Nick Btconometrics
from data_preprocessing import DataPreprocessing
from model import Model
from visualisation import Visualisation
from analysis import Analysis
import datetime as dt

def main():
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
    print(f"Out-of-sample R-squared\n Base Model: {R2_oos} \n Phased Model: {R2_oos2}")

    analysis.calculate_rolling_r_squared()

    visualisation = Visualisation(data_preprocessing.df)
    plot_df = data_preprocessing.df[(data_preprocessing.df['date'] > dt.datetime.strptime('2011-01-01', '%Y-%m-%d')) & (data_preprocessing.df['date'] < dt.datetime.strptime('2026-01-01', '%Y-%m-%d'))]
    visualisation.plot_charts(plot_df)
    visualisation.plot_rolling_r_squared(data_preprocessing.df)

    #uncomment to run the coefficient study
   # coefficients, dates = model.study_coefficient_evolution()
   # visualisation.plot_coefficient_evolution(coefficients, dates)
    
if __name__ == "__main__":
    main()
