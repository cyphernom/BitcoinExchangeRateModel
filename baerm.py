#Copyright (c) 2020-2023 Btconometrics
import tkinter as tk
from tkinter import simpledialog
from data_preprocessing import DataPreprocessing
from model import Model
from visualisation import Visualisation
from analysis import Analysis
import datetime as dt
from tkinter import ttk
import tkcalendar
from tkcalendar import DateEntry

def on_option1_click():
    execute_option(1)

def on_option2_click():
    execute_option(2)

def on_option3_click():
    execute_option(3)

def on_option4_click():
    execute_option(4)

def execute_option(option):
    url = data_source_entry.get()
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


    selected_date = date_picker.get_date()  # Get the date as a datetime.date object

    plot_df = data_preprocessing.df[
        (data_preprocessing.df['date'] > dt.datetime.strptime('2011-01-01', '%Y-%m-%d')) &
        (data_preprocessing.df['date'] < dt.datetime(selected_date.year, selected_date.month, selected_date.day))]

    if option == 1:
        visualisation.plot_charts(plot_df)
    elif option == 2:
        visualisation.plot_rolling_r_squared(data_preprocessing.df)
    elif option == 3:
        coefficients, dates = model.study_coefficient_evolution()
        visualisation.plot_coefficient_evolution(coefficients, dates)
    elif option == 4: # New option
        all_yhats, dates = model.study_yhats()
        visualisation.plot_all_yhats_with_rainbow(all_yhats, dates) 
     
        
root = tk.Tk()
root.title("BAERM Project")


# Create a frame for the buttons and date selector
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10, pady=10)  # Pack the left_frame into the main window

# Buttons for options
option1_button = tk.Button(left_frame, text="Plot BAERM chart", command=on_option1_click)
option1_button.pack(side=tk.TOP, anchor='w')

option2_button = tk.Button(left_frame, text="Plot rolling R-squared", command=on_option2_click)
option2_button.pack(side=tk.TOP, anchor='w')

option3_button = tk.Button(left_frame, text="Plot coefficient evolution", command=on_option3_click)
option3_button.pack(side=tk.TOP, anchor='w')

option4_button = tk.Button(left_frame, text="Plot all YHAT lines", command=on_option4_click)
option4_button.pack(side=tk.TOP, anchor='w')

# Date picker
date_frame = ttk.Frame(left_frame)
date_frame.pack(side=tk.TOP, anchor='w', padx=10, pady=10)

date_label = ttk.Label(date_frame, text="Plot to date:")
date_label.pack(side=tk.LEFT)

date_picker = tkcalendar.DateEntry(date_frame, date_pattern='y-mm-dd')
date_picker.pack(side=tk.LEFT)

# Text field for data source
data_source_label = tk.Label(left_frame, text="Enter Data Source:")
data_source_label.pack(side=tk.TOP, anchor='w')  # Pack the label into the left_frame

data_source_entry = tk.Entry(left_frame)
data_source_entry.insert(0, "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv") # Default URL
data_source_entry.pack(side=tk.TOP, anchor='w', padx=10, pady=10)  # Pack the entry into the left_frame

root.mainloop()
