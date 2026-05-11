# Renewables in Electricity Markets - Assignment 2

This repository contains the code for solving Assignment 2: Market Participation (Stakeholder Perspective) in the course Renewables in Electricity Markets - 46755. 


## Project Structure

The two main entry points are:

- [Step1.py](Step1.py): Participation in Day-ahead and Balancing Markets

- [Step2.py](Step2.py): Participation in Ancillary Service Markets

- [usefulfunctions.py](usefulfunctions.py): All helper functions used by Step 1 and Step 2  are implemented here.
 
- [Data/](Data): contains the two input datasets used in Step 1 to generate hourly scenarios. The data files is described below.


## Data Files

### `Data/DayAheadPrices_DK2.csv`
This file contains historical hourly day-ahead electricity prices for DK2.

Columns:

- `HourUTC`: timestamp in UTC
- `HourDK`: timestamp in Danish local time
- `PriceArea`: bidding zone, here DK2
- `SpotPriceDKK`: spot price in DKK/MWh
- `SpotPriceEUR`: spot price in EUR/MWh


### `Data/ninja_wind_55.5783_15.7764_corrected.csv`
This file contains historical wind power data from Renewables.ninja for the location used in the assignment.

Columns:

- `time`: UTC timestamp
- `local_time`: local timestamp
- `electricity`: wind power output in MWh


## Requirements

The code is written in Python and uses the following packages:

- numpy
- pandas
- matplotlib
- gurobipy

A working Gurobi installation and valid license are required to run the optimization models.

