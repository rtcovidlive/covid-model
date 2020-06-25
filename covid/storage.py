def get_inference_data_key(run_date, region, country="us"):
    date_str = run_date.strftime("%Y%m%d")
    return f"{country}/{date_str}/inference_data/{region}.nc"


def get_state_output_key(run_date, region, country="us"):
    date_str = run_date.strftime("%Y%m%d")
    return f"{country}/{date_str}/stateoutput/{region}.csv"


def get_overall_output_key(run_date, country="us"):
    date_str = run_date.strftime("%Y%m%d")
    return f"{country}/{date_str}/joined.csv"


def get_processed_covidtracking_key(run_date, country="us"):
    date_str = run_date.strftime("%Y%m%d")
    return f"{country}/{date_str}/covidtracking_processed.csv"


def get_covidtracking_csv_key(run_date, country="us"):
    date_str = run_date.strftime("%Y%m%d")
    return f"{country}/{date_str}/covidtracking.csv"
