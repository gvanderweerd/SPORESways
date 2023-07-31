import os
from utils.data_io import *


if __name__ == '__main__':
    # Set parameters
    years = ["2030", "2050"]
    save_processed_spores = True

    # Set this to a list of years (like ["2030"]) if the spores results are provided in different folders for different categories of spores results
    # Set this to None if spores results are provided in one folder containing all spores
    categorised_spores_years = None

    # Aggregate spores results that are provided in different folders for different categories of spores results instead of having all spores in each file
    if categorised_spores_years is not None:
        for year in categorised_spores_years:
            aggregate_categorised_spores(
                path_to_spores=os.path.join(os.getcwd(), "..", "data", "raw", f"euro-spores-results-{year}", "categorised"),
                path_to_result=os.path.join(os.getcwd(), "..", "data", "raw", f"euro-spores-results-{year}", "aggregated", "data")
            )
        #FIXME: we need to manually change the title name of the grid capacity from "grid_transfer_capacity.csv" to "grid_transfer_capacity.csv"


    # Define path to where the raw euro-spores-results can be found
    paths_to_raw_spores = {
        "2030": os.path.join(os.getcwd(), "..", "data", "raw", "euro-spores-results-2030", "aggregated"),
        "2050": os.path.join(os.getcwd(), "..", "data", "raw", "euro-spores-results-2050", "aggregated-slack-10")
    }
    match_column_name_with_index_file(paths_to_raw_spores.get("2030"))

    # Define path to where the script will save the processed spores results
    path_to_processed_spores = os.path.join(os.getcwd(), "..", "data", "processed")

    # Define which files we want to read
    files = [
        "nameplate_capacity",
        "grid_transfer_capacity",
        "storage_capacity"
    ]

    # Read spores results for the years that were defined
    data = {}
    for year in years:
        print(year)
        data[year] = read_spores_data(path_to_spores=paths_to_raw_spores[year], file_names=files)

    # Comparison of the difference and overlap of technologies that exist in 2030 and 2050
    # compare_technologies_2030_vs_2050(data, "nameplate_capacity")
    # compare_technologies_2030_vs_2050(data, "storage_capacity")

    # Process spores results:
    #   - to a national level (and include total values for the whole continent under region "Europe")
    #   - combine the results for all years in that are defined in one file
    power = get_power_capacity(
        spores_data=data, result_path=path_to_processed_spores, save_to_csv=save_processed_spores
    )
    heat = get_heat_capacity(
        spores_data=data, result_path=path_to_processed_spores, save_to_csv=save_processed_spores
    )
    storage = get_storage_capacity(
        spores_data=data, result_path=path_to_processed_spores, save_to_csv=save_processed_spores
    )

    # FIXME: how do we deal with the difference in spatial granularity? 2030 has national granualarity so it only contains international grid capacity, the grid capacity within each country is lost
    # grid = get_grid_capacity(
    #     spores_data=data, result_path=path_to_processed_spores, save_to_csv=save_processed_spores
    # )