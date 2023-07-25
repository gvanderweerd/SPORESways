import os
from utils.data_io import *

if __name__ == '__main__':

    # Define paths
    path_to_categorised_spores = os.path.join(os.getcwd(), "../new_repository_structure/data", "raw", "eurospores-results-2030", "categorised")
    path_to_aggregated_spores = os.path.join(os.getcwd(), "../new_repository_structure/data", "raw", "eurospores-results-2030", "aggregated")
    print(path_to_categorised_spores)
    print(path_to_aggregated_spores)

    # Aggregate 2030 spores results such that all spores are present in one data file for each type of data
    # aggregate_categorised_spores(
    #     path_to_spores=path_to_categorised_spores,
    #     path_to_result=path_to_aggregated_spores
    # )
    print(path_to_categorised_spores)
    print(path_to_aggregated_spores)


    # Aggregate 2030 spores results and 2050 spores results to a continental level
    