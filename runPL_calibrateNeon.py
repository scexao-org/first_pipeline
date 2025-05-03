
import numpy as np
from itertools import combinations
import heapq
from optparse import OptionParser
import sys

# Add options
usage = """
    usage:  %prog [options] files.fits

    Goal: Calibrate a list of detected peaks on a pixel range on a wavelenght range using a set list of wavelength.

    Example:
    runPL_calibrateNeon.py --pixel_min=100 --pixel_max=1600 --pixel_wide=2 --output_channels=38 *.fits

    Options:
    --all_peaks:  list of all detected peaks
    --peaks_weight: list of weight associated with each peak 
    --wavelength_list: list of wavelenght to fit
    --skip_n_wave: max number of wavelenghts to drop in the reference list (default :0)
    --how_many_more_peaks : how many more peaks than the number of peaks in the wavelenght_list to consider in our search for them (default:0)
"""



def get_combinations(input_list, combination_length):
    """
    A list containing all possible combinations (as lists) of the specified length. Each combination is represented as a sublist.
    Eg : 
    input_list = [1, 2, 3, 4]
    combination_length = 2
    Will return : [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    """

    res = [list(combo) for combo in combinations(input_list, combination_length)] 
    return res


def heaviest_n_peaks(list_of_peaks, list_of_weight, minimum_number_of_peaks_to_return):
    '''
    Returns the n heaviest peaks in the list of peak.
    eg : if we have waves [10,20,30,40, 50], [1,2,3,4,5] as a list of weight, and want the 3 biggest, we'll return [30,40,50].
    if we have weight [1,2,2,3,4] we'll return [20,30,40,50]

    This function is very time expensive so start at n=len(list_of_wave) and move on from there.
    '''
    to_return =[]
    #search for the minimum weight to get all values
    for k in range(1,len(list_of_peaks)):
        max_n=heapq.nlargest(k, list_of_weight)[-1] #return the k-th biggest value of the list
        if how_many_bigger_or_equals_to(list_of_weight,max_n) >=minimum_number_of_peaks_to_return:
            break

    for k in range(len(list_of_weight)-1):
        if list_of_weight[k]>=max_n:
            to_return.append(list_of_peaks[k])
    return to_return


def how_many_bigger_or_equals_to(list_to_check, value):
    # Counts how many elements in the list are greater than or equal to the given value.
    count = 0
    for item in list_to_check:
        if item >= value:
            count += 1
    return count


def how_close_is_this_list(list_candidate, wavelength_list):
    '''
    For each list of possible fit (list_candidate) to the wavelenght list,
    we're going to check how close they are to the 1D fit we expect, and grade them 
    based on the how close to the y = ax+b each points are.
    '''

    # Fit a first-degree polynomial (linear fit)
    linear_poly = np.polyfit(list_candidate, wavelength_list, 1)
    linear_fit = np.poly1d(linear_poly)
    
    # Calculate residuals
    residuals = [abs(wavelength_list[i] - linear_fit(list_candidate[i])) for i in range(len(wavelength_list))]
    
    # Calculate a residual-based error metric (RMSE)
    rmse = np.sqrt(np.mean(np.square(residuals)))
    
    return rmse  # The lower the RMSE, the better the fit


def how_close_are_all_lists(all_lists, wavelength_list):
    '''
    We run all the possible list in the grading function.
    The output is a list containing all the grades.
    '''
    res = []
    for candidate in all_lists:
        res.append(how_close_is_this_list(candidate, wavelength_list))
    return res 

def closest_list(all_combinations, wavelength_list):
    '''
    This function get the index of the list with the best grade, and returns the corresponding list.
    '''
    result_list = how_close_are_all_lists(all_combinations, wavelength_list)
    index = result_list.index(min(result_list))
    return all_combinations[index]

def its_a_match(all_peaks, peaks_weight, wavelength_list, how_many_more_peaks_to_consider):
    '''
    For all peaks, consider each possible fit with a given number of detected peaks and returns
    the best graded list.
    '''
    number_of_waves = len(wavelength_list)
    first_candidate = heaviest_n_peaks(all_peaks, peaks_weight, number_of_waves+how_many_more_peaks_to_consider)
    all_possible_outputs = get_combinations(first_candidate, number_of_waves)
    result = closest_list(all_possible_outputs, wavelength_list)
    return result

def let_go_of_some_peaks(wavelength_list, how_many_peaks_to_ignore):
    '''
    For n peaks to ignore, returns a list containing all new possible combinations of wavelenght list.
    '''
    if how_many_peaks_to_ignore==0:
        return [wavelength_list]
    result = []
    # Generate all combinations of indices to remove
    for indices in combinations(range(len(wavelength_list)), how_many_peaks_to_ignore):
        # Create a new list excluding the elements at the selected indices
        new_lst = [wavelength_list[i] for i in range(len(wavelength_list)) if i not in indices]
        result.append(new_lst)
    return result

def run_trials_for_all_combination_of_waves(all_peaks,peaks_weight,wavelength_list, skip_n_waves=0,how_many_more_peaks_to_consider=0):
    '''
    Generate all possible versions of the wavelength list and run the fit for all of them
    '''
    all_results = []
    all_rates = []
    wavelength_all = let_go_of_some_peaks(wavelength_list, skip_n_waves)
    for wavelist in wavelength_all:
        result_now = its_a_match(all_peaks,peaks_weight, wavelist, how_many_more_peaks_to_consider)
        all_rates.append(how_close_is_this_list(result_now, wavelist))
        all_results.append(its_a_match(all_peaks,peaks_weight, wavelist, how_many_more_peaks_to_consider))
    best_fit_index = all_rates.index(min(all_rates))
    return all_results[best_fit_index], wavelength_all[best_fit_index]


if __name__ == "__main__":
    parser = OptionParser(usage)

    # Default values
    all_peaks =[]
    peaks_weight =[]
    wavelength_list =[]
    skip_n_wave =0
    how_many_more_peaks =0

    # Add options for these values
    parser.add_option("--all_peaks", type="list", default=all_peaks,
                      help="List of detected peaks")
    parser.add_option("--peaks_weight", type="int", default=peaks_weight,
                    help="List of weight associated with each peak ")
    parser.add_option("--wavelenght_list", type="int", default=wavelength_list,
                    help="List of wavelength to fit")
    parser.add_option("--skip_n_wave", type="int", default=skip_n_wave,
                    help="Maximum number of wavelength to skip if unfit (default : 0)")
    parser.add_option("--how_many_more_peaks", type="int", default=how_many_more_peaks,
                    help="How many additional detected peak to consider in the search (default : 0)")
    
    # Parse the options
    (options, args) = parser.parse_args()

    # Pass the parsed options to the function*
    if all_peaks==[]: 
        raise ValueError("No detected peaks")
        sys.exit(1) 
    elif peaks_weight==[]:
        raise ValueError("No peak weight")
        sys.exit(1) 
    elif len(all_peaks)!=len(peaks_weight):
        raise ValueError("Peaks don't match weight")
        sys.exit(1) 
    elif wavelength_list==[]:
        raise ValueError("No reference wavelenght list")
        sys.exit(1) 
    
    run_trials_for_all_combination_of_waves(all_peaks,peaks_weight,wavelength_list, skip_n_wave,how_many_more_peaks)
