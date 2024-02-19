import random
from collections import defaultdict
from typing import *
import math
import time
# import matplotlib.pyplot as plt


def create_preference_list(choices_: List[int]) -> List[int]:
    """ Orders choices, based on a 1-n rank, a 0 for unranked elements and -1 for undesired elements.
        example: [0,1,0,2,3,-1] -> [1,3,4,0,2,5]
        WARNING: this is actually an issue as, it will artificially rank people who are all at 0,
        in a consistent way (top of the list is better ranked). """
    preferences = []
    default_choices = [i for i, v in enumerate(choices_) if v == 0]
    unwanted_choices = [i for i, v in enumerate(choices_) if v == -1]
    next_rank = 1
    while max(choices_) >= next_rank:
        for i, v in enumerate(choices_):
            if v == next_rank:
                preferences.append(i)
                next_rank += 1
                break
    preferences += default_choices + unwanted_choices
    return preferences


def gale_shapley(men_preferences_: List[List[int]],
                 women_preferences_: List[List[int]],
                 size_: int,
                 verbose_: bool) -> List[int]:
    """ Gale-Shapley algorithm for the stable marriages' problem.
        men_preferences is a matrix, each row i contains the preference score of each woman.
        women_preferences is a matrix, each row i contains the preference score of each man. """
    women_partners = [-1 for i in range(size_)]
    man_is_free = [True for i in range(size_)]
    free_count = size_

    while free_count > 0:
        # pick the first free man (we could pick any)
        m = 0
        while m < size_:
            if man_is_free[m]:
                break
            m += 1
        # one by one, go to all women according to m's preferences
        i = 0
        while i < size_ and man_is_free[m]:
            w = men_preferences_[m][i]
            # if the woman of preference is free, w and m become partners
            if women_partners[w] == -1:
                women_partners[w] = m
                man_is_free[m] = False
                free_count -= 1
            else:
                # if w is not free, find current engagement of w
                m1 = women_partners[w]
                # if w prefers m over her current engagement m1, then break the engagement between w and m1 and
                # engage m with w.
                if women_preferences_[w].index(m) < women_preferences_[w].index(m1):
                    women_partners[w] = m
                    man_is_free[m] = False
                    man_is_free[m1] = True
            i += 1
        if verbose_:
            print(f"Pairing progress: {100*(size_-free_count)/size_:.0f}%")
    return women_partners


def rank_to_score(rank_: int, size_: int) -> int:
    """ Transform a ranking into a score. """
    return size_ - rank_ if rank_ > 0 else rank_


def score_function(rank1_: int, rank2_: int, size_: int) -> int:
    """ Compute a score, based on mutual ranking of both parties. Symmetrical. """
    return rank_to_score(rank1_, size_) + rank_to_score(rank2_, size_)


def choice_to_exp_score(choice_: int, size_: int) -> float:
    """ Compute a score, based on a choice (-1, 0, 1, 2, 3, 4, 5). """
    if choice_ < 0:  # allows to use -2, -3, ... for high severity
        return choice_*choice_to_exp_score(1, size_)  # use the maximum score as a reference
    elif choice_ == 0:
        return 0
    else:
        return math.exp((size_-choice_)/size_)


def disp_pairing_result(pairs_: List[Tuple[str, str]], mentees_dict_choices_: Dict, mentors_dict_choices_: Dict,
                        size_: int, verbose_: bool) -> int:
    """ Compute the score of each pair, displays it and returns the total pairing score. """
    total_score = 0
    for mentee, mentor in pairs_:
        mentee_choice, mentor_choice = mentees_dict_choices_[mentee][mentor], mentors_dict_choices_[mentor][mentee]
        pair_score = score_function(mentee_choice, mentor_choice, size_)
        if verbose_:
            print(f"{mentee}({mentee_choice}) - {mentor}({mentor_choice})\t: {pair_score}")
        total_score += pair_score
    if verbose_:
        print(f"Total matching score\t: {total_score}")
    return total_score


def rate_input_pairing(pairs_file_: str, mentees_dict_choices_: Dict, mentors_dict_choices_: Dict,
                       pairing_score_: int, size_: int) -> None:
    """ Reads and checks a submitted pairing file. If valid, the pairs are scored and the total score is
        compared to the computed score.
        The format of the input file is "mentee name - mentor name" on each line.
        Mind the " - " in the middle, with one blankspace on each side. """
    try:
        with open(pairs_file_) as f:
            input_pairs = [tuple(line.rstrip().split(" - ")) for line in f]
    except FileNotFoundError:
        print(f"WARNING: cannot read file: \"{pairs_file_}\"")
        return
    # trim the file to the problem size
    input_pairs = input_pairs[:size_]
    # perform some checks on the integrity of the pairs
    input_mentees, input_mentors = [pair[0] for pair in input_pairs], [pair[1] for pair in input_pairs]
    for mentee in input_mentees:  # each submitted mentee must be known
        if mentee not in mentees_dict_choices_:
            print(f"Unknown mentee: \"{mentee}\"")
            exit(1)
    for mentor in input_mentors:  # each submitted mentor must be known
        if mentor not in mentors_dict_choices_:
            print(f"Unknown mentor: \"{mentor}\"")
            exit(1)
    if len(set(input_mentees)) != len(set(input_mentors)) or len(set(input_mentees)) != len(mentees_dict_choices_):
        print(f"You have submitted a different number of unique mentees and mentors")
        exit(1)
    print("\nSubmitted pairing:")
    submitted_pairing_score = disp_pairing_result(input_pairs, mentees_dict_choices_, mentors_dict_choices_, size_, True)
    if submitted_pairing_score > pairing_score_:
        print("You have submitted a better pairing than I could compute.")


def partial_gale_shapley(mentees_dict_choices: Dict, mentors_dict_choices: Dict,
                         size: int, verbose_: bool) -> Tuple[int, List[List[int]]]:
    """ Apply the Gale-Shapley algorithm on partially ranked data. Returns the score of the result and the pairing. """
    # compute an ordered list of people, depending on the choices that were made
    mentees_choices = [[mentees_dict_choices[mentee][mentor] for mentor in mentors_dict_choices] for mentee in mentees_dict_choices]
    mentors_choices = [[mentors_dict_choices[mentor][mentee] for mentee in mentees_dict_choices] for mentor in mentors_dict_choices]
    mentees_preferences = [create_preference_list(choices) for choices in mentees_choices]
    mentors_preferences = [create_preference_list(choices) for choices in mentors_choices]
    # use the Gale-Shapley algorithm to do the pairing
    # mentees are considered as "men" in this algorithm - they get better results than mentors in their matches
    mentors_pairs = gale_shapley(mentees_preferences, mentors_preferences, size, verbose_)
    # transform the pairs to a "mentee - mentor" format
    mentees_list = list(mentees_dict_choices.keys())
    mentors_list = list(mentors_dict_choices.keys())
    pairs = [(mentees_list[mentee_index], mentors_list[mentor_index]) for mentor_index, mentee_index in enumerate(mentors_pairs)]
    # compute the overall score
    pairing_score = disp_pairing_result(pairs, mentees_dict_choices, mentors_dict_choices, size, verbose_)
    return pairing_score, pairs


def shuffle_dict(dict_: Dict) -> Dict:
    """ Shuffle dict entries (keys). """
    list_items = list(dict_.items())
    random.shuffle(list_items)
    return dict(list_items)


def read_bfb_csv(filename_: str, sep_: str) -> Tuple[int, List[str]]:
    """ Read a .csv file formatted for BfB and returns the lines and the problem size. """
    try:
        with open(filename_) as f:
            lines = [line.rstrip() for line in f]
    except FileNotFoundError:
        print(f"ERROR: cannot read file: \"{filename_}\", aborting...")
        exit(1)
    size = len(lines[0].split(sep_)) - 1
    return size, lines


def main(csv_file_: str, csv_sep_: str, iterations_: int,
         export_pairs_: bool, input_pairs_file_: str, display_individual_scores_: bool):
    """ Solve the stable-matching problem https://www.cs.cmu.edu/~arielpro/15896s16/slides/896s16-16.pdf
        The use-case is pairing mentees and mentors, based on:
            - partial ordering on both sides (only from 1 to 5),
            - unordered people receive the 0 value,
            - undesired matches receive the -1 value.
        The input data is in the form of a .csv file containing two matrices:
            - the first contains, by row, the choices of a mentee,
            - the second contains, by column, the choices of a mentor.
        The output is printed out in the console and shows:
            - the mentee-mentor pairing, along with each of their mutual ranking,
            - the pair score,
            - the total pairing score.
        The Gale-Shapley algorithm is used, with the following results and limitations:
            - stable-matching is supposed to be reached,
            - however, under conditions of partial ranking, this property cannot be guaranteed,
            - this algorithm does not give sufficient leverage to prevent undesired matches (-1)
              from being made, check the output to see if this has occured.
    """
    print("\n\t\t==== LET'S GO ====")

    # read the csv file
    size, lines = read_bfb_csv(csv_file_, csv_sep_)
    if size != 0:
        print(f"There are {size} mentees and mentors, totalling {2*size} people.\n")
    else:
        print(f"WARNING: the problem size is 0: there probably was an issue when reading the file.\n" +
              f"Check the definition of the CSV separator, it is currently set to '{csv_sep_}'.\n" +
              "Trying to use another separator...")
        # try using ; if , was used, and conversely
        csv_sep_ = {",", ";"}.difference({csv_sep_}).pop()
        size, lines = read_bfb_csv(csv_file_, csv_sep_)
        if size == 0:
            print(f"ERROR: still could not read data from file \"{csv_file_}\"")
            exit(1)
        print(f"File succesfully loaded! There are {size} mentees and mentors, totalling {2*size} people.\n")

    
    # read the first row, which contains the mentors names
    mentors = lines[0].split(csv_sep_)[1:]
    # initialize a dictionnary that will contain the mentor's ordered choices
    mentors_dict_choices = {mentor_name: {} for mentor_name in lines[0].split(csv_sep_)[1:]}
    mentees = []
    mentees_dict_choices = defaultdict(dict)
    # read <size_> rows, each containing a mentee's choices
    for index, row in enumerate(lines[1:size + 1]):
        mentee_name = row.split(csv_sep_)[0]
        mentees.append(mentee_name)
        this_mentee_choices = [int(c) for c in row.split(csv_sep_)[1:]]
        # do some consistency checks
        filtered_choices = [c for c in this_mentee_choices if c not in [0, -1]]
        if len(set(filtered_choices)) < len(filtered_choices):
            print(f"Mentee {mentee_name} choices are inconsistent:\n\t{filtered_choices}.")
            exit(1)
        for mentor_idx, mentor_choice in enumerate(this_mentee_choices):
            mentees_dict_choices[mentee_name][mentors[mentor_idx]] = mentor_choice
    # skip one line, which is considered to be empty
    # read <size_> rows, each containing the mentors' choices for one mentee
    for mentee_index, row in enumerate(lines[size + 2:]):
        for mentor_index, choice in enumerate([int(c) for c in row.split(csv_sep_)[1:]]):
            mentors_dict_choices[mentors[mentor_index]][mentees[mentee_index]] = choice
    
    # do some consistency checks
    for mentor_choices in [[c for c in mentors_dict_choices[mentor].values()] for mentor in mentors_dict_choices]:
        filtered_choices = [c for c in mentor_choices if c not in [0, -1]]
        if len(set(filtered_choices)) < len(filtered_choices):
            print(f"Mentor {mentors[mentor_index]} choices are inconsistent:\n\t{filtered_choices}.")
            exit(1)

    # compute a "popularity" score for mentors and mentees
    if display_individual_scores_:
        print("\nMentors score")
        mentees_choices = [[mentees_dict_choices[mentee][mentor] for mentor in mentors_dict_choices] for mentee in
                           mentees_dict_choices]
        mentors_choices = [[mentors_dict_choices[mentor][mentee] for mentee in mentees_dict_choices] for mentor in
                           mentors_dict_choices]
        mentors_scores = {name: 0 for name in mentors}
        for mentee_choices in mentees_choices:
            for mentor_index, choice in enumerate(mentee_choices):
                mentors_scores[mentors[mentor_index]] += choice_to_exp_score(choice, size)
        for name in sorted(mentors_scores, key=mentors_scores.get, reverse=True):
            print(f"{name}:\t {mentors_scores[name]:.2f}")
        print("\nMentees score")
        mentees_scores = {name: 0 for name in mentees}
        for mentor_choices in mentors_choices:
            for mentee_index, choice in enumerate(mentor_choices):
                mentees_scores[mentees[mentee_index]] += choice_to_exp_score(choice, size)
        for name in sorted(mentees_scores, key=mentees_scores.get, reverse=True):
            print(f"{name}:\t {mentees_scores[name]:.2f}")

    # look for a locally optimal pairing
    best_pairing = (0, [])
    pairing_scores = []
    for i in range(iterations_):
        # get the corresponding pairing
        pairing_score, pairs = partial_gale_shapley(mentees_dict_choices, mentors_dict_choices, size, False)
        pairing_scores.append(pairing_score)
        # if better than current, save
        if pairing_score > best_pairing[0]:
            best_pairing = (pairing_score, pairs)
        # do a permutation of the input matrices (mentees, mentors)
        mentees_dict_choices = shuffle_dict(mentees_dict_choices)
        mentors_dict_choices = shuffle_dict(mentors_dict_choices)
    disp_pairing_result(best_pairing[1], mentees_dict_choices, mentors_dict_choices, size, True)

    # export the computed pairs to a file, mainly for modification and re-submission
    if export_pairs_:
        filename = "output_pairs.txt"
        with open(filename, "w") as f:
            for mentee, mentor in best_pairing[1]:
                f.write(f"{mentee} - {mentor}\n")
            f.write(f"Total score: {best_pairing[0]}")
        print(f"The file {filename} containing the computed pairing has been written.")

    if input_pairs_file_:
        rate_input_pairing(input_pairs_file_, mentees_dict_choices, mentors_dict_choices,
                           best_pairing[0], size)

    # plt.hist(pairing_scores, bins=20)
    # plt.show()
    print("\n\t\t==== JOB DONE ====")     


if __name__ == '__main__':
    start_time = time.time_ns()
    """ Use the following call to parametrize the pariring run.
            * csv_file (str): path to the .csv file containing the input data;
            * csv_sep (str): separator used in the csv file;
            * iterations (int): number of iterations to do, in the order of a few thousands,
                                more iterations gives a higher chance of optimal pairing;
            * export_pairs (bool): true if you want a text file to be created with the pairing;
            * input_pairs_file (str): provide a former, or manually corrected pairing so as to be scored
                                      and compared to the pairing produced by the tool;
            * display_individual_scores (bool): true if you want statistics on who was best ranked on both sides."""
    main(csv_file_ = "bfb_matrix.csv", csv_sep_ = ",", iterations_ = 10000,
         export_pairs_ = False, input_pairs_file_ = "input_pairs.txt",
         display_individual_scores_ = False)
    
    print(f"Elapsed time: {(time.time_ns()-start_time)*1e-6:.2f} ms")
