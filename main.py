from typing import *


def create_preference_list(choices_: List[int]) -> List[int]:
    """ Orders choices, based on a 1-n rank, a 0 for unranked elements and -1 for undesired elements.
        example: [0,1,0,2,3,-1] -> [1,3,4,0,2,5]"""
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
    if rank_ > 0:
        return size_ - rank_
    else:
        return rank_


def score_function(rank1_: int, rank2_: int, size_: int) -> int:
    """ Compute a score, based on mutual ranking of both parties. """
    return rank_to_score(rank1_, size_) + rank_to_score(rank2_, size_)


def main(filepath_: str, verbose_: bool = False):
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

    # read the csv file
    sep = ";"
    with open(filepath_) as f:
        lines = [line.rstrip() for line in f]
    size = len(lines[0].split(sep)) - 1
    mentees_choices = [[0 for i in range(size)] for j in range(size)]
    mentors_choices = [[0 for i in range(size)] for j in range(size)]
    if verbose_:
        for line in lines:
            print("\t".join(line.split(sep)))
    
    # read the first row, which contains the mentors names
    mentors = lines[0].split(sep)[1:]
    mentees = []
    # read <size_> rows, each containing a mentee's choices
    for index, row in enumerate(lines[1:size + 1]):
        mentees.append(row.split(sep)[0])
        this_mentee_choices = [int(c) for c in row.split(sep)[1:]]
        # do some consistency checks
        filtered_choices = [c for c in this_mentee_choices if c not in [0, -1]]
        if len(set(filtered_choices)) < len(filtered_choices):
            print(f"Mentee {mentees[-1]} choices are inconsistent:\n\t{filtered_choices}.")
            exit(1)
        mentees_choices[index] = this_mentee_choices
    # skip one line, which is considered to be empty
    # read <size_> rows, each containing the mentors' choices for one mentee
    for mentee_index, row in enumerate(lines[size + 2:]):
        for mentor_index, choice in enumerate([int(c) for c in row.split(sep)[1:]]):
            mentors_choices[mentor_index][mentee_index] = choice
    
    # do some consistency checks
    for mentor_index, mentor_choices in enumerate(mentors_choices):
        filtered_choices = [c for c in mentor_choices if c not in [0, -1]]
        if len(set(filtered_choices)) < len(filtered_choices):
            print(f"Mentor {mentors[mentor_index]} choices are inconsistent:\n\t{filtered_choices}.")
            exit(1)

    # compute an ordered list of people, depending on the choices that were made
    mentees_preferences = [create_preference_list(choices) for choices in mentees_choices]
    mentors_preferences = [create_preference_list(choices) for choices in mentors_choices]

    # use the Gale-Shapley algorithm to do the pairing
    # mentees are considered as "men" in this algorithm - they get better results than mentors in their matches
    mentors_pairs = gale_shapley(mentees_preferences, mentors_preferences, size, verbose_)

    # display the results and compute the overall score
    total_score = 0
    for mentor_index, mentee_index in enumerate(mentors_pairs):
        mentee_ranks_mentor = mentees_choices[mentee_index][mentor_index]
        mentor_ranks_mentee = mentors_choices[mentor_index][mentee_index]
        pair_score = score_function(mentee_ranks_mentor, mentor_ranks_mentee, size)
        print(f"{mentees[mentee_index]}({mentee_ranks_mentor})\t- "\
              f"{mentors[mentor_index]}({mentor_ranks_mentee})\t: {pair_score}")
        total_score += pair_score
    print(f"Total matching score\t: {total_score}")


if __name__ == '__main__':
    main("bfb_matrix.csv", False)
