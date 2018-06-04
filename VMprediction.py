#!/usr/bin/env python

# By: Kristian Bjoerke

import numpy as np
import matplotlib.pyplot as plt
import random as rnd

# Match overview: https://en.wikipedia.org/wiki/2018_FIFA_World_Cup


teams_points={ # http://www.fifa.com/fifa-world-ranking/ranking-table/men/index.html
    'Australia': 700,
    'Iran': 727,
    'Japan': 528,
    'Saudi Arabia': 462,
    'South Korea': 520,
    'Egypt': 636,
    'Morocco': 681,
    'Nigeria': 608,
    'Senegal': 825,
    'Tunisia': 1012,
    'Costa Rica': 858,
    'Mexico': 1008,
    'Panama': 574,
    'Argentina': 1254,
    'Brazil': 1384,
    'Colombia': 989,
    'Peru': 1106,
    'Uruguay': 976,
    'Belgium': 1346,
    'Croatia': 975,
    'Denmark': 1054,
    'England': 1040,
    'France': 1166,
    'Germany': 1544,
    'Iceland': 930,
    'Poland': 1128,
    'Portugal': 1306,
    'Russia': 493,
    'Serbia': 732,
    'Spain': 1162,
    'Sweden': 889,
    'Switzerland': 1179
}


def determine_winner(matchup):
    tune_factor = 0.6 # By testing, might need re-tuning
    draw_prob = 0.74 # http://pena.lt/y/2015/12/12/frequency-of-draws-in-football/

    prob_val = float(teams_points[matchup[0]])/(teams_points[matchup[0]]+teams_points[matchup[1]])

    #print abs(teams_points[matchup[0]]-teams_points[matchup[1]])
    draw_modifier = abs(teams_points[matchup[0]]-teams_points[matchup[1]])/1e4
    #print draw_modifier
    draw_prob += draw_modifier
    #print draw_prob
    prob_val_tuned = draw_prob*(prob_val**tune_factor)

    random_number = rnd.random()
    if random_number <= prob_val_tuned:
        winner = matchup[0]
    elif random_number > draw_prob:
        winner = 'Draw'
    else:
        winner = matchup[1]

    if knockout and winner is 'Draw':
        random_number = rnd.random()
        if random_number <= prob_val:
            winner = matchup[0]+' (p)'
        else:
            winner = matchup[1]+' (p)'
    return winner

def determine_result(winner, matchup):
    # https://fivethirtyeight.com/features/in-126-years-english-football-has-seen-13475-nil-nil-draws/
    draw_results = ['0-0', '1-1', '2-2', '3-3', '4-4']
    draw_probs = np.array([7.2, 11.6, 5.2, 1.1, 0.2])
    
    win_results_0 = ['1-0', '2-1', '2-0', '3-2', '3-1', '3-0', '4-3', '4-2', '4-1', '4-0', '5-4', '5-3', '5-2', '5-1', '5-0']
    win_results_1 = ['0-1', '1-2', '0-2', '2-3', '1-3', '0-3', '3-4', '2-4', '1-4', '0-4', '4-5', '3-5', '2-5', '1-5', '0-5']
    win_probs = np.array([9.8+6.3, 8.9+5.6, 8.1+3.4, 2.8+1.8, 5.2+2.3, 4.8+1.4, 0.5+0.3, 1.4+0.6, 2.5+0.7, 2.3+0.4, 0.1+0.05, 0.2+0.1, 0.6+0.2, 1.1+0.2, 1.0+0.1])
    
    draw_probs = draw_probs/sum(draw_probs)
    draw_probs = np.cumsum(draw_probs)
    #print draw_probs
    
    win_probs = win_probs/sum(win_probs)
    win_probs = np.cumsum(win_probs)
    #print win_probs
    
    random_number = rnd.random()

    if winner == 'Draw':
        for i in range(len(draw_probs)):
            #print random_number, draw_probs[i]
            if random_number <= draw_probs[i]:
                result = draw_results[i]
                break
    else:
        for i in range(len(win_probs)):
            #print random_number, win_probs[i]
            if random_number <= win_probs[i]:
                if matchup.index(winner) == 0:
                    result = win_results_0[i]
                elif matchup.index(winner) == 1:
                    result = win_results_1[i]
                break

    return result

### Group play: ###

knockout=False

g1_m1 = ['Russia', 'Saudi Arabia']
#g1_m1 = ['Germany', 'Saudi Arabia']

winner = determine_winner(g1_m1)
print "%s -- %s: %s" % (g1_m1[0], g1_m1[1], determine_result(winner, g1_m1))
