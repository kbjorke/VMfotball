#!/usr/bin/env python

# By: Kristian Bjoerke

import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import sys

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
    draw_results = ['0 - 0', '1 - 1', '2 - 2', '3 - 3', '4 - 4']
    draw_probs = np.array([7.2, 11.6, 5.2, 1.1, 0.2])
    
    win_results_0 = ['1 - 0', '2 - 1', '2 - 0', '3 - 2', '3 - 1', '3 - 0', '4 - 3', '4 - 2', '4 - 1', '4 - 0', '5 - 4', '5 - 3', '5 - 2', '5 - 1', '5 - 0']
    win_results_1 = ['0 - 1', '1 - 2', '0 - 2', '2 - 3', '1 - 3', '0 - 3', '3 - 4', '2 - 4', '1 - 4', '0 - 4', '4 - 5', '3 - 5', '2 - 5', '1 - 5', '0 - 5']
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

def group_play(matches, group_teams):
    results = dict()
    for match in matches:
        results[match] = ['team1', 'team2', 'results']

    score_table = dict()
    for team in group_teams:
        score_table[team] = [0,0,0,0,0,0,0]

    for match in matches:
        teams = matches[match]
        winner = determine_winner(teams)
        result = determine_result(winner, teams)
        results[match] = [teams[0], teams[1], result]
        
        for team in teams:
            score_table[team][3] += int(result.split('-')[teams.index(team)])
            score_table[team][4] += int(result.split('-')[teams.index(team)-1])
            score_table[team][5] += (int(result.split('-')[teams.index(team)]) - int(result.split('-')[teams.index(team)-1]))
            if winner is team:
                score_table[team][0] += 1
                score_table[team][6] += 3
            elif winner is "Draw":
                score_table[team][1] += 1
                score_table[team][6] += 1
            else:
                score_table[team][2] += 1
                score_table[team][6] += 0

    return (results, score_table)

def print_results(results):
    print "\n%-8s | %s | %s" %(" Matches", "Teams".center(34), "Results")
    print "%s" %(60*'-') 
    for match in results:
        print "%-8s | %15s -- %-15s | %s" %(match, results[match][0], results[match][1], results[match][2])

def print_score_table(score_table):
    score_table_sorted = sorted(score_table.items(), key=lambda e: e[1][6], reverse=True)
    #print "Team | W : D : L | GF : GA : GD | Pts"
    print "\n%15s | %3s : %3s : %3s | %3s : %3s : %3s | %3s" %( 'Teams', 'W', 'D', 'L', 'GF', 'GA', ' GD', 'Pts')
    print "%s" %(60*'-') 
    for i in range(len(score_table_sorted)):
        print "%15s | %3d : %3d : %3d | %3d : %3d : %3d | %3d" % (
                score_table_sorted[i][0],
                score_table_sorted[i][1][0],
                score_table_sorted[i][1][1],
                score_table_sorted[i][1][2],
                score_table_sorted[i][1][3],
                score_table_sorted[i][1][4],
                score_table_sorted[i][1][5],
                score_table_sorted[i][1][6])

### Group play: ###

knockout=False

# Group A:

groupA_teams = ['Russia', 'Saudi Arabia', 'Egypt', 'Uruguay']

groupA_matches = {
        'Match 1': ['Russia', 'Saudi Arabia'],
        'Match 2': ['Egypt', 'Uruguay'],
        'Match 17': ['Russia', 'Egypt'],
        'Match 18': ['Uruguay', 'Saudi Arabia'],
        'Match 33': ['Uruguay', 'Russia'],
        'Match 34': ['Saudi Arabia', 'Egypt']
        }


#orig_stdout = sys.stdout
#f = open('./group_stage/groupA.txt', 'w')
#sys.stdout = f
#    
#print "######### Group stage: Group A #########"
#
#for i in range(1,11):
#    print "###### RUN %3s ######" % i
#
#    groupA_results, groupA_score_table = group_play(groupA_matches, groupA_teams)
#    print_results(groupA_results)
#    print_score_table(groupA_score_table)
#    print "\n"
#
#sys.stdout = orig_stdout
#f.close()
