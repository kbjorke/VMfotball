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
    #tune_factor = 0.4 # By testing, might need re-tuning (using lower value for knockout phase)
    tune_factor = 2.0 # By testing, might need re-tuning (using lower value for knockout phase)
    draw_prob = 0.74 # http://pena.lt/y/2015/12/12/frequency-of-draws-in-football/

    prob_val = float(teams_points[matchup[0]])/(teams_points[matchup[0]]+teams_points[matchup[1]])

    draw_modifier = abs(teams_points[matchup[0]]-teams_points[matchup[1]])/1e4
    draw_prob += draw_modifier
    #prob_val_tuned = draw_prob*(prob_val**tune_factor)
    prob_val_tuned = draw_prob*(((2*prob_val)**tune_factor)/(2**tune_factor))
    
    print teams_points[matchup[0]], teams_points[matchup[1]]
    print prob_val
    print (((2*prob_val)**tune_factor)/(2**tune_factor))
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
    
    win_probs = win_probs/sum(win_probs)
    win_probs = np.cumsum(win_probs)
    
    random_number = rnd.random()

    if winner[-4:] == ' (p)':
        for i in range(len(draw_probs)):
            if random_number <= draw_probs[i]:
                result = draw_results[i]
                break
    else:
        for i in range(len(win_probs)):
            if random_number <= win_probs[i]:
                if matchup.index(winner) == 0:
                    result = win_results_0[i]
                elif matchup.index(winner) == 1:
                    result = win_results_1[i]
                break

    return result

def knockout_round(matches):
    round_results = dict()
    for match in matches:
        round_results[match] = ['team1', 'team2', 'winner', 'results']
    
    for match in matches:
        teams = matches[match]
        winner = determine_winner(teams)
        result = determine_result(winner, teams)
        round_results[match] = [teams[0], teams[1], winner, result]

    return round_results


def print_results(label, results):
    print "\n --- %s ---" % label
    print "\n%-8s | %s | %s | %s" %(" Matches", "Teams".center(26), "Winner".center(15), "Results")
    print "%s" %(68*'-') 
    for match in results:
        print "%-8s | %11s -- %-11s | %-15s | %s" %(match, results[match][0], results[match][1], results[match][2], results[match][3])

def getQF(round16_results):
    QF_matches = {
            'Match 57': ['Match 49', 'Match 50'],
            'Match 58': ['Match 53', 'Match 54'],
            'Match 60': ['Match 55', 'Match 56'],
            'Match 59': ['Match 51', 'Match 52']
            }
    
    for match in QF_matches:
        for i in range(2):
            QF_matches[match][i] = round16_results[QF_matches[match][i]][2].replace(' (p)', '')

    return QF_matches

def getSF(QF_results):
    SF_matches = {
            'Match 61': ['Match 57', 'Match 58'],
            'Match 62': ['Match 59', 'Match 60']
            }
    
    for match in SF_matches:
        for i in range(2):
            SF_matches[match][i] = QF_results[SF_matches[match][i]][2].replace(' (p)', '')

    return SF_matches

def getTPP(SF_results):
    TPP_match = {
            'Match 63': ['Match 61', 'Match 62'],
            }
    
    for match in TPP_match:
        for i in range(2):
            winner = SF_results[TPP_match[match][i]][2].replace(' (p)', '')
            for team in SF_results[TPP_match[match][i]][0:2]:
                if team != winner:
                    TPP_match[match][i] = team

    return TPP_match

def getFinal(SF_results):
    Final_match = {
            'Match 64': ['Match 61', 'Match 62'],
            }
    
    for match in Final_match:
        for i in range(2):
            Final_match[match][i] = SF_results[Final_match[match][i]][2].replace(' (p)', '')

    return Final_match




def group_runs(label, group_teams, group_matches, n_runs):

    print " --- Computing %2s runs for Group %s --- " % (n_runs, label)
    
    orig_stdout = sys.stdout
    outfile = open('./group_stage/group%s.txt' % label, 'w')
    sys.stdout = outfile
        
    print "######### Group stage: Group %s #########\n\n" % label
    
    for i in range(1,n_runs+1):
        print "###### RUN %3s ######" % i
    
        results, score_table = group_play(group_teams, group_matches)
        print_results(results)
        print_score_table(score_table)
        print "\n"
    
    sys.stdout = orig_stdout
    outfile.close()

### Knockout stage: ###

knockout=True

# Round of 16:

round16_matches = {
        'Match 50': ['Denmark', 'Iceland'],
        'Match 49': ['Uruguay', 'Portugal'],
        'Match 51': ['Spain', 'Egypt'],
        'Match 52': ['Argentina', 'Peru'],
        'Match 53': ['Switzerland', 'Sweden'],
        'Match 54': ['England', 'Senegal'],
        'Match 55': ['Germany', 'Brazil'],
        'Match 56': ['Colombia', 'Belgium']
        }

round16_results = knockout_round(round16_matches)
print_results('Round of 16', round16_results)

QF_matches = getQF(round16_results)

QF_results = knockout_round(QF_matches)
print_results('Quarter-finals', QF_results)

SF_matches = getSF(QF_results)

SF_results = knockout_round(SF_matches)
print_results('Semi-finals', SF_results)

TPP_match = getTPP(SF_results)

TPP_result = knockout_round(TPP_match)
print_results('Third place play-off', TPP_result)

Final_match = getFinal(SF_results)

Final_result = knockout_round(Final_match)
print_results('Final', Final_result)

# Runs of knockout stage:
n_runs = 10
