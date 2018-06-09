#!/usr/bin/env python

# By: Kristian Bjoerke

import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import sys

# Match overview: https://en.wikipedia.org/wiki/2018_FIFA_World_Cup

teams_points={ # Updated 8. june 2018 https://www.eloratings.net/
    'Australia': 1733,
    'Iran': 1787,
    'Japan': 1670,
    'Saudi Arabia': 1592,
    'South Korea': 1729,
    'Egypt': 1646,
    'Morocco': 1723,
    'Nigeria': 1681,
    'Senegal': 1740,
    'Tunisia': 1659,
    'Costa Rica': 1750,
    'Mexico': 1861,
    'Panama': 1659,
    'Argentina': 1986,
    'Brazil': 2136,
    'Colombia': 1928,
    'Peru': 1916,
    'Uruguay': 1894,
    'Belgium': 1933,
    'Croatia': 1848,
    'Denmark': 1845,
    'England': 1948,
    'France': 1995,
    'Germany': 2076,
    'Iceland': 1764,
    'Poland': 1831,
    'Portugal': 1970,
    'Russia': 1678,
    'Serbia': 1763,
    'Spain': 2042,
    'Sweden': 1794,
    'Switzerland': 1886
}

#teams_points={ # Updated 7. june 2018 http://www.fifa.com/fifa-world-ranking/ranking-table/men/index.html
#    'Australia': 718,
#    'Iran': 708,
#    'Japan': 521,
#    'Saudi Arabia': 465,
#    'South Korea': 544,
#    'Egypt': 649,
#    'Morocco': 686,
#    'Nigeria': 618,
#    'Senegal': 838,
#    'Tunisia': 910,
#    'Costa Rica': 884,
#    'Mexico': 989,
#    'Panama': 571,
#    'Argentina': 1241,
#    'Brazil': 1431,
#    'Colombia': 986,
#    'Peru': 1125,
#    'Uruguay': 1018,
#    'Belgium': 1298,
#    'Croatia': 945,
#    'Denmark': 1051,
#    'England': 1051,
#    'France': 1198,
#    'Germany': 1558,
#    'Iceland': 908,
#    'Poland': 1183,
#    'Portugal': 1274,
#    'Russia': 457,
#    'Serbia': 751,
#    'Spain': 1126,
#    'Sweden': 880,
#    'Switzerland': 1199
#}


def prob_tune_func(ranking_ratio):
    tune_factor = 12.0
    return 1.0/(1 + np.exp(-tune_factor*(ranking_ratio-0.5)))

def win_expectancy(rating_diff):
    return 1.0/(10**(-rating_diff/400.0) + 1)

def determine_winner(matchup):
    draw_prob = 0.74 # http://pena.lt/y/2015/12/12/frequency-of-draws-in-football/

    rating_diff = float(teams_points[matchup[0]]-teams_points[matchup[1]])
    #ranking_ratio = float(teams_points[matchup[0]])/(teams_points[matchup[0]]+teams_points[matchup[1]])

    draw_modifier = abs(teams_points[matchup[0]]-teams_points[matchup[1]])/5e3
    #draw_modifier = abs(teams_points[matchup[0]]-teams_points[matchup[1]])/1e4
    draw_prob += draw_modifier
    win_exp = draw_prob*win_expectancy(rating_diff)
    #win_exp = draw_prob*prob_tune_func(ranking_ratio)

    random_number = rnd.random()
    if random_number <= win_exp:
        winner = matchup[0]
    elif random_number > draw_prob:
        winner = 'Draw'
    else:
        winner = matchup[1]

    if knockout and winner is 'Draw':
        random_number = rnd.random()
        if random_number <= win_exp:
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




def knockout_runs(round16_matches_matches, n_runs):

    print " --- Computing %2s runs for Knockout stage --- " % (n_runs)
    
    orig_stdout = sys.stdout
    outfile = open('./knockout_stage/knockout_stage.txt', 'w')
    sys.stdout = outfile
        
    print "######### Knockout stage: #########\n\n"
    
    for i in range(1,n_runs+1):
        print "###### RUN %3s ######" % i

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
        
        print "\n"
    
    sys.stdout = orig_stdout
    outfile.close()

### Knockout stage: ###

knockout=True

# Round of 16:

round16_matches = {
        'Match 50': ['Denmark', 'Iceland'], # 1C - 2D
        'Match 49': ['Uruguay', 'Portugal'], # 1A - 2B
        'Match 51': ['Spain', 'Egypt'], # 1B - 2A
        'Match 52': ['Argentina', 'Peru'], # 1D - 2C
        'Match 53': ['Switzerland', 'Sweden'], # 1E - 2F
        'Match 54': ['England', 'Senegal'], # 1G - 2H
        'Match 55': ['Germany', 'Brazil'], # 1F - 2E
        'Match 56': ['Colombia', 'Belgium'] # 1H - 2G
        }

# Runs of knockout stage:
n_runs = 10

#knockout_runs(round16_matches, n_runs)


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
