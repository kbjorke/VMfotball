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

def prob_tune_func(ranking_ratio):
    tune_factor = 6.0
    return 1.0/(1 + np.exp(-tune_factor*(ranking_ratio-0.5)))

def determine_winner(matchup):
    draw_prob = 0.74 # http://pena.lt/y/2015/12/12/frequency-of-draws-in-football/

    ranking_ratio = float(teams_points[matchup[0]])/(teams_points[matchup[0]]+teams_points[matchup[1]])

    draw_modifier = abs(teams_points[matchup[0]]-teams_points[matchup[1]])/1e4
    draw_prob += draw_modifier
    ranking_ratio_tuned = draw_prob*prob_tune_func(ranking_ratio)

    random_number = rnd.random()
    if random_number <= ranking_ratio_tuned:
        winner = matchup[0]
    elif random_number > draw_prob:
        winner = 'Draw'
    else:
        winner = matchup[1]

    if knockout and winner is 'Draw':
        random_number = rnd.random()
        if random_number <= ranking_ratio:
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

    if winner == 'Draw':
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

def group_play(group_teams, matches):
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

### Group play: ###

group = sys.argv[1] # Commandline arguments for what group to compute

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

# Group B:

groupB_teams = ['Portugal', 'Spain', 'Morocco', 'Iran']

groupB_matches = {
        'Match 4': ['Morocco', 'Iran'],
        'Match 3': ['Portugal', 'Spain'],
        'Match 19': ['Portugal', 'Morocco'],
        'Match 20': ['Iran', 'Spain'],
        'Match 35': ['Iran', 'Portugal'],
        'Match 36': ['Spain', 'Morocco']
        }

# Group C:

groupC_teams = ['France', 'Australia', 'Peru', 'Denmark']

groupC_matches = {
        'Match 5': ['France', 'Australia'],
        'Match 6': ['Peru', 'Denmark'],
        'Match 22': ['Denmark', 'Australia'],
        'Match 21': ['France', 'Peru'],
        'Match 37': ['Denmark', 'France'],
        'Match 38': ['Australia', 'Peru']
        }

# Group D:

groupD_teams = ['Argentina', 'Iceland', 'Croatia', 'Nigeria']

groupD_matches = {
        'Match 7': ['Argentina', 'Iceland'],
        'Match 8': ['Croatia', 'Nigeria'],
        'Match 23': ['Argentina', 'Croatia'],
        'Match 24': ['Nigeria', 'Iceland'],
        'Match 39': ['Nigeria', 'Argentina'],
        'Match 40': ['Iceland', 'Croatia']
        }

# Group E:

groupE_teams = ['Brazil', 'Switzerland', 'Costa Rica', 'Serbia']

groupE_matches = {
        'Match 10': ['Costa Rica', 'Serbia'],
        'Match 9': ['Brazil', 'Switzerland'],
        'Match 25': ['Brazil', 'Costa Rica'],
        'Match 26': ['Serbia', 'Switzerland'],
        'Match 41': ['Serbia', 'Brazil'],
        'Match 42': ['Switzerland', 'Costa Rica']
        }

# Group F:

groupF_teams = ['Germany', 'Mexico', 'Sweden', 'South Korea']

groupF_matches = {
        'Match 11': ['Germany', 'Mexico'],
        'Match 12': ['Sweden', 'South Korea'],
        'Match 28': ['South Korea', 'Mexico'],
        'Match 27': ['Germany', 'Sweden'],
        'Match 43': ['South Korea', 'Germany'],
        'Match 44': ['Mexico', 'Sweden']
        }

# Group G:

groupG_teams = ['Belgium', 'Panama', 'Tunisia', 'England']

groupG_matches = {
        'Match 13': ['Belgium', 'Panama'],
        'Match 14': ['Tunisia', 'England'],
        'Match 29': ['Belgium', 'Tunisia'],
        'Match 30': ['England', 'Panama'],
        'Match 45': ['England', 'Belgium'],
        'Match 46': ['Panama', 'Tunisia']
        }

# Group H:

groupH_teams = ['Poland', 'Senegal', 'Colombia', 'Japan'] 
groupH_matches = {
        'Match 16': ['Colombia', 'Japan'],
        'Match 15': ['Poland', 'Senegal'],
        'Match 32': ['Japan', 'Senegal'],
        'Match 31': ['Poland', 'Colombia'],
        'Match 47': ['Japan', 'Poland'],
        'Match 48': ['Senegal', 'Colombia']
        }


# Runs for groups:

n_runs = 10

if group == "groupA" or group == "all":
    group_runs("A", groupA_teams, groupA_matches, n_runs)

if group == "groupB" or group == "all":
    group_runs("B", groupB_teams, groupB_matches, n_runs)

if group == "groupC" or group == "all":
    group_runs("C", groupC_teams, groupC_matches, n_runs)

if group == "groupD" or group == "all":
    group_runs("D", groupD_teams, groupD_matches, n_runs)

if group == "groupE" or group == "all":
    group_runs("E", groupE_teams, groupE_matches, n_runs)

if group == "groupF" or group == "all":
    group_runs("F", groupF_teams, groupF_matches, n_runs)

if group == "groupG" or group == "all":
    group_runs("G", groupG_teams, groupG_matches, n_runs)

if group == "groupH" or group == "all":
    group_runs("H", groupH_teams, groupH_matches, n_runs)
