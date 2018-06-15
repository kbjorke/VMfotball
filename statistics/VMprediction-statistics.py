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

def draw_mod(matchup):
    return abs(teams_points[matchup[0]]-teams_points[matchup[1]])/5e3

def win_expectancy(rating_diff):
    return 1.0/(10**(-rating_diff/400.0) + 1)

def prediction_pdf(match, draw_prob):
    team1 = match[0]
    team2 = match[1]
    team1_goals = match[2]
    team2_goals = match[3]

def determine_winner(matchup, cup_is_cup=False):
    draw_prob = 0.74 # http://pena.lt/y/2015/12/12/frequency-of-draws-in-football/

    rating_diff = float(teams_points[matchup[0]]-teams_points[matchup[1]])
    #ranking_ratio = float(teams_points[matchup[0]])/(teams_points[matchup[0]]+teams_points[matchup[1]])

    draw_modifier = abs(teams_points[matchup[0]]-teams_points[matchup[1]])/5e3
    #draw_modifier = abs(teams_points[matchup[0]]-teams_points[matchup[1]])/1e4
    if cup_is_cup is True:
        draw_modifier = 0
        win_exp = 0.5*draw_prob
    else:
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

def group_play(group_teams, matches, cup_is_cup=False):
    results = dict()
    for match in matches:
        results[match] = ['team1', 'team2', 'results']

    score_table = dict()
    for team in group_teams:
        score_table[team] = [0,0,0,0,0,0,0]

    for match in matches:
        teams = matches[match]
        winner = determine_winner(teams, cup_is_cup)
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

def group_runs(label, group_teams, group_matches, n_runs, cup_is_cup=False):

    print " --- Computing %2s runs for Group %s --- " % (n_runs, label)
    
    orig_stdout = sys.stdout
    outfile = open('./group_stage/group%s.txt' % label, 'w')
    sys.stdout = outfile
        
    print "######### Group stage: Group %s #########\n\n" % label
    
    for i in range(1,n_runs+1):
        print "###### RUN %3s ######" % i
    
        results, score_table = group_play(group_teams, group_matches, cup_is_cup)
        print_results(results)
        print_score_table(score_table)
        print "\n"
    
    sys.stdout = orig_stdout
    outfile.close()

#### Group play: ###
#
#group = sys.argv[1] # Commandline arguments for what group to compute
#
#knockout=False
#
## Group A:
#
#groupA_teams = ['Russia', 'Saudi Arabia', 'Egypt', 'Uruguay']
#
#groupA_matches = {
#        'Match 1': ['Russia', 'Saudi Arabia'],
#        'Match 2': ['Egypt', 'Uruguay'],
#        'Match 17': ['Russia', 'Egypt'],
#        'Match 18': ['Uruguay', 'Saudi Arabia'],
#        'Match 33': ['Uruguay', 'Russia'],
#        'Match 34': ['Saudi Arabia', 'Egypt']
#        }
#
## Group B:
#
#groupB_teams = ['Portugal', 'Spain', 'Morocco', 'Iran']
#
#groupB_matches = {
#        'Match 4': ['Morocco', 'Iran'],
#        'Match 3': ['Portugal', 'Spain'],
#        'Match 19': ['Portugal', 'Morocco'],
#        'Match 20': ['Iran', 'Spain'],
#        'Match 35': ['Iran', 'Portugal'],
#        'Match 36': ['Spain', 'Morocco']
#        }
#
## Group C:
#
#groupC_teams = ['France', 'Australia', 'Peru', 'Denmark']
#
#groupC_matches = {
#        'Match 5': ['France', 'Australia'],
#        'Match 6': ['Peru', 'Denmark'],
#        'Match 22': ['Denmark', 'Australia'],
#        'Match 21': ['France', 'Peru'],
#        'Match 37': ['Denmark', 'France'],
#        'Match 38': ['Australia', 'Peru']
#        }
#
## Group D:
#
#groupD_teams = ['Argentina', 'Iceland', 'Croatia', 'Nigeria']
#
#groupD_matches = {
#        'Match 7': ['Argentina', 'Iceland'],
#        'Match 8': ['Croatia', 'Nigeria'],
#        'Match 23': ['Argentina', 'Croatia'],
#        'Match 24': ['Nigeria', 'Iceland'],
#        'Match 39': ['Nigeria', 'Argentina'],
#        'Match 40': ['Iceland', 'Croatia']
#        }
#
## Group E:
#
#groupE_teams = ['Brazil', 'Switzerland', 'Costa Rica', 'Serbia']
#
#groupE_matches = {
#        'Match 10': ['Costa Rica', 'Serbia'],
#        'Match 9': ['Brazil', 'Switzerland'],
#        'Match 25': ['Brazil', 'Costa Rica'],
#        'Match 26': ['Serbia', 'Switzerland'],
#        'Match 41': ['Serbia', 'Brazil'],
#        'Match 42': ['Switzerland', 'Costa Rica']
#        }
#
## Group F:
#
#groupF_teams = ['Germany', 'Mexico', 'Sweden', 'South Korea']
#
#groupF_matches = {
#        'Match 11': ['Germany', 'Mexico'],
#        'Match 12': ['Sweden', 'South Korea'],
#        'Match 28': ['South Korea', 'Mexico'],
#        'Match 27': ['Germany', 'Sweden'],
#        'Match 43': ['South Korea', 'Germany'],
#        'Match 44': ['Mexico', 'Sweden']
#        }
#
## Group G:
#
#groupG_teams = ['Belgium', 'Panama', 'Tunisia', 'England']
#
#groupG_matches = {
#        'Match 13': ['Belgium', 'Panama'],
#        'Match 14': ['Tunisia', 'England'],
#        'Match 29': ['Belgium', 'Tunisia'],
#        'Match 30': ['England', 'Panama'],
#        'Match 45': ['England', 'Belgium'],
#        'Match 46': ['Panama', 'Tunisia']
#        }
#
## Group H:
#
#groupH_teams = ['Poland', 'Senegal', 'Colombia', 'Japan'] 
#groupH_matches = {
#        'Match 16': ['Colombia', 'Japan'],
#        'Match 15': ['Poland', 'Senegal'],
#        'Match 32': ['Japan', 'Senegal'],
#        'Match 31': ['Poland', 'Colombia'],
#        'Match 47': ['Japan', 'Poland'],
#        'Match 48': ['Senegal', 'Colombia']
#        }
#
#
## Runs for groups:
#
#n_runs = 1
#cup_is_cup=False
#
#if group == "groupA" or group == "all":
#    group_runs("A", groupA_teams, groupA_matches, n_runs, cup_is_cup)
#
#if group == "groupB" or group == "all":
#    group_runs("B", groupB_teams, groupB_matches, n_runs, cup_is_cup)
#
#if group == "groupC" or group == "all":
#    group_runs("C", groupC_teams, groupC_matches, n_runs, cup_is_cup)
#
#if group == "groupD" or group == "all":
#    group_runs("D", groupD_teams, groupD_matches, n_runs, cup_is_cup)
#
#if group == "groupE" or group == "all":
#    group_runs("E", groupE_teams, groupE_matches, n_runs, cup_is_cup)
#
#if group == "groupF" or group == "all":
#    group_runs("F", groupF_teams, groupF_matches, n_runs, cup_is_cup)
#
#if group == "groupG" or group == "all":
#    group_runs("G", groupG_teams, groupG_matches, n_runs, cup_is_cup)
#
#if group == "groupH" or group == "all":
#    group_runs("H", groupH_teams, groupH_matches, n_runs, cup_is_cup)

draw_prob = 0.74 # http://pena.lt/y/2015/12/12/frequency-of-draws-in-football/

# SCORE PROBS: https://fivethirtyeight.com/features/in-126-years-english-football-has-seen-13475-nil-nil-draws/
draw_probs = np.array([7.2, 11.6, 5.2, 1.1, 0.2, 0])
win_probs = np.array([9.8+6.3, 8.9+5.6, 8.1+3.4, 2.8+1.8, 5.2+2.3, 4.8+1.4, 0.5+0.3, 1.4+0.6, 2.5+0.7, 2.3+0.4, 0.1+0.05, 0.2+0.1, 0.6+0.2, 1.1+0.2, 1.0+0.1])

draw_probs = draw_probs/sum(draw_probs)
win_probs = win_probs/sum(win_probs)
#win_probs = win_probs/2
score_prob_matrix = np.array([
    [draw_probs[0], win_probs[0], win_probs[2], win_probs[5], win_probs[9], win_probs[14]],
    [win_probs[0], draw_probs[1], win_probs[1], win_probs[4], win_probs[8], win_probs[13]],
    [win_probs[2], win_probs[1], draw_probs[2], win_probs[3], win_probs[7], win_probs[12]],
    [win_probs[5], win_probs[4], win_probs[3], draw_probs[3], win_probs[6], win_probs[11]],
    [win_probs[9], win_probs[8], win_probs[7], win_probs[6], draw_probs[4], win_probs[10]],
    [win_probs[14], win_probs[13], win_probs[12], win_probs[11], win_probs[10], draw_probs[5]],
    ])

draw_results = ['0 - 0', '1 - 1', '2 - 2', '3 - 3', '4 - 4', '5 - 5']
win_results_0 = ['1 - 0', '2 - 1', '2 - 0', '3 - 2', '3 - 1', '3 - 0', '4 - 3', '4 - 2', '4 - 1', '4 - 0', '5 - 4', '5 - 3', '5 - 2', '5 - 1', '5 - 0']
win_results_1 = ['0 - 1', '1 - 2', '0 - 2', '2 - 3', '1 - 3', '0 - 3', '3 - 4', '2 - 4', '1 - 4', '0 - 4', '4 - 5', '3 - 5', '2 - 5', '1 - 5', '0 - 5']
results = np.array([
    [draw_results[0], win_results_1[0], win_results_1[2], win_results_1[5], win_results_1[9], win_results_1[14]],
    [win_results_0[0], draw_results[1], win_results_1[1], win_results_1[4], win_results_1[8], win_results_1[13]],
    [win_results_0[2], win_results_0[1], draw_results[2], win_results_1[3], win_results_1[7], win_results_1[12]],
    [win_results_0[5], win_results_0[4], win_results_0[3], draw_results[3], win_results_1[6], win_results_1[11]],
    [win_results_0[9], win_results_0[8], win_results_0[7], win_results_0[6], draw_results[4], win_results_1[10]],
    [win_results_0[14], win_results_0[13], win_results_0[12], win_results_0[11], win_results_0[10], draw_results[5]],
    ])
print results

#score_prob_matrix = score_prob_matrix/np.sum(score_prob_matrix)
#print np.sum(score_prob_matrix)

#print score_prob_matrix

#match1 = ['Russia', 'Saudi Arabia', 5, 0]
match1 = ['Russia', 'Saudi Arabia']
match1_exp = [1, 1]
match1_obs = [5, 0]
match2 = ['Egypt', 'Uruguay']
match3 = ['Portugal', 'Spain']
match4 = ['Morocco', 'Iran']

match = match1
#match = match2
#match = match3
#match = match4

draw_modifier = draw_mod(match)
rating_diff = float(teams_points[match[0]]-teams_points[match[1]])

prob_win_1 = (draw_prob+draw_modifier)*win_expectancy(rating_diff)
prob_win_2 = (draw_prob+draw_modifier)*win_expectancy(-rating_diff)
prob_draw = 1-(draw_prob+draw_modifier)

print prob_win_1,prob_win_2,prob_draw

for i in range(6):
    for j in range(6):
        if i == j:
            score_prob_matrix[i,j] *= prob_draw
        elif i > j:
            score_prob_matrix[i,j] *= prob_win_1
        elif i < j:
            score_prob_matrix[i,j] *= prob_win_2

        
print score_prob_matrix

print np.sum(score_prob_matrix[np.where(score_prob_matrix <= score_prob_matrix[match1_exp[0],match1_exp[1]])])
print np.sum(score_prob_matrix[np.where(score_prob_matrix <= score_prob_matrix[match1_obs[0],match1_obs[1]])])

fix, ax = plt.subplots()
plt.imshow(100*score_prob_matrix[:5,:5], cmap="hot", interpolation='nearest')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top') 
cbar = plt.colorbar()
plt.xlabel(match[1]+" [goals]")
plt.ylabel(match[0]+" [goals]")
cbar.ax.set_ylabel("Probability density [%]", rotation=270, labelpad=20)
plt.show()
