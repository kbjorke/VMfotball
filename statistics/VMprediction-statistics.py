#!/usr/bin/env python

# By: Kristian Bjoerke

import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import sys

from matplotlib.lines import Line2D

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

def scores_pdf(matchup, score_prob_matrix):
    draw_modifier = draw_mod(matchup)
    rating_diff = float(teams_points[matchup[0]]-teams_points[matchup[1]])
    
    prob_win_1 = (draw_prob+draw_modifier)*win_expectancy(rating_diff)
    prob_win_2 = (draw_prob+draw_modifier)*win_expectancy(-rating_diff)
    prob_draw = 1-(draw_prob+draw_modifier)

    print prob_win_1,prob_win_2,prob_draw

    prob_density_matrix = np.zeros([6,6])
   
    for i in range(6):
        for j in range(6):
            if i == j:
                prob_density_matrix[i,j] = score_prob_matrix[i,j]*prob_draw
            elif i > j:
                prob_density_matrix[i,j] = score_prob_matrix[i,j]*prob_win_1
            elif i < j:
                prob_density_matrix[i,j] = score_prob_matrix[i,j]*prob_win_2

    return prob_density_matrix

def result_prob(matchup):
    draw_modifier = draw_mod(matchup)
    rating_diff = float(teams_points[matchup[0]]-teams_points[matchup[1]])

    prob_win_1 = (draw_prob+draw_modifier)*win_expectancy(rating_diff)
    prob_win_2 = (draw_prob+draw_modifier)*win_expectancy(-rating_diff)
    prob_draw = 1-(draw_prob+draw_modifier)

    return [prob_win_1, prob_win_2, prob_draw]

def pvalue(score, match_prob_matrix):
    return np.sum(match_prob_matrix[np.where(match_prob_matrix <= match_prob_matrix[score[0],score[1]])])

def get_std(score_matrix):
    sigma1_matrix = np.ones([6,6])
    prob_matrix = np.copy(score_matrix)
    
    pval = np.sum(prob_matrix)
    while pval > 0.32:
        sigma1_matrix[np.where(prob_matrix==np.max(prob_matrix))] = 0
        prob_matrix[np.where(prob_matrix==np.max(prob_matrix))] = 0
        pval = np.sum(prob_matrix)
    
    sigma2_matrix = np.copy(sigma1_matrix)
    
    pval = np.sum(prob_matrix)
    while pval > 0.05:
        sigma2_matrix[np.where(prob_matrix==np.max(prob_matrix))] = 0
        prob_matrix[np.where(prob_matrix==np.max(prob_matrix))] = 0
        pval = np.sum(prob_matrix)

    return [sigma1_matrix, sigma2_matrix]

def make_pdf_plot(analysis_data):
    match = analysis_data[0]
    matchup = analysis_data[1]
    predicted_score = analysis_data[2]
    observed_score = analysis_data[3]
    match_prob_matrix = analysis_data[4]
    sigma1_matrix = analysis_data[5]
    sigma2_matrix = analysis_data[6]
    prob_win_1 = analysis_data[7]
    prob_win_2 = analysis_data[8]
    prob_draw = analysis_data[9]

    font_size = 15
    font_size_title = 18
    marker_size = 20
    marker_size2 = 14
    line_width = 3

    fig, ax = plt.subplots()
    
    predict = plt.plot(predicted_score[1], predicted_score[0], marker='o', markersize=marker_size, color="blue")
    if observed_score != "N/A":
        observe = plt.plot(observed_score[1], observed_score[0], marker='o', markersize=marker_size2, color="green")
    
    plt.imshow(100*match_prob_matrix, cmap="hot", interpolation='nearest')
    cbar = plt.colorbar()
    s1 = plt.contour(sigma1_matrix, colors="y", levels=[0.5], linewidths=line_width)
    s2 = plt.contour(sigma2_matrix, colors="r", levels=[0.5], linewidths=line_width)
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 

    ax.legend(handles=[
        Line2D([], [], marker='o', color="w", label="Predicted", 
            markerfacecolor="blue", markersize=marker_size), 
        Line2D([0,0], [0,1], color="y", label="p < 0.32", linewidth=line_width),
        Line2D([], [], marker='o', color="w", label="Observed", 
            markerfacecolor="green", markersize=marker_size2), 
        Line2D([0,0], [0,1], color="r", label="p < 0.05", linewidth=line_width)
        ], loc='lower right', numpoints=1, ncol=2, bbox_to_anchor=(1.07, -0.07))

    plt.xlabel(matchup[1]+" [goals] | Win: %.1f %s" % (100*prob_win_2, "%"))
    plt.ylabel(matchup[0]+" [goals] | Win: %.1f %s" % (100*prob_win_1, "%"))

    cbar.ax.set_ylabel("Probability density [%]", rotation=270, labelpad=20, fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    ax.set_title("%s: %s - %s | Draw: %.1f %s" % (match, matchup[0], matchup[1], 100*prob_draw, "%"), x=0.5, y=-0.12, fontsize=font_size_title)

    fig_title="./pdfs/%s-%s-%s.png" %(match.replace(" ", "_"),matchup[0].replace(" ", "_"),matchup[1].replace(" ", "_"))
    plt.savefig(fig_title)
    print 'Figure "%s" made!' % fig_title
    plt.close()

def analysis_predictions(matches):
    for match in matches:
        matchup = matches[match][0]
        predicted_score = matches[match][1]
        observed_score = matches[match][2]
        
        #print matchup,predicted_score,observed_score
        
        match_prob_matrix = scores_pdf(matchup, score_prob_matrix)
        
        #print match_prob_matrix
        
        sigma1_matrix, sigma2_matrix = get_std(match_prob_matrix)
        
        print "Pval predicted: %f" % pvalue(predicted_score, match_prob_matrix)
        if observed_score == "N/A":
            print "Pval observed : N/A"
        else:
            print "Pval observed : %f" % pvalue(observed_score, match_prob_matrix)
   
        match_result_prob = result_prob(matchup)
        prob_win_1 = match_result_prob[0]
        prob_win_2 = match_result_prob[1]
        prob_draw = match_result_prob[2]

        analysis_data = [match, matchup, predicted_score, observed_score, match_prob_matrix, sigma1_matrix, sigma2_matrix, prob_win_1, prob_win_2, prob_draw]

        make_pdf_plot(analysis_data)

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
    
    print "###### RUN %3s ######" % i
    
    results, score_table = group_play(group_teams, group_matches, cup_is_cup)
    print_results(results)
    print_score_table(score_table)
    print "\n"
    
    sys.stdout = orig_stdout
    outfile.close()


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


group_stage_matches = {
        'Match 01': [['Russia', 'Saudi Arabia'], 
            [1, 1],     # predicted
            [5, 0]],    # observed
        'Match 02': [['Egypt', 'Uruguay'],
            [1, 2],     # predicted
            [0, 1]],    # observed
        'Match 04': [['Morocco', 'Iran'],
            [1, 1],     # predicted
            [0, 1]],    # observed
        'Match 03': [['Portugal', 'Spain'],
            [0, 1],     # predicted
            [3, 3]],    # observed
        'Match 05': [['France', 'Australia'],
            [2, 0],     # predicted
            [2, 1]],    # observed
        'Match 07': [['Argentina', 'Iceland'],
            [2, 1],     # predicted
            [1, 1]],    # observed
        'Match 06': [['Peru', 'Denmark'],
            [1, 0],     # predicted
            [0, 1]],    # observed
        'Match 08': [['Croatia', 'Nigeria'],
            [0, 3],     # predicted
            [2, 0]],    # observed
        'Match 10': [['Costa Rica', 'Serbia'],
            [1, 1],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 11': [['Germany', 'Mexico'],
            [2, 0],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 09': [['Brazil', 'Switzerland'],
            [1, 0],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 12': [['Sweden', 'South Korea'],
            [0, 2],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 13': [['Belgium', 'Panama'],
            [4, 1],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 14': [['Tunisia', 'England'],
            [2, 3],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 16': [['Colombia', 'Japan'],
            [1, 0],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 15': [['Poland', 'Senegal'],
            [0, 0],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 17': [['Russia', 'Egypt'],
            [0, 1],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 19': [['Portugal', 'Morocco'],
            [3, 1],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 18': [['Uruguay', 'Saudi Arabia'],
            [3, 0],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 20': [['Iran', 'Spain'],
            [3, 5],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 22': [['Denmark', 'Australia'],
            [2, 0],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 21': [['France', 'Peru'],
            [4, 2],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 23': [['Argentina', 'Croatia'],
            [0, 2],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 25': [['Brazil', 'Costa Rica'],
            [1, 0],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 24': [['Nigeria', 'Iceland'],
            [0, 1],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 26': [['Serbia', 'Switzerland'],
            [1, 4],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 29': [['Belgium', 'Tunisia'],
            [1, 0],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 28': [['South Korea', 'Mexico'],
            [0, 3],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 27': [['Germany', 'Sweden'],
            [1, 1],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 30': [['England', 'Panama'],
            [2, 0],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 32': [['Japan', 'Senegal'],
            [1, 3],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 31': [['Poland', 'Colombia'],
            [0, 1],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 33': [['Uruguay', 'Russia'],
            [3, 0],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 34': [['Saudi Arabia', 'Egypt'],
            [3, 3],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 35': [['Spain', 'Morocco'],
            [4, 0],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 35': [['Iran', 'Portugal'],
            [1, 2],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 38': [['Australia', 'Peru'],
            [1, 2],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 37': [['Denmark', 'France'],
            [0, 1],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 39': [['Nigeria', 'Argentina'],
            [2, 5],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 40': [['Iceland', 'Croatia'],
            [1, 0],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 43': [['South Korea', 'Germany'],
            [0, 1],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 44': [['Mexico', 'Sweden'],
            [2, 0],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 41': [['Serbia', 'Brazil'],
            [0, 3],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 42': [['Switzerland', 'Costa Rica'],
            [3, 1],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 47': [['Japan', 'Poland'],
            [2, 0],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 48': [['Senegal', 'Colombia'],
            [2, 2],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 46': [['Panama', 'Tunisia'],
            [2, 0],     # predicted
            "N/A"], #[0, 0]],    # observed
        'Match 45': [['England', 'Belgium'],
            [2, 4],     # predicted
            "N/A"], #[0, 0]],    # observed
        }

analysis_predictions(group_stage_matches)

#for match in group_stage_matches:
#    matchup = group_stage_matches[match][0]
#    predicted_score = group_stage_matches[match][1]
#    observed_score = group_stage_matches[match][2]
#
#    print matchup,predicted_score,observed_score
#
#    match_prob_matrix = scores_pdf(matchup, score_prob_matrix)
#
#    print match_prob_matrix
#
#    sigma1_matrix, sigma2_matrix = get_std(match_prob_matrix)
#
#    print pvalue(predicted_score, match_prob_matrix)
#    print pvalue(observed_score, match_prob_matrix)
#
#    match_result_prob = result_prob(matchup)
#    prob_win_1 = match_result_prob[0]
#    prob_win_2 = match_result_prob[1]
#    prob_draw = match_result_prob[2]
#
#    fix, ax = plt.subplots()
#    predict = plt.plot(predicted_score[1], predicted_score[0], marker='o', markersize=14, color="blue")
#    observe = plt.plot(observed_score[1], observed_score[0], marker='o', markersize=14, color="green")
#    plt.imshow(100*match_prob_matrix, cmap="hot", interpolation='nearest')
#    cbar = plt.colorbar()
#    s1 = plt.contour(sigma1_matrix, colors="y", levels=[0.5], linewidths=3)
#    s2 = plt.contour(sigma2_matrix, colors="r", levels=[0.5], linewidths=3)
#    ax.xaxis.tick_top()
#    ax.xaxis.set_label_position('top') 
##    ax.legend(handles=[Line2D([], [], marker='o', color="w", label="Predicted", markerfacecolor="blue", markersize=14), Line2D([], [], marker='o', color="w", label="Observed", markerfacecolor="green", markersize=14)], loc='lower right', numpoints=1)
#    ax.legend(handles=[
#        Line2D([], [], marker='o', color="w", label="Predicted", 
#            markerfacecolor="blue", markersize=14), 
#        Line2D([0,0], [0,1], color="y", label="p < 0.32", linewidth=3), 
#        Line2D([], [], marker='o', color="w", label="Observed", 
#            markerfacecolor="green", markersize=14), 
#        Line2D([0,0], [0,1], color="r", label="p < 0.05", linewidth=3)
#        ], loc='lower right', numpoints=1, ncol=2, bbox_to_anchor=(1.05, -0.12))
#    plt.xlabel(matchup[1]+" [goals] | Win: %.1f %s" % (100*prob_win_2, "%"))
#    plt.ylabel(matchup[0]+" [goals] | Win: %.1f %s" % (100*prob_win_1, "%"))
#    cbar.ax.set_ylabel("Probability density [%]", rotation=270, labelpad=20)
#    plt.show()

#fix, ax = plt.subplots()
#predict = plt.plot(match_exp[1], match_exp[0], marker='o', markersize=14, color="blue")
#observe = plt.plot(match_obs[1], match_obs[0], marker='o', markersize=14, color="green")
#plt.imshow(100*score_prob_matrix, cmap="hot", interpolation='nearest')
#cbar = plt.colorbar()
##plt.imshow(100*np.ma.masked_values(sigma2_matrix, 0), cmap="Wistia", interpolation='nearest', alpha=0.3)
#s1 = plt.contour(sigma1_matrix, colors="y", levels=[0.5], linewidths=3)
#s2 = plt.contour(sigma2_matrix, colors="r", levels=[0.5], linewidths=3)
#ax.xaxis.tick_top()
#ax.xaxis.set_label_position('top') 
#ax.legend(handles=[
#    Line2D([], [], marker='o', color="w", label="Predicted", 
#        markerfacecolor="blue", markersize=14), 
#    Line2D([0,0], [0,1], color="y", label="p < 0.32", linewidth=3), 
#    Line2D([], [], marker='o', color="w", label="Observed", 
#        markerfacecolor="green", markersize=14), 
#    Line2D([0,0], [0,1], color="r", label="p < 0.05", linewidth=3)
#    ], loc='lower right', numpoints=1, ncol=2, bbox_to_anchor=(1.05, -0.12))
##ax.legend([predict, observe], ["Predicted", "Observed"], loc='center')
#plt.xlabel(match[1]+" [goals] | Win: %.1f %s" % (100*prob_win_2, "%"))
#plt.ylabel(match[0]+" [goals] | Win: %.1f %s" % (100*prob_win_1, "%"))
#cbar.ax.set_ylabel("Probability density [%]", rotation=270, labelpad=20)
#plt.show()

##match1 = ['Russia', 'Saudi Arabia', 5, 0]
#match1 = ['Russia', 'Saudi Arabia']
#match1_exp = [1, 1]
#match1_obs = [5, 0]
#match2 = ['Egypt', 'Uruguay']
#match2_exp = [1, 2]
#match2_obs = [0, 1]
#match3 = ['Portugal', 'Spain']
#match3_exp = [0, 1]
#match3_obs = [3, 3]
#match4 = ['Morocco', 'Iran']
#match4_exp = [1, 1]
#match4_obs = [0, 1]
#
#match = match1
##match = match2
##match = match3
##match = match4
#match_exp = match1_exp
#match_obs = match1_obs
##match_exp = match2_exp
##match_obs = match2_obs
##match_exp = match3_exp
##match_obs = match3_obs
##match_exp = match4_exp
##match_obs = match4_obs
#
#draw_modifier = draw_mod(match)
#rating_diff = float(teams_points[match[0]]-teams_points[match[1]])
#
#prob_win_1 = (draw_prob+draw_modifier)*win_expectancy(rating_diff)
#prob_win_2 = (draw_prob+draw_modifier)*win_expectancy(-rating_diff)
#prob_draw = 1-(draw_prob+draw_modifier)
#
#print prob_win_1,prob_win_2,prob_draw
#
#for i in range(6):
#    for j in range(6):
#        if i == j:
#            score_prob_matrix[i,j] *= prob_draw
#        elif i > j:
#            score_prob_matrix[i,j] *= prob_win_1
#        elif i < j:
#            score_prob_matrix[i,j] *= prob_win_2
#
#sigma1_matrix, sigma2_matrix = get_std(score_prob_matrix)
#
#print score_prob_matrix
#
#print np.sum(score_prob_matrix[np.where(score_prob_matrix <= score_prob_matrix[match_exp[0],match_exp[1]])])
#print np.sum(score_prob_matrix[np.where(score_prob_matrix <= score_prob_matrix[match_obs[0],match_obs[1]])])
#
#
#fix, ax = plt.subplots()
#predict = plt.plot(match_exp[1], match_exp[0], marker='o', markersize=14, color="blue")
#observe = plt.plot(match_obs[1], match_obs[0], marker='o', markersize=14, color="green")
#plt.imshow(100*score_prob_matrix, cmap="hot", interpolation='nearest')
#cbar = plt.colorbar()
##plt.imshow(100*np.ma.masked_values(sigma2_matrix, 0), cmap="Wistia", interpolation='nearest', alpha=0.3)
#s1 = plt.contour(sigma1_matrix, colors="y", levels=[0.5], linewidths=3)
#s2 = plt.contour(sigma2_matrix, colors="r", levels=[0.5], linewidths=3)
#ax.xaxis.tick_top()
#ax.xaxis.set_label_position('top') 
#ax.legend(handles=[
#    Line2D([], [], marker='o', color="w", label="Predicted", 
#        markerfacecolor="blue", markersize=14), 
#    Line2D([0,0], [0,1], color="y", label="p < 0.32", linewidth=3), 
#    Line2D([], [], marker='o', color="w", label="Observed", 
#        markerfacecolor="green", markersize=14), 
#    Line2D([0,0], [0,1], color="r", label="p < 0.05", linewidth=3)], 
#    loc='lower right', numpoints=1, ncol=2, bbox_to_anchor=(1.05, -0.12))
##ax.legend([predict, observe], ["Predicted", "Observed"], loc='center')
#plt.xlabel(match[1]+" [goals] | Win: %.1f %s" % (100*prob_win_2, "%"))
#plt.ylabel(match[0]+" [goals] | Win: %.1f %s" % (100*prob_win_1, "%"))
#cbar.ax.set_ylabel("Probability density [%]", rotation=270, labelpad=20)
#plt.show()
#
#plt.figure()
#plt.imshow(sigma1_matrix, cmap="hot", interpolation='nearest')
#plt.show()
#
#plt.figure()
#plt.imshow(sigma2_matrix, cmap="hot", interpolation='nearest')
#plt.show()
