#%%
import numpy as np
import matplotlib.pyplot as plt
import Population
import statistics 
%%matplotlib inline
%config InlineBackend.figure_format='retina'
plt.rcParams['figure.figsize'] = (10, 10)
#%%
#pop is population   
pop = Population.POPULATION(civil=1000, military=100, scientists=5)
    
events = [
    #CIVILS
    {"NAME": "ZOMBIE KILLS CIVIL",
     "W_a": lambda pop: 0.1*pop.CIVIL*pop.ZOMBIES / pop.total_population(),
     "EFFECT": lambda pop: pop.decrease_civil()
     },
    {"NAME": "CIVIL KILLS ZOMBIE",
     "W_a": lambda pop: 0.01*pop.CIVIL*pop.ZOMBIES / pop.total_population(),
     "EFFECT": lambda pop: pop.decrease_zombie()
     },
    {"NAME": "CIVIL GETS INFECTED",
     "W_a": lambda pop: 0.4*pop.CIVIL*pop.ZOMBIES / pop.total_population(),
     "EFFECT": lambda pop: pop.civil_becomes_zombie()
     },
    # MILITARY 
    {"NAME": "MILITARY GETS INFECTED",
     "W_a": lambda pop: (1-vaccine_effectiveness*pop.is_vaccine_invented())*0.02*pop.MILITARY*pop.ZOMBIES / pop.total_population(),
     "EFFECT": lambda pop: pop.military_becomes_zombie()
     },
    {"NAME": "MILITARY KILLS ZOMBIE",
     "W_a": lambda pop: 0.8*pop.MILITARY*pop.ZOMBIES / pop.total_population(),
     "EFFECT": lambda pop: pop.decrease_zombie()},
    {"NAME": "ZOMBIE KILLS MILITARY",
     "W_a": lambda pop: 0.02*pop.MILITARY*pop.ZOMBIES / pop.total_population(),
     "EFFECT": lambda pop: pop.decrease_military()
     },
     {"NAME": "MILITARY KILLS CIVIL",
     "W_a": lambda pop: 0.002*pop.MILITARY*pop.ZOMBIES*pop.CIVIL / pop.total_population(),
     "EFFECT": lambda pop: pop.decrease_civil()
     },
    # SCIENTISTS
    {"NAME": "VACCINE GETS INVENTED",
     "W_a": lambda pop: (pop.is_vaccine_invented())*.04*pop.SCIENTISTS,
     "EFFECT": lambda pop: pop.invent_vaccine()
    },
     {"NAME": "ZOMBIE KILLS SCIENTIST",
     "W_a": lambda pop: (1-pop.MILITARY/pop.total_population())*0.1*pop.SCIENTISTS*pop.ZOMBIES / pop.total_population(),
     "EFFECT": lambda pop: pop.decrease_scientists()
     },
     {"NAME": "SCIENTIST GETS INFECTED",
     "W_a": lambda pop: (1-vaccine_effectiveness*pop.is_vaccine_invented())*0.1*pop.SCIENTISTS*pop.ZOMBIES / pop.total_population(),
     "EFFECT": lambda pop: pop.scientist_becomes_zombie()
     },
     #RESISTANT
     {"NAME": "CIVIL BECOMES RESISTANT",
     "W_a": lambda pop: (1-pop.is_vaccine_invented())*0.9*pop.MILITARY*pop.CIVIL / pop.total_population(),
     "EFFECT": lambda pop: pop.civil_becomes_resistant()
     },
     {"NAME": "ZOMBIE KILLS RESISTANT",
     "W_a": lambda pop: 0.1*pop.RESISTANT*pop.ZOMBIES / pop.total_population(),
     "EFFECT": lambda pop: pop.decrease_resistants()
     },
     {"NAME": "MILITARY KILLS RESISTANT",
     "W_a": lambda pop: 0.002*pop.MILITARY*pop.RESISTANT*pop.CIVIL / pop.total_population(),
     "EFFECT": lambda pop: pop.decrease_resistants()
     },
     {"NAME": "RESISTANT KILLS ZOMBIE",
     "W_a": lambda pop: 0.01*pop.RESISTANT*pop.ZOMBIES / pop.total_population(),
     "EFFECT": lambda pop: pop.decrease_zombie()
     }
     
    ]
# Run N simulations
s=5
m=60
N = 40
ph_list = []
ts_list = []
for i in range(N):
    end_time = 300
    vaccine_effectiveness = 1  
    pop = Population.POPULATION(zombies=1, civil=11000, military=m, scientists=s)
    ts = Kendall_Feller(events, 0, end_time)
    ts_list.append(ts)
    phs = [ph for ph in pop.get_history().values()]
    ph_list.append(phs)

# Plot the result
mass_plot1(ts_list, ph_list, title='#S=' +str(s) + ', #M=' + str(m), alph=.05, linewidth=5)
#%%
'''
1. Place yourself immediately after and event
has occurred and compute the new populations.

2. Compute R = 􏰀α Wα(X). Generate an exponentially
distributed time T for the next event, with parameter R.

3. Generate a uniformly distributed 
number s in [0,R] and pick which event
has occurred as follows: if 􏰀β−1 Wα(X) ≤ s < 􏰀β Wα(X) we
say that event β has occurred.

4. Update Time, events and populations,
and repeat the process.
'''

def do(event):
    if event:
        event["EFFECT"](pop)
    

def Kendall_Feller_Step(events, time):
    R = 0
    R_sums = []
    for a in events:
        w_a = a["W_a"](pop)
        R += w_a*0.1
        R_sums.append(R)
        
    if R == 0:
        return end_time - time, None 
      
    T = np.random.exponential(scale=1/R)
    s = np.random.uniform(low=0, high=R)

    for b in range(len(events)):
        if R_sums[b] <= s < R_sums[b+1]:
            return T, events[b+1]
    return T, None             

def plot_pop_history(time, pop_history):
    for history in pop_history.values():
        plt.plot(time,history, marker="x")
    plt.legend(pop_history.keys())
    plt.show()

def Kendall_Feller(events, start, stop):
    time = start
    ts = [time]
    event = None
    
    while time < stop:
        do(event)
        T, event = Kendall_Feller_Step(events, time)
        time += T
        ts.append(time)
        pop.update_history()
        #if event:
            #print(time, event["NAME"])
    return ts
    

#ts = Kendall_Feller(events, 0, end_time)
#plot_pop_history(ts, pop.get_history())
#plt.show()

#%%

def merge_list_lists(list_lists):
    merged_list = []
    for list in list_lists:
        merged_list += list
    return merged_list

test = [[1,1,1,1,1,1,1,1,1], [1,2,1,3,2,4,1,2,1], [3,3,3,3,3,3,3,3,3]]

ts_list = test
#sorted(merge_list_list(ts_list))

def assign_pop(list1, list2, index1, index2):
    for i in range(len(list2)):
        list1[i][index1] = list2[i][index2]
    return list1

def scale_multiple_xs(long_ts, listof_short_xs, short_ts):
    #longer ts, shorter xs, scale xs to be same length as ts
    
    listof_scaled_xs = [ [None] * len(long_ts) for i in range(len(listof_short_xs)) ]
    for short_i in range(len(short_ts)):
        time = short_ts[short_i]
        long_i = long_ts.index(time)
        
        for i in range(len(listof_short_xs)): 
            listof_scaled_xs[i][long_i] = listof_short_xs[i][short_i] 
    return listof_scaled_xs   

def fill_nan(list):
    filled_list = [] 
    last_value = 0
    for value in list:
        if value == None:
            filled_list.append(last_value)
        else:
            last_value = value
            filled_list.append(last_value)
    return filled_list

def fill_nan_ph(ph_lists):
    out_list = []
    for ph in ph_lists:
       out_list.append(fill_nan(ph))
    return out_list

def main(ts_lists, ph_lists):     
    merged_ts = sorted(merge_list_lists(ts_lists))
    scaled_ph_lists = []
    for i in range(len(ph_lists)):
        nan_filled_ph_list = scale_multiple_xs(merged_ts, ph_lists[i], ts_lists[i])
        
        scaled_ph_lists.append(fill_nan_ph(nan_filled_ph_list))
    
    colors = [(0, 0, 1), (0, 0.5, 0), (1, 0, 0), (0, 0.75, 0.75), (0.75, 0, 0.75)]
    for list in scaled_ph_lists:
        
        ax1 = plt.subplot(511)
        plt.plot(merged_ts, list[0], color='r', alpha=.25)
        plt.ylabel('Zombies')
        plt.title('Scenario X')

        ax2 = plt.subplot(512)
        plt.plot(merged_ts, list[1] , color='r', alpha=.25)
        #plt.ylim((100000, 400000))
        plt.ylabel('Civil')

        ax2 = plt.subplot(513)
        plt.plot(merged_ts, list[2], color='r', alpha=.25)
        plt.ylabel('Military')

        ax2 = plt.subplot(514)
        plt.plot(merged_ts, list[3], color='r', alpha=.25)
        plt.ylabel('Scientists')

        ax2 = plt.subplot(515)
        plt.plot(merged_ts, list[4], color='r', alpha=.25)
        plt.ylabel('Resistant')

    plt.subplots_adjust(hspace=0)

        #i = 0
        #for ph in list:
        #    plt.plot(merged_ts, ph, color=colors[i], alpha=.2)
        #    i+=1
    #plt.legend(['Zombies', 'Civil', 'Military', 'Scientists', 'Resistant'])
    
    
    asdf = []
    for ph in scaled_ph_lists:
        asdf.append(ph[1])
    list_list = get_mean_and_extremes(asdf)
    print(len(asdf[0]), len(ts))
    print(len(list_list[0]), len(ts))
    #plt.plot(ts, list_list[1])
    #plt.fill_between(ts, list_list[0], list_list[2] )
    plt.show()            

def get_mean_and_extremes(list_list):
    max_list = []
    mean_list = []
    min_list = []
    for i in range(len(list_list[0])):
        # make a list of values @index
        tmp_list =[]
        for j in range(len(list_list)):
            tmp_list.append(list_list[j][i])
        max_list.append(max(tmp_list))
        mean_list.append(statistics.fmean(tmp_list))
        min_list.append(min(tmp_list))
    #rint(len(list_list[0]), len(max_list))
    return max_list, mean_list, min_list





# %% 


def mass_plot(ts_sims, sims, title='Scenario X', alph=.1):
    for i in range(len(sims)):
        sim=sims[i] 
        ts = ts_sims[i]      
        ax1 = plt.subplot(511)
        plt.plot(ts, sim[0], color='r', alpha=alph)
        plt.ylabel('Zombies')
        plt.title(title)

        ax2 = plt.subplot(512)
        plt.plot(ts, sim[1] , color='r', alpha=alph)
        #plt.ylim((100000, 400000))
        plt.ylabel('Civil')

        ax2 = plt.subplot(513)
        plt.plot(ts, sim[2], color='r', alpha=alph)
        plt.ylabel('Military')

        ax2 = plt.subplot(514)
        plt.plot(ts, sim[3], color='r', alpha=alph)
        plt.ylabel('Scientists')

        ax2 = plt.subplot(515)
        plt.plot(ts, sim[4], color='r', alpha=alph)
        plt.ylabel('Resistant')

    plt.subplots_adjust(hspace=0)
    plt.show()         

def mass_plot1(ts_sims, sims, title='Scenario X', alph=.1, linewidth=5):
    for i in range(len(sims)):
        sim=sims[i] 
        ts = ts_sims[i]      
        ax1 = plt.subplot(311)
        plt.plot(ts, sim[0], color='r', alpha=alph, linewidth=linewidth)
        plt.ylabel('Zombies')
        plt.title(title)

        ax2 = plt.subplot(312)
        plt.plot(ts, sim[1] , color='r', alpha=alph, linewidth=linewidth)
        #plt.ylim((100000, 400000))
        plt.ylabel('Civil')

        #ax2 = plt.subplot(513)
        #plt.plot(ts, sim[2], color='r', alpha=alph)
        #plt.ylabel('Military')

        #ax2 = plt.subplot(514)
        #plt.plot(ts, sim[3], color='r', alpha=alph)
        #plt.ylabel('Scientists')

        ax2 = plt.subplot(313)
        plt.plot(ts, sim[4], color='r', alpha=alph, linewidth=linewidth)
        plt.ylabel('Resistant')

    plt.subplots_adjust(hspace=0)
    plt.show()      

def mass_plot11(ts_sims, sims, title='Scenario X', alph=.1, linewidth=3):
    for i in range(len(sims)):
        sim=sims[i] 
        ts = ts_sims[i]      
        ax1 = plt.subplot(311)
        plt.plot(ts, sim[0], color='r', alpha=alph, linewidth=linewidth)
        #plt.ylabel('Zombies')
        plt.title(title)

        ax2 = plt.subplot(312)
        plt.plot(ts, sim[1] , color='r', alpha=alph, linewidth=linewidth)
        #plt.ylim((100000, 400000))
        #plt.ylabel('Civil')

        #ax2 = plt.subplot(513)
        #plt.plot(ts, sim[2], color='r', alpha=alph)
        #plt.ylabel('Military')

        #ax2 = plt.subplot(514)
        #plt.plot(ts, sim[3], color='r', alpha=alph)
        #plt.ylabel('Scientists')

        ax2 = plt.subplot(313)
        plt.plot(ts, sim[4], color='r', alpha=alph, linewidth=linewidth)
        #plt.ylabel('Resistant')

    plt.subplots_adjust(hspace=0)
    plt.show()         
#%% Run N simulations
N = 40
ph_list = []
ts_list = []
for i in range(N):
    end_time = 300
    vaccine_effectiveness = 1  
    pop = Population.POPULATION(zombies=1, civil=11000, military=60, scientists=5)
    ts = Kendall_Feller(events, 0, end_time)
    ts_list.append(ts)
    phs = [ph for ph in pop.get_history().values()]
    ph_list.append(phs)

# Plot the result
mass_plot(ts_list, ph_list)




#%%
# Import pandas and matplotlib.pyplot

# Create a sample DataFrame with different length and ending time series



# Plot the two series on the same figure with different x-axes
fig, ax = plt.subplots () # Create a new figure and axes
ax.plot ([1, 2, 2.1, 3, 4], [1.2, 1.3, 1.1, 0.9, 0.8], "-b", label = "Series 1") # Plot the first series with blue solid line
ax_tw = ax.twiny () # Create a twin axes that shares the y-axis
ax_tw.plot ([2, 3, 4.1], [0.7, 0.6, 0.4], "--r", label = "Series 2") # Plot the second series with red dashed line


ax.set_xlabel ("Time 1") # Add a label for the first x-axis
ax_tw.set_xlabel ("Time 2") # Add a label for the second x-axis
ax.set_ylabel ("Data values") # Add a label for the y-axis
ax.legend (loc = "upper left") # Add a legend for the first series
ax_tw.legend (loc = "upper right") # Add a legend for the second series
plt.show () # Show the plot
