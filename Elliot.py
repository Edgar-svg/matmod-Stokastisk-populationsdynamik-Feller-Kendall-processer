
import numpy as np
import matplotlib.pyplot as plt
import Population


#pop is population   
pop = Population.POPULATION(civil=1000, military=100, zombies=1, scientists=5)
        
events = [
    {"NAME": "ZOMBIE KILLS CIVIL",
     "W_a": lambda pop: 0.01*pop.CIVIL*pop.ZOMBIES / pop.total_population(),
     "EFFECT": pop.decrease_civil()
     },
    
    {"NAME": "CIVIL GETS INFECTED",
     "W_a": lambda pop: 0.05*pop.CIVIL*pop.ZOMBIES / pop.total_population(),
     "EFFECT": pop.civil_becomes_zombie()
     }
    ]


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
        event["EFFECT"]
    

def Kendall_Feller_Step(events):
    R = 0
    R_sums = []
    for a in events:
        w_a = a["W_a"](pop)
        R += w_a
        R_sums.append(R)

    T = np.random.exponential(scale=1/R)
    s = np.random.uniform(low=0, high=R)

    for b in range(len(events)):
        if R_sums[b] <= s < R_sums[b+1]:
            return T, events[b+1]
    return T, None             

def Kendall_Feller(events, start, stop):
    time = start
    ts = [time]
    event = None
    
    while time < stop:
        do(event)
        T, event = Kendall_Feller_Step(events)
        time += T
        ts.append(time)
        pop.update_history()
        print(event)
        if event:
            print(time, event["NAME"])

        #do(event)
        #update(events)
    plt.plot(ts,range(0, len(ts)), marker="x")
    plot_pop_history(time, pop.get_history())
    plt.show()
    
def plot_pop_history(time, pop_history):
    for history in pop_history.values():
        plt.plot(time,history, marker="x")
    plt.legend(pop_history.keys())
    plt.show()

Kendall_Feller(events, 0, 100)









