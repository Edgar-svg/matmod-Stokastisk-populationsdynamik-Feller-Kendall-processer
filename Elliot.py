
import numpy as np
import matplotlib as plt

time = 100 #Number of time intervals

class POPULATION:
    
    def __init__(self):
        self.CIVIL = 1000
        self.MILITARY = 100
        self.ZOMBIES = 1
        self.SCIENTISTS = 5
    
    def total_population(self):
        return self.CIVIL + self.MILITARY + self.ZOMBIES + self.SCIENTISTS

    
population = POPULATION()
  
zombie_kills_civil = {"ZOMBIE KILLS CIVIL": lambda pop: 0.01*pop.CIVIL*pop.ZOMBIES / pop.total_population()}
        
events = [
    {"NAME": "ZOMBIE KILLS CIVIL",
     "W_a": lambda pop: 0.01*pop.CIVIL*pop.ZOMBIES / pop.total_population(),
     "EFFECT": lambda pop: pop.CIVIL -= 1
     },
    {"NAME": "TEST",
     "W_a": lambda pop: 0.05*pop.CIVIL*pop.ZOMBIES / pop.total_population(),
     "EFFECT": lambda pop: pop.CIVIL += 1
     }
    ]
#pop is population
#%%events = {
    "ZOMBIE KILLS CIVIL": lambda pop: 0.01*pop.CIVIL*pop.ZOMBIES / pop.total_population(),
    "MILITARY KILLS ZOMBIE": lambda pop: 0.5*pop.CIVIL*pop.ZOMBIES / pop.total_population(),
    "ZOMBIE INFECTS CIVIL": lambda pop: 0.03*pop.CIVIL*pop.ZOMBIES / pop.total_population()
      
}       
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
    pass

def update(events):
    pass

def Kendall_Feller_Step(events):
    R = 0
    R_sums = []
    for a in events.keys():
        w_a = events[a](population)
        R += w_a
        R_sums.append(R)

    T = np.random.exponential(scale=1/R)
    s = np.random.uniform(low=0, high=R)

    b = -1
    for a in events.keys():
        b+=1
        if R_sums[b] < s < R_sums[b+1]:
            return T, a
    return T, 'error'           

def Kendall_Feller(events, start, stop):
    time = start
    ts = [time]
    while time < stop:
        T, event = Kendall_Feller_Step(events)
        time += T
        ts.append(time)
        print(time, event)

        #do(event)
        #update(events)
    plt.plot(ts,range(0, len(ts)), marker="x")
    plt.show()

Kendall_Feller(events, 0, 100)









