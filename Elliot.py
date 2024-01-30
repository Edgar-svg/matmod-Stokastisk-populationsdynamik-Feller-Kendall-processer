
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



population.CIVIL = 100
print(population.CIVIL)  
        
        
beta = 0.1

#pop is population
events = {
    "ZOMBIE KILLS CIVIL": lambda pop: 0.1*pop.CIVIL*pop.ZOMBIES / pop.total_population(),
    "MILITARY KILLS ZOMBIE": lambda pop: 0.5*pop.CIVIL*pop.ZOMBIES / pop.total_population(),
    "ZOMBIE INFECTS CIVIL": lambda pop: 0.03*pop.CIVIL*pop.ZOMBIES / pop.total_population()
      
}       

print(events["CIVIL DIES"](population))

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

def Kendall_Feller_Step(events):
    
    pass



def Kendall_Feller(events, time, )
    pass









