class POPULATION:
    
    def __init__(self, civil, military, scientists, zombies = 1, resistant = 0):
        self.CIVIL = civil
        self.MILITARY = military
        self.ZOMBIES = zombies
        self.SCIENTISTS = scientists
        self.RESISTANT = resistant
        
        self.population_history = {
        "ZOMBIES":[],
        "CIVIL": [],
        "MILITARY": [], 
        "SCIENTISTS":[],
        "RESISTANT": []}
        self.update_history()
        
    def update_history(self):
        self.population_history["ZOMBIES"].append(self.ZOMBIES)
        self.population_history["CIVIL"].append(self.CIVIL)
        self.population_history["MILITARY"].append(self.MILITARY)
        self.population_history["SCIENTISTS"].append(self.SCIENTISTS)
        self.population_history["RESISTANT"].append(self.RESISTANT)
    
    def get_history(self):
        return self.population_history
    
    def decrease_civil(self):
        if self.CIVIL > 0:
            self.CIVIL -= 1
        
    def increase_civil(self):
        self.CIVIL += 1
        
    def decrease_zombie(self):
        self.ZOMBIES -= 1
        
    def increase_zombie(self):
        self.ZOMBIES += 1
        
    def decrease_military(self):
        self.MILITARY -= 1
    
    def increase_military(self):
        self.MILITARY += 1  
        
    def decrease_scientists(self):
        self.SCIENTISTS -= 1
        
    def increase_scientists(self):
        self.SCIENTISTS += 1 
        
    def decrease_resistants(self):
        self.SCIENTISTS -= 1
        
    def increase_resistants(self):
        self.SCIENTISTS += 1
        
    def civil_becomes_zombie(self):
        if self.CIVIL > 0:
            self.decrease_civil()
            self.increase_zombie() 
            
    def military_becomes_zombie(self):
        self.decrease_military()
        self.increase_zombie()
        
    def scientist_becomes_zombie(self):
        self.decrease_scientists()
        self.increase_zombie
      
    def total_population(self):
        return self.CIVIL + self.MILITARY + self.ZOMBIES + self.SCIENTISTS + self.RESISTANT