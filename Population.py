class POPULATION:
    
    def __init__(self, civil, military, scientists, zombies = 1, resistant = 0, vaccine = 1):
        self.CIVIL = civil
        self.MILITARY = military
        self.ZOMBIES = zombies
        self.SCIENTISTS = scientists
        self.RESISTANT = resistant
        self.VACCINE = vaccine
        
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
        
            self.CIVIL -= 1
        
    def increase_civil(self):
        self.CIVIL += 1
        
    def decrease_zombie(self):
        if self.MILITARY > 0:
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
        self.RESISTANT -= 1
        
    def increase_resistants(self):
        self.RESISTANT += 1
        
    def civil_becomes_zombie(self):
            self.decrease_civil()
            self.increase_zombie() 
            
    def military_becomes_zombie(self):
        self.decrease_military()
        self.increase_zombie()
        
    def scientist_becomes_zombie(self):
        self.decrease_scientists()
        self.increase_zombie
        
    def is_vaccine_invented(self):
        return self.VACCINE
    
    def invent_vaccine(self):
        self.VACCINE = 0
        
    def civil_becomes_resistant(self):
        self.increase_resistants()
        self.decrease_civil()
        
    def total_population(self):
        return self.CIVIL + self.MILITARY + self.ZOMBIES + self.SCIENTISTS + self.RESISTANT