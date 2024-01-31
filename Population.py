class POPULATION:
    population_history = {
        "ZOMBIES":[],
        "CIVIL": [],
        "MILITARY": [], 
        "SCIENTISTS":[]}
    
    def __init__(self, civil, military, zombies, scientists):
        self.CIVIL = civil
        self.MILITARY = military
        self.ZOMBIES = zombies
        self.SCIENTISTS = scientists
        
    def update_history(self):
        self.population_history["ZOMBIES"].append(self.ZOMBIES)
        self.population_history["CIVIL"].append(self.CIVIL)
        self.population_history["MILITARY"].append(self.MILITARY)
        self.population_history["SCIENTISTS"].append(self.SCIENTISTS)
    
    def get_history(self):
        return self.population_history
    
    def decrease_civil(self):
        self.CIVIL -= 1
        
    def increase_civil(self):
        self.CIVIL += 1
        
    def civil_becomes_zombie(self):
        self.decrease_civil(self)
        self.increase_zombie(self)
        
    def total_population(self):
        return self.CIVIL + self.MILITARY + self.ZOMBIES + self.SCIENTISTS