class POPULATION:
    
    def __init__(self, civil, military, zombies, scientists):
        self.CIVIL = civil
        self.MILITARY = military
        self.ZOMBIES = zombies
        self.SCIENTISTS = scientists
        
    def decrease_civil(self):
        self.CIVIL -= 1
        
    def increase_civil(self):
        self.CIVIL += 1
        
    def civil_becomes_zombie(self):
        self.decrease_civil(self)
        self.increase_zombie(self)
        
    def total_population(self):
        return self.CIVIL + self.MILITARY + self.ZOMBIES + self.SCIENTISTS