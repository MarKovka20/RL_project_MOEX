import numpy as np


class Environment:
    
    def __init__(self, filename, levels = 41, parameters = 80, commission = 0.0005, multiplier = 10.0):
        
        self.parameters = parameters
        self.commission = commission
        self.multiplier = multiplier
        
        self.level = 0
        self.index = 0
        self.position = 0.0

        self.sizes = []
        data = []

        fin = open(filename, "r")
        for i in range (levels):
            
            size = int(next(fin).split()[2])
            self.sizes.append(size)
            batch = []
            
            for j in range(size):
                batch.append([float(x) for x in (next(fin).split(","))[0 : parameters]])  
                
            data.append(np.array(batch))
            
        self.data = np.array(data, dtype = object)
        fin.close()
    
    def reset(self):
        
        self.level = 0
        self.index = 0
        self.position = 0.0
        
        return self.data[self.level][self.index]
    
    def step(self, required_state, transaction = 100.0):

        done = False
        
        #comission computation
        if(required_state == 0): # Long
            if(self.position <= 0.0 ):
                reward = -(abs(self.position) + transaction) * self.commission
                self.position = transaction
            else:
                reward = 0.0
        elif(required_state == 1): # Short
            if(self.position >= 0.0):
                reward = -(abs(self.position) + transaction) * self.commission
                self.position = -transaction
            else:
                reward = 0.0
        else:
            raise Exception("invalid argument")
        
        self.index += 1
        
        if(self.index == self.sizes[self.level]):  
            self.level += 1
            self.index = 0 
        
        state = self.data[self.level][self.index]
        price_deviation = (state[self.parameters - 1] + state[self.parameters - 2]) / self.multiplier
        reward += price_deviation * self.position
        self.position += self.position * price_deviation
        if(self.level == self.data.shape[0] - 1 and self.index == self.sizes[self.level] - 1):
            done = True
                
        return state, reward, done    
    


if __name__ == '__main__':

    env = Environment('train.data')

    print(env.reset())
    state, reward, done = env.step(1)
    print(state, reward, done)