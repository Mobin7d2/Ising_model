import numpy as np
import matplotlib.pyplot as plt

#Global variables 
J_1 = 1
J_2 = 0
J_3 = 0
# Note that Size_system**2 = Number of particles in the system
Size_system = 20
# Temperature
T = 20
# Number of Monte Carlo in one round
Number_Monte_Carlo = 10


def Grid(n_x, n_y):
    #This function gets dimension of the grid as inputs
    #And gives the grid of spins (n_1 X n_2) as the output
    #And initial spin of each element is randomly chosen
    x = []
    
    for i in range(n_y):
        x.append([])
        for j in range(n_x):
            #A random number between zero and one is generated to define spin of the particle
            random_number = np.random.rand()
            #Decide if spin is up or down
            if random_number > 0.5:
                x[i].append(1)
            else:
                x[i].append(-1)
        

    return x


def Energy_two_spin(spin_1, spin_2, J):
    #This function calculates the energy of a two body system (because of their spin). 
    Energy = -J * spin_1 * spin_2
    return Energy


def Energy_with_neighbors(x_index, y_index, grid):
    #This function calculates the energy of a cell in interaction with its neighbors
    #x and y index of the cell is given as the input
    #The energy between this cell and its neighbors is shown in the output
    E_current = 0
    
    # Energy calculation for its neighbors on its right and left
    for i in [-1, 1]:
        # Applying periodic boundary condition to the x component
        if x_index + i > len(grid[0])-1:
            E_current += Energy_two_spin(grid[x_index][y_index], grid[0][y_index], J_1)
        elif x_index + i < 0:
            E_current += Energy_two_spin(grid[x_index][y_index], grid[-1][y_index], J_1)
        else:
            E_current += Energy_two_spin(grid[x_index][y_index], grid[x_index + i][y_index], J_1)
    # Energy calculation for the cell and its up and down neighbors
    for j in [-1, 1]:
        # Applying periodic boundary condition to the y component
        if y_index + j > len(grid) - 1:
            E_current += Energy_two_spin(grid[x_index][y_index], grid[x_index][0], J_1)
        elif y_index + j < 0:
            E_current += Energy_two_spin(grid[x_index][y_index], grid[x_index][-1], J_1)
        else:
            E_current += Energy_two_spin(grid[x_index][y_index], grid[x_index][y_index + j], J_1)


    # Energy calculation for its neighbors on (i+-1, j+-1).
    for i in [[-1, -1], [1, -1], [-1, 1], [1, 1]]:
        # Applying periodic boundary condition to the x component
        if x_index + i[0] > len(grid[0])-1:
            if y_index + i[1] > len(grid[1])-1:
                E_current += Energy_two_spin(-grid[x_index][y_index], grid[0][0], J_2)
            elif y_index + i[1] < 0:
                E_current += Energy_two_spin(-grid[x_index][y_index], grid[0][-1], J_2)
            else:
                E_current += Energy_two_spin(-grid[x_index][y_index], grid[0][y_index+i[1]], J_2)

        elif x_index + i[0] < 0:
            if y_index + i[1] > len(grid[1])-1:
                E_current += Energy_two_spin(-grid[x_index][y_index], grid[-1][0], J_2)
            elif y_index + i[1] < 0:
                E_current += Energy_two_spin(-grid[x_index][y_index], grid[-1][-1], J_2)
            else:
                E_current += Energy_two_spin(-grid[x_index][y_index], grid[-1][y_index+i[1]], J_2)

        else:
            if y_index + i[1] > len(grid[1])-1:
                E_current += Energy_two_spin(-grid[x_index][y_index], grid[x_index + i[0]][0], J_2)
            elif y_index + i[1] < 0:
                E_current += Energy_two_spin(-grid[x_index][y_index], grid[x_index + i[0]][-1], J_2)
            else:
                E_current += Energy_two_spin(-grid[x_index][y_index], grid[x_index + i[0]][y_index+i[1]], J_2)


    # Energy calculation for its neighbors on (i+-2, j) & (i, j+-2).
    for i in [-2, 2]:
        # Applying periodic boundary condition to the x component
        # The case when neighbor x index is one unit greater than the grid size
        if x_index + i == len(grid[0]):
            E_current += Energy_two_spin(grid[x_index][y_index], grid[0][y_index], J_3)
        elif x_index + i == len(grid[0])+1:
            # The case when neighbor x index is two units greater than the grid size
            E_current += Energy_two_spin(grid[x_index][y_index], grid[1][y_index], J_3)
        elif x_index + i == -1:
            # The case when neighbor x index is one unit smaller than the grid size
            E_current += Energy_two_spin(grid[x_index][y_index], grid[-1][y_index], J_3)
        elif x_index + i == -2:
            # The case when neighbor x index is two units smaller than the grid size
            E_current += Energy_two_spin(grid[x_index][y_index], grid[-2][y_index], J_3)
        else:
            E_current += Energy_two_spin(grid[x_index][y_index], grid[x_index + i][y_index], J_3)
    # Energy calculation for the cell and its up and down neighbors
    for j in [-2, 2]:
        # Applying periodic boundary condition to the y component
        # The case when neighbor y index is one unit greater than the grid size
        if y_index + j == len(grid):
            E_current += Energy_two_spin(grid[x_index][y_index], grid[x_index][0], J_3)
        elif y_index + j == len(grid[0])+1:
            # The case when neighbor y index is two units greater than the grid size
            E_current += Energy_two_spin(grid[x_index][y_index], grid[x_index][1], J_3)
        elif y_index + j == -1:
            # The case when neighbor y index is one unit smaller than the grid size
            E_current += Energy_two_spin(grid[x_index][y_index], grid[x_index][-1], J_3)
        elif y_index + j == -2:
            # The case when neighbor y index is two units smaller than the grid size
            E_current += Energy_two_spin(grid[x_index][y_index], grid[x_index][-2], J_3)
        else:
            E_current += Energy_two_spin(grid[x_index][y_index], grid[x_index][y_index + j], J_3)

    return E_current


def Energy_with_neighbors_flipped(x_index, y_index, grid):
    #This function hypothetically consideres the cell[x_index, y_index] to have 
    #the opposite spin, then calculates the energy of this hypothetical version
    #in interaction with its neighbors. Note that the - sign before the first
    #argument of Energy_two_spin means that the cell has been flipped.
    E_flip = 0
    
    # Energy calculation for its neighbors on its right and left
    for i in [-1, 1]:
        # Applying periodic boundary condition to the x component
        if x_index + i > len(grid[0])-1:
            E_flip += Energy_two_spin(-grid[x_index][y_index], grid[0][y_index], J_1)
        elif x_index + i < 0:
            E_flip += Energy_two_spin(-grid[x_index][y_index], grid[-1][y_index], J_1)
        else:
            E_flip += Energy_two_spin(-grid[x_index][y_index], grid[x_index + i][y_index], J_1)
    # Energy calculation for the cell and its up and down neighbors
    for j in [-1, 1]:
        # Applying periodic boundary condition to the y component
        if y_index + j > len(grid) - 1:
            E_flip += Energy_two_spin(-grid[x_index][y_index], grid[x_index][0], J_1)
        elif y_index + j < 0:
            E_flip += Energy_two_spin(-grid[x_index][y_index], grid[x_index][-1], J_1)
        else:
            E_flip += Energy_two_spin(-grid[x_index][y_index], grid[x_index][y_index + j], J_1)


    # Energy calculation for its neighbors on (i+-1, j+-1).
    for i in [[-1, -1], [1, -1], [-1, 1], [1, 1]]:
        # Applying periodic boundary condition to the x component
        if x_index + i[0] > len(grid[0])-1:
            if y_index + i[1] > len(grid[1])-1:
                E_flip += Energy_two_spin(-grid[x_index][y_index], grid[0][0], J_2)
            elif y_index + i[1] < 0:
                E_flip += Energy_two_spin(-grid[x_index][y_index], grid[0][-1], J_2)
            else:
                E_flip += Energy_two_spin(-grid[x_index][y_index], grid[0][y_index+i[1]], J_2)

        elif x_index + i[0] < 0:
            if y_index + i[1] > len(grid[1])-1:
                E_flip += Energy_two_spin(-grid[x_index][y_index], grid[-1][0], J_2)
            elif y_index + i[1] < 0:
                E_flip += Energy_two_spin(-grid[x_index][y_index], grid[-1][-1], J_2)
            else:
                E_flip += Energy_two_spin(-grid[x_index][y_index], grid[-1][y_index+i[1]], J_2)

        else:
            if y_index + i[1] > len(grid[1])-1:
                E_flip += Energy_two_spin(-grid[x_index][y_index], grid[x_index + i[0]][0], J_2)
            elif y_index + i[1] < 0:
                E_flip += Energy_two_spin(-grid[x_index][y_index], grid[x_index + i[0]][-1], J_2)
            else:
                E_flip += Energy_two_spin(-grid[x_index][y_index], grid[x_index + i[0]][y_index+i[1]], J_2)

    
       # Energy calculation for its neighbors on (i+-2, j) & (i, j+-2).
    for i in [-2, 2]:
        # Applying periodic boundary condition to the x component
        # The case when neighbor x index is one unit greater than the grid size
        if x_index + i == len(grid[0]):
            E_flip += Energy_two_spin(grid[x_index][y_index], grid[0][y_index], J_3)
        elif x_index + i == len(grid[0])+1:
            # The case when neighbor x index is two units greater than the grid size
            E_flip += Energy_two_spin(grid[x_index][y_index], grid[1][y_index], J_3)
        elif x_index + i == -1:
            # The case when neighbor x index is one unit smaller than the grid size
            E_flip += Energy_two_spin(grid[x_index][y_index], grid[-1][y_index], J_3)
        elif x_index + i == -2:
            # The case when neighbor x index is two units smaller than the grid size
            E_flip += Energy_two_spin(grid[x_index][y_index], grid[-2][y_index], J_3)
        else:
            E_flip += Energy_two_spin(grid[x_index][y_index], grid[x_index + i][y_index], J_3)
    # Energy calculation for the cell and its up and down neighbors
    for j in [-2, 2]:
        # Applying periodic boundary condition to the y component
        # The case when neighbor y index is one unit greater than the grid size
        if y_index + j == len(grid):
            E_flip += Energy_two_spin(grid[x_index][y_index], grid[x_index][0], J_3)
        elif y_index + j == len(grid[0])+1:
            # The case when neighbor y index is two units greater than the grid size
            E_flip += Energy_two_spin(grid[x_index][y_index], grid[x_index][1], J_3)
        elif y_index + j == -1:
            # The case when neighbor y index is one unit smaller than the grid size
            E_flip += Energy_two_spin(grid[x_index][y_index], grid[x_index][-1], J_3)
        elif y_index + j == -2:
            # The case when neighbor y index is two units smaller than the grid size
            E_flip += Energy_two_spin(grid[x_index][y_index], grid[x_index][-2], J_3)
        else:
            E_flip += Energy_two_spin(grid[x_index][y_index], grid[x_index][y_index + j], J_3)

    return E_flip


def Boltzmann_probability(Energy_of_state, Beta):
    #This function returns probability of the state using Boltzmann factor
    probability = np.exp(-Beta*Energy_of_state)
    return probability


def If_flip(E_current, E_flipped, x_index, y_index, grid, Beta):
    #This function decides if the cell needs to be flipped or not.
    #E_current is the energy of the cell in its current configuration.
    #E_flipped is the energy of the cell in its flipped configuration.
    Delta_E = E_current - E_flipped
    #print(Delta_E)
    if Delta_E > 0:
        #Flip the spin
        grid[x_index][y_index] *= -1
    else:
        random_number = np.random.rand()
        Probability_of_state = Boltzmann_probability(abs(Delta_E), Beta)
        if random_number < Probability_of_state:
            #Flip the spin
            #Else do not chage the spin
            grid[x_index][y_index] *= -1
            


def Monte_carlo(grid, Beta):
    #Applying Monte Carlo
    for j_row in range(len(grid)):
        for i_column in range(len(grid[0])):
            # Calculating energy of the current configuration
            Energy_current = Energy_with_neighbors(i_column, j_row, grid)
            # Calculating energy of the flipped configuration
            Energy_filipped = Energy_with_neighbors_flipped(i_column, j_row, grid)
            # Checking if the spin needs to be flipped
            If_flip(Energy_current, Energy_filipped, i_column, j_row, grid, Beta)


def Energy_of_system(grid):
    #Finding number of cells
    Number_of_spins_y = len(grid)
    Number_of_spins_x = len(grid[0])
    
    #Calculating energy of the system
    E = 0

    for y in range(Number_of_spins_y):
        for x in range(Number_of_spins_x):
            E += Energy_with_neighbors(x, y, grid)

    return E

def Magnetization(grid):
    #This function calculates magnetization of our system.
    #Finding number of cells
    Number_of_spins_y = len(grid)
    Number_of_spins_x = len(grid[0])
    
    # Magnetization
    M = 0
    # Summation over all spins
    for y in range(Number_of_spins_y):
        for x in range(Number_of_spins_x):
            M += grid[y][x]

    return M / (Number_of_spins_x* Number_of_spins_y)


def main():
    # This function calculates and graphs the Mag. Vs Time
    Beta = 1/T
    System = Grid(Size_system, Size_system)

    step = 1000
    Mag = []
    Energy = []

    #print(Energy_of_system(System))
    #print(Magnetization(System))


    for i in range(step):
        for j in range(Number_Monte_Carlo):
            Monte_carlo(System, Beta)
        #print(Energy_of_system(System))
        M = Magnetization(System)
        E = Energy_of_system(System)
        Energy.append(E/10**2)
        Mag.append(M)
        #print(M)


    # Plotting
    Time = [t+1 for t in range(step)]
    # Plotting Magnetization
    #plt.plot(Time, Mag, label="#"+str(Size_system)+"#Mont"+str(Number_Monte_Carlo))
    #plt.ylim(-1, 1)
    #plt.title("Magnetization (Temp = " + str(1/Beta) + ")")
    #plt.ylabel("Magnetization")

    # Plotting Energy
    plt.plot(Time, Energy, label="#"+str(Size_system)+"#Mont"+str(Number_Monte_Carlo))
    plt.title("Energy (Temp =" + str(1/Beta) + ")")
    plt.ylabel("Energy")
    plt.ylim(-10, 10)
    plt.xlabel("Time(step)")
    plt.legend()
    plt.show()




def main_curie():
    #This functionn plots Temp Vs Magnetization
    #Temperature
    T = [(0.5+t*0.025) for t in range(1, 130)]
    Mag = []

    for t in T:
        #Beta is equivalent to temperature which is to be updated
        Beta = 1/t
        # Let us call a new system with this new temp to eventually calculate our new Magnetiation
        System = Grid(Size_system, Size_system)
        step = 1000
        #print(Energy_of_system(System))
        #print(Magnetization(System))


        for i in range(step):
            Monte_carlo(System, Beta)
            #print(Energy_of_system(System))
            #M = Magnetization(System)
            #Mag.append(M)
            #print(M)

        M = Magnetization(System)
        Mag.append(abs(M))


    plt.scatter(T, Mag)
    plt.xlabel("Temperature")
    plt.ylabel("Magnetization")
    plt.show()

main()
#main_curie()




#T = [0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.8500000000000001, 0.875, 0.9, 0.925, 0.95, 0.9750000000000001, 1.0, 1.025, 1.05, 1.0750000000000002, 1.1, 1.125, 1.15, 1.175, 1.2000000000000002, 1.225, 1.25, 1.275, 1.3, 1.3250000000000002, 1.35, 1.375, 1.4, 1.425, 1.4500000000000002, 1.475, 1.5, 1.5250000000000001, 1.55, 1.575, 1.6, 1.625, 1.6500000000000001, 1.675, 1.7000000000000002, 1.725, 1.75, 1.7750000000000001, 1.8, 1.8250000000000002, 1.85, 1.875, 1.9000000000000001, 1.925, 1.9500000000000002, 1.975, 2.0, 2.0250000000000004, 2.05, 2.075, 2.1, 2.125, 2.1500000000000004, 2.175, 2.2, 2.225, 2.25, 2.2750000000000004, 2.3, 2.325, 2.35, 2.375, 2.4000000000000004, 2.425, 2.45, 2.475, 2.5, 2.525, 2.5500000000000003, 2.575, 2.6, 2.625, 2.65, 2.6750000000000003, 2.7, 2.725, 2.75, 2.775, 2.8000000000000003, 2.825, 2.85, 2.875, 2.9000000000000004, 2.9250000000000003, 2.95, 2.975, 3.0, 3.0250000000000004, 3.0500000000000003, 3.075, 3.1, 3.125, 3.1500000000000004, 3.1750000000000003, 3.2, 3.225, 3.25, 3.2750000000000004, 3.3000000000000003, 3.325, 3.35, 3.375, 3.4000000000000004, 3.4250000000000003, 3.45, 3.475, 3.5, 3.5250000000000004, 3.5500000000000003, 3.575, 3.6, 3.625, 3.6500000000000004, 3.6750000000000003, 3.7, 3.725, 3.75, 3.7750000000000004, 3.8000000000000003, 3.825, 3.85, 3.875, 3.9000000000000004, 3.9250000000000003, 3.95, 3.975, 4.0, 4.025, 4.050000000000001, 4.075, 4.1, 4.125, 4.15, 4.175000000000001, 4.2, 4.225, 4.25, 4.275, 4.300000000000001, 4.325, 4.35, 4.375, 4.4, 4.425000000000001, 4.45, 4.475]
#M = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9911111111111112, 1.0, 1.0, 1.0, 1.0, 0.9911111111111112, 1.0, 1.0, 1.0, 0.9822222222222222, 0.9733333333333334, 0.9911111111111112, 1.0, 1.0, 1.0, 0.9555555555555556, 1.0, 1.0, 0.9733333333333334, 0.9644444444444444, 0.9733333333333334, 0.9644444444444444, 0.9555555555555556, 0.9822222222222222, 0.9822222222222222, 0.9822222222222222, 0.9822222222222222, 0.92, 0.8755555555555555, 0.9555555555555556, 0.9555555555555556, 0.9911111111111112, 0.9288888888888889, 0.9022222222222223, 0.9466666666666667, 0.9644444444444444, 0.9555555555555556, 0.9733333333333334, 0.92, 0.8755555555555555, 0.8222222222222222, 0.8844444444444445, 0.9111111111111111, 0.9555555555555556, 0.7244444444444444, 0.38666666666666666, 0.12, 0.8844444444444445, 0.4222222222222222, 0.8133333333333334, 0.5644444444444444, 0.7422222222222222, 0.6977777777777778, 0.4222222222222222, 0.19111111111111112, 0.7066666666666667, 0.7155555555555555, 0.16444444444444445, 0.5466666666666666, 0.5911111111111111, 0.6, 0.13777777777777778, 0.06666666666666667, 0.3422222222222222, 0.39555555555555555, 0.04888888888888889, 0.5288888888888889, 0.08444444444444445, 0.21777777777777776, 0.09333333333333334, 0.04, 0.057777777777777775, 0.03111111111111111, 0.24444444444444444, 0.29777777777777775, 0.29777777777777775, 0.16444444444444445, 0.12, 0.21777777777777776, 0.1288888888888889, 0.03111111111111111, 0.13777777777777778, 0.13777777777777778, 0.06666666666666667, 0.013333333333333334, 0.12, 0.2088888888888889, 0.15555555555555556, 0.16444444444444445, 0.17333333333333334, 0.12, 0.0044444444444444444, 0.03111111111111111, 0.1288888888888889, 0.2, 0.26222222222222225, 0.30666666666666664, 0.15555555555555556, 0.38666666666666666, 0.17333333333333334, 0.022222222222222223, 0.10222222222222223, 0.15555555555555556, 0.04888888888888889, 0.15555555555555556, 0.13777777777777778, 0.41333333333333333, 0.28888888888888886, 0.06666666666666667, 0.04, 0.057777777777777775, 0.10222222222222223, 0.03111111111111111, 0.013333333333333334, 0.057777777777777775, 0.057777777777777775, 0.10222222222222223, 0.04888888888888889, 0.04888888888888889, 0.0044444444444444444, 0.26222222222222225, 0.14666666666666667, 0.07555555555555556, 0.03111111111111111, 0.04, 0.08444444444444445, 0.03111111111111111, 0.28888888888888886, 0.14666666666666667, 0.1111111111111111, 0.0044444444444444444, 0.08444444444444445, 0.04, 0.06666666666666667, 0.013333333333333334, 0.13777777777777778, 0.06666666666666667]
#print(T.index(2.05))
#print(T[61])
#print(M[61:70])
#print(M.index(0.12))
#print(T[68])