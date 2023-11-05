
import pygame
import math
import numpy as np
import matplotlib.cm as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs

#itialize cmap

norm = mpl.colors.Normalize(vmin=0, vmax=0.0015)
cmap = cm.viridis
m = cm.ScalarMappable(norm=norm, cmap=cmap)

#Initial math to prepare the first 10 energy eigenstates of a potential V

N=50                                                                                  #2D schrodinger equation grid size
dt=25                                                                                  #time step size
X, Y = np.meshgrid(np.linspace(0,N-1,N,dtype=float),np.linspace(0,N-1,N,dtype=float))   #create meshgrid

#Define a potential
def potential(x,y):
    return x*0

V=potential(X,Y)

#Creating matrices for numerical simulation
#When you discretize the Schrodinger equation in 2D you get a T term
#Involves the time derivatives and a U term that invovles the potential
#Making NxN sparse diagonal matrix with -2 on main diagonals and 1s on the first off diagonals
D = sparse.spdiags(np.array([np.ones([N]), -2*np.ones([N]),np.ones([N])]), np.array([-1,0,1]),N,N)
#Turning 1D analogue to 2D analogue making a N^2 x N^2 matrix
T = -1/2 * sparse.kronsum(D,D)
#Potential term
U = sparse.diags(V.reshape(N*N),(0))
#Total Matrix
H=U+T
#The energy states are the eigenvalues of this matrix

#Get energy eigenstates using scipy.sparse.linalg
energys, states = eigsh(H, k=10, which='SM')
#Take transpose of states which allows us to index individual eigenstates
states=states.T


#Initialize Game

pygame.init()

# Font that is used to render the text
font20 = pygame.font.Font('freesansbold.ttf', 20)

# RGB values of standard colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Basic parameters of the screen
WIDTH, HEIGHT = 950, 950
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

clock = pygame.time.Clock()
FPS = 30


class Eigenstate:
    def __init__(self,n):
        self.num=n
        self.state=states[n]
        self.energy=energys[n]
    def update(self):
        self.state=np.exp(1j*self.energy*dt)*self.state               #time evolution

class Wavefunction:
    def __init__(self, coefs):
        self.coefs=np.array(coefs)/np.linalg.norm(np.array(coefs))
        self.numstates=len(coefs)
        self.eigenstates=[]
        for i in range(self.numstates):
            self.eigenstates.append(Eigenstate(i))
    def update(self):
        for eigenstate in self.eigenstates:
            eigenstate.update()
        ##add update coefficients from user input
    def totalstate(self):       #adds up superposition of states
        total=0+0j
        for i in range(self.numstates):
            total=total+self.coefs[i]*self.eigenstates[i].state
        return total
    def pdensity(self):
        total=self.totalstate()
        return (total * np.conjugate(total)).real
    def plot(self):
        fig=plt.figure()
        fig.figsize=(10,10)
        levels = np.linspace(-0.0003,0.0003,21)
        fig=plt.contourf(X, Y, self.pdensity().reshape(N,N),levels=levels,cmap=cmap)
        plt.colorbar(fig)
    def collapse(self):
        index=int(np.random.choice(np.linspace(0,N**2-1,N**2),1,p=self.pdensity())[0])
        #location = np.zeros(N**2)
        #location[index]=1
        return index
        

class Grid:
    def __init__(self, wavefunction):
        self.wavefunction=wavefunction
        self.blocksize=HEIGHT/N
        self.collapsed=False
        self.safe=False
        self.safe2=False
        self.index=0
        self.score=0
        self.tries=5
        #self.scoreMatrix=0
    def display(self):
        pdensity=self.wavefunction.pdensity().reshape(N,N)
        for x in range(0, N-1):
            for y in range(0, N-1):
                color=m.to_rgba(pdensity[x,y])
                newcolor=(math.floor(color[0]*255),math.floor(color[1]*255),math.floor(color[2]*255),color[3])
                rect = pygame.Rect(x*self.blocksize, y*self.blocksize, x*self.blocksize, y*self.blocksize)
                pygame.draw.rect(SCREEN, newcolor, rect)
        if self.collapsed:
            if self.safe:
                self.index=self.wavefunction.collapse()
                self.safe=False
            R=10
            y=int(self.index/N)
            x=self.index%N
            color=RED
            pygame.draw.circle(SCREEN, color, (x*self.blocksize,y*self.blocksize), R)
        #display score
        text = font20.render("SCORE: " + str(self.score), True, WHITE)
        textRect = text.get_rect()
        textRect.center = ((WIDTH-1)/2, 75)
        #display Tries
        text5 = font20.render("Tries: " + str(self.tries), True, WHITE)
        textRect5 = text5.get_rect()
        textRect5.center = ((WIDTH-1)/2, 125)
        #display Range1
        text2 = font20.render("1 Point!", True, WHITE)
        textRect2 = text2.get_rect()
        textRect2.center = ((WIDTH-1)/2, 200)
        #display Range2
        text3 = font20.render("10 Points!", True, WHITE)
        textRect3 = text3.get_rect()
        textRect3.center = ((WIDTH-1)/2, 350)
        #display Range3
        text4 = font20.render("25 Points!", True, WHITE)
        textRect4 = text4.get_rect()
        textRect4.center = ((WIDTH-1)/2, 450)

        SCREEN.blit(text, textRect)
        SCREEN.blit(text2, textRect2)
        SCREEN.blit(text3, textRect3)
        SCREEN.blit(text4, textRect4)
        SCREEN.blit(text5, textRect5)
    def collapse(self):
        self.collapsed= not self.collapsed
        self.safe = self.collapsed
        self.safe2 = self.collapsed
        if self.collapsed:
            self.tries-=1
    def update(self):
        if not self.collapsed:
            self.wavefunction.update()
        else:
            if self.safe2:
                r1=50
                r2=150
                r3=300
                y=HEIGHT/2-int(self.index/N)*self.blocksize
                x=WIDTH/2-self.index%N*self.blocksize
                r=y**2+x**2
                if r < r1**2:
                    self.score+=25
                if r>r1**2 and r < r2**2:
                    self.score+=10
                if r>r2**2 and r < r3**2:
                    self.score+=1
                self.safe2 = False
        if self.tries == 0:
            pygame.time.wait(500)
            SCREEN.fill(BLACK)
            text = font20.render("GAME OVER! YOU SCORED: " + str(self.score), True, WHITE)
            textRect = text.get_rect()
            textRect.center = ((WIDTH-1)/2, 450)
            SCREEN.blit(text, textRect)
            
                #if r > r3**2:
                #    self.score+=1
                    
                
                #self.score=self.score+self.scoreMatrix[int(self.index/N),self.index%N]
            
class Target:
    def __init__(self):
        self.middle=(((HEIGHT-1)/2),((WIDTH-1)/2))
    def display(self):
        #pygame.draw.circle(SCREEN, BLACK, self.middle, 600, 1)
        pygame.draw.circle(SCREEN, WHITE, self.middle, 300, 1)
        pygame.draw.circle(SCREEN, WHITE, self.middle, 150, 1)
        pygame.draw.circle(SCREEN, WHITE, self.middle, 50, 1)
        #for i in range(20):
        #    pygame.draw.line(SCREEN, BLACK, self.middle, (WIDTH*math.cos(2*math.pi*i/20), WIDTH*math.sin(2*math.pi*i/20)))
        

#wave=Wavefunction([0.5])
#wave.plot()
        

def main():
    running = True

    # Defining the objects
    coefs=np.random.uniform(low=0.0, high=1.0, size=9)
    grid=Grid(Wavefunction(coefs))
    target=Target()

    while running:
        clock.tick(FPS)
        #SCREEN.fill(BLACK)
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    grid.collapse()

        # Displaying the objects on the screen
        grid.display()
        target.display()

        # Updating the objects
        grid.update()

        pygame.display.update()

main()
        
        

