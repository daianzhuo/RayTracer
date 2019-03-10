# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:52:02 2017

@author: Anzhuo Dai
Imperial College London
"""

__doc__ = '''
Light ray position and direction representation
'''

import numpy as np
import pylab as pl

class Ray:

    def __init__(self, points = [], direction = [0,0,0], p = [], k = []):
        self.points = points
        self.pos1 = np.array(p, dtype = 'float64')
        self.points.append(self.pos1)
        
        self.direction = direction #to call direction later
        self.dir = np.array(k, dtype = 'float64')
        self.vec = self.dir / np.linalg.norm(self.dir)        
        self.direction[0] = self.vec
        
    def _clear(self):
        del self.points[:]
        self.points.append(self.pos1)
        del self.direction[:]
        self.direction.append(self.dir)
        return self.points
        
    def p(self):
        return self.points[-1]
    
    def k(self):
        return self.vec

    def append(self, p = [], k = []):
        self.pos1 = np.array(p, dtype = 'float64')
        tk = np.array(k, dtype = 'float64')
        self.vec = tk / np.linalg.norm(tk)
        self.points.append(self.pos1)
        

    def vertices(self):
        return self.points


class Optical_Element:
    
    def propagate_ray(self, ray):
        "propagate a ray through the optical element"
        intercept = self.intercept(ray)
        if intercept is None:
            return 'No valid intercept'
        else:
            ray.points.append(intercept)
            return ray.vertices()
        
class Spherical_Refraction(Optical_Element):
    __doc__ = '''
    Spherical transperant surface
    NB: NOT A SOLID OBJECT
    '''
    
    def __init__(self, z0 = 0, c = 0, n1 = 0.0, n2 = 0.0, ar = 0.0): #r = radius of circle, ar = aperture radius 
        self.zintercept = z0
        self.curvature = c
        self.ri1 = n1
        self.ri2 = n2
        self.aperture_radius = ar
        if self.curvature != 0:
            self.radius = 1/self.curvature
        
    def intercept(self, ray):
        '''calculates the first valid intercept of a ray with the surface'''
        
        if self.curvature == 0:
            d = self.zintercept - ray.p()[-1]
            l = d/(ray.k()[-1])
        
                
        elif self.radius > 0:
            rdir = ray.p() - np.array([0, 0, self.zintercept + self.radius], dtype = 'float64')  #r vector from O to P
            modrdir = np.linalg.norm(rdir)
            rdotk = np.dot(rdir, ray.k())
            if ((rdotk**2) - ((modrdir**2) - (self.radius**2))) <= 0:
                return None
            l = - rdotk - np.sqrt( (rdotk**2) - ((modrdir**2) - (self.radius**2)) )
            
        elif self.radius < 0:
            rdir = ray.p() - np.array([0, 0, self.zintercept + self.radius], dtype = 'float64')  #r vector from O to P
            modrdir = np.linalg.norm(rdir)
            rdotk = np.dot(rdir, ray.k())
            if (rdotk**2) - ((modrdir**2) - (self.radius**2)) <= 0:
                return None
            l = - rdotk + np.sqrt( (rdotk**2) - ((modrdir**2) - (self.radius**2)) ) #only difference here is the minus sign
            
        Q = ray.k() * l
        return ray.p() + Q

    def snells(self, ray, n1 = 0.0, n2 = 0.0):
        ray1_norm = np.linalg.norm(ray.k())
        n1 = self.ri1
        n2 = self.ri2
        
        
        if self.curvature == 0:
            surface_vec_nn =  np.array([0, 0, self.zintercept], dtype = 'float64')
                 
        elif self.curvature > 0:
            surface_vec_nn = np.array([0, 0, self.zintercept + self.radius] - self.intercept(ray), dtype = 'float64') #not normalised plane vector 
        
        else:
            surface_vec_nn = np.array(self.intercept(ray) - [0, 0, self.zintercept + self.radius], dtype = 'float64')
            
        surface_vec = surface_vec_nn / np.linalg.norm(surface_vec_nn)
        surface_norm = np.linalg.norm(surface_vec)
        theta1 = np.arccos(-np.dot(-surface_vec, ray.k()) / (ray1_norm * surface_norm)) #ray1_norm and surface_norm both = 1, here for integrity's sake
        if np.sin(theta1) > (n2 / n1):
            return 'None'
        theta2 = np.arcsin(n1 * np.sin(theta1) / n2)
        e = n2 / n1
        ray2 = e * ray.k() + (e * np.cos(theta2) - np.cos(theta1)) * surface_vec
        return ray2
    
    
    def propagate_ray(self, ray):
        '''start from here'''
        if self.intercept(ray) is None or np.absolute(self.intercept(ray)[0]) >= self.aperture_radius:
            ray.vec = ray.k()

        else:
            ray.points.append(self.intercept(ray))
            ray.vec = self.snells(ray) / np.linalg.norm(self.snells(ray))
            return ray.vertices()

    
class Outputplane(Optical_Element):
    
    def __init__(self, r0 = [0,0,0], n = [0,0,0]):
        if len(r0) > 3 or len(n) > 3:
            return None
        
        self.position = np.array(r0, dtype = 'float64')
        self.norm_vec = np.array(n / np.linalg.norm(n), dtype = 'float64')
        
    def intercept(self, ray):
        top = np.dot((self.position - ray.p()), self.norm_vec)
        bot = np.dot(ray.k(), self.norm_vec)
        lamb = top / bot
        if lamb is 0:
            return None
        fpoint = lamb * ray.k() + ray.p()
        return fpoint
    
class Spherical_Reflection(Optical_Element):
    
    __doc__ ='''
    Attempt at creating a class for reflective surface
    doesn't work
    '''
    
    def __init__(self, r1 = [0,0,0], n1 = [0,0,0]):
        self.position = np.array(r1, dtype = 'float64')
        self.norm_vec = np.array(n1 / np.linalg.norm(n1), dtype = 'float64')
        
    def intercept(self, ray):
        top = np.dot((self.position - ray.p()), self.norm_vec)
        bot = np.dot(ray.k(), self.norm_vec)
        lamb = top / bot
        if lamb is 0:
            return None
        fpoint = lamb * ray.k() + ray.p()
        return fpoint
        
    def reflect(self, ray):
        ray3 = ray.k() - 2 * np.dot(ray.k(), self.norm_vec) * self.norm_vec
        return ray3
        
    def propagate_ray(self,ray):
        ray.points.append(self.intercept(ray))
        ray.vec = self.reflect(ray) / np.linalg.norm(self.reflect(ray))
        return ray.vertices()
        
        
def frange(start, stop, step): #generator function for beam arrangement
     list1 = []
     i = start
     while i < stop:
         list1.append(i)
         i += step
     return list1         


b = Spherical_Refraction(z0 = 100, c = 0.03, n1 = 1.0, n2 = 1.5168, ar = 50)
b1 = Spherical_Refraction(z0 = 110, c = -0.03, n1 = 1.5168, n2 = 1.0, ar = 50)
b2 = Spherical_Refraction(z0 = 115, c = 0, n1 = 1.620, n2 = 1.0, ar = 50) #for doublet
ref = Spherical_Reflection(r1 = [0,0,200], n1 = [0,1,-1]) 
c = Outputplane(r0 = [0, 0, 120.829], n = [0, 0, -1])

#for curved surface facing left: f=176.667, right: f=153.333, dublet BK7-F2: f=180.156 (nbk7 = 1.5168, nf2 = 1.620)

'''square arrangement of beams'''
ray_spread = frange(-0.1,0.12,0.02)
ray_spread2 = ray_spread
ray_spread3 = zip(ray_spread, ray_spread2)


def bundle(beam_radius, steps, depth):
    r = np.linspace(0, beam_radius, steps) #middle entry is beam radius
    theta = (2*np.pi)*np.linspace(0, 10, depth)
    testx = []
    testy = []
    for i in r:
        for j in theta:
            testx.append(i*np.cos(j))
            testy.append(i*np.sin(j))
    testxy = zip(testx, testy)
    return list(testxy)  



ray_spread_bundle = bundle(0.1, 5, 20)



def propall(ray):
    ray._clear()
    b.propagate_ray(ray)
    b1.propagate_ray(ray)
#    ref.propagate_ray(ray)
    b2.propagate_ray(ray)
    c.propagate_ray(ray)
    return ray.vertices()


def trace():
    pl.figure('Ray trace')
    for i,j in ray_spread_bundle:
        aa = Ray(p = [i, j, 0], k = [0, 0, 10])    
        ax = []
        ay = []
        az = []
        aa._clear()
        for f in propall(aa):
            ax.append(f[0])
            ay.append(f[1])
            az.append(f[2])
            pl.plot(az,ax,color = 'r')
            pl.xlabel('z')
            pl.ylabel('y')
            pl.grid(True)
              
def finplot():
    pl.figure('Final positons')
    for i, j in ray_spread_bundle:
        aa = Ray(p = [i, j, 0], k = [0, 0, 10])  
        aa._clear()
        ax = []
        ay = []
        for g in propall(aa):
            ax.append(aa.vertices()[-1][0])
            ay.append(aa.vertices()[-1][1])
            pl.plot(ay,ax,'bo')
            pl.xlabel('x')
            pl.ylabel('y')
    rsum = 0
    for i in range(len(ax)):
        rsquared = ax[i]**2 + ay[i]**2
        rsum += rsquared
    mean = rsum / len(ax)
    fin = np.sqrt(mean)
    print ('The RMS spread is %s' %(fin))
    return fin


def initplot():
    pl.figure('Initial positions')
    for i, j in ray_spread_bundle:
        aa = Ray(p = [i, j, 0], k = [0, 0, 0])  
        aa._clear()
        ax = []
        ay = []
        for g in propall(aa):
            ax.append(aa.vertices()[0][0])
            ay.append(aa.vertices()[0][1])
            pl.plot(ay,ax,'bo')
            pl.xlabel('x')
            pl.ylabel('y')
    rsum = 0
    for i in range(len(ax)):
        rsquared = ax[i]**2 + ay[i]**2
        rsum += rsquared
    mean = rsum / len(ax)
    fin = np.sqrt(mean)
    print ('The RMS spread is %s' %(fin))
    return fin
    
#trace()
#finplot()
#initplot()


'''change configuration by switching between bundle and ray_spread3'''
'''RMS distribution finder'''
'''N.B. COMMENT OUT PLOTTING BEFORE USING THIS OR CRASHES!!!'''


x = []
y = []
for i in frange(0, 10, 0.01): #mf takes a while to run, be patient
    ray_spread_bundle = bundle(i, 5, 10)
    trace()
    finplot()
    y.append(finplot())
    initplot()
    x.append(initplot())
    
pl.figure('rms')
pl.plot(x,y, label = 'P left')
pl.xlabel('Beam radius')
pl.ylabel('RMS spread') 
pl.grid(True)
pl.legend(loc = 'upper left')



