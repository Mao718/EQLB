from keras.utils import np_utils
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from datetime import datetime
from sklearn.metrics.cluster import normalized_mutual_info_score
class Sigma:
	
	def __init__(self, input_size, dimensions, adaptation_rate=0.01, noise=False):
		self.name = "Sigma"
		self.organized= False
		self.space_size= 10
		self.dimensions= dimensions
		self.input_size= input_size
		np.random.seed(50)
		self.syncmap= np.random.rand(input_size,dimensions)
		self.velocity= np.zeros((input_size,dimensions))
		self.synapses_matrix = np.zeros([input_size, input_size])
		self.adaptation_rate= adaptation_rate
		#self.syncmap= np.random.rand(dimensions, input_size)
		
		self.ims = []
		self.fps = 0
		
		# self.anim = True
		# if self.anim == True:
		#	  self.fig = plt.figure()
		#	  self.ax = self.fig.add_subplot(111, projection='3d')
		
		self.syncmap_history = []
		
		self.noise_b = 0
		if noise is True:
			self.noise_b = 1
			self.name = "SyncMap with Noise"
	
	def inputGeneral(self, x):
		plus= x > 0.1
		minus = ~ plus

		sequence_size = x.shape[0]
		#print(sequence_size, "asfasdfasdfasd")
		for i in range(sequence_size):
			
			vplus= plus[i,:]
			vminus= minus[i,:]
			plus_mass = vplus.sum()
			minus_mass = vminus.sum()

			#print(plus_mass)
			#print(minus_mass)
			
			if plus_mass <= 1:
				continue
			
			if minus_mass <= 1:
				continue

			#print("vplus")
			#print(vplus)
			
			center_plus= np.dot(vplus,self.syncmap)/plus_mass
			center_minus= np.dot(vminus,self.syncmap)/minus_mass
		
			#print(self.syncmap.shape)
			#exit()
			dist_plus= distance.cdist(center_plus[None,:], self.syncmap, 'euclidean')
			dist_minus= distance.cdist(center_minus[None,:], self.syncmap, 'euclidean')
			dist_plus= np.transpose(dist_plus)
			dist_minus= np.transpose(dist_minus)
			
			#update_plus= vplus[:,np.newaxis]*((center_plus - self.syncmap)/dist_plus + (self.syncmap - center_minus)/dist_minus)
			#update_minus= vminus[:,np.newaxis]*((center_minus -self.syncmap)/dist_minus + (self.syncmap - center_plus)/dist_plus)
			update_plus= vplus[:,np.newaxis]*(3*(center_plus - self.syncmap)/dist_plus - (self.syncmap - center_minus)/dist_minus)
			update_minus= vminus[:,np.newaxis]*(3*(center_minus -self.syncmap)/dist_minus - (self.syncmap - center_plus)/dist_plus)
			
			#damping force
			self.velocity*= 0.999
			#self.adaptation_rate*=0.999999

			#noise = np.random.normal(0, 0.01, self.syncmap.shape)
			#noise= np.amax(self.syncmap)*np.random.rand(self.input_size, self.dimensions)
			
			#self.syncmap= self.space_size*self.syncmap/maximum
			update= update_plus - update_minus #+ noise
			self.velocity += update
			self.syncmap+= self.adaptation_rate*self.velocity
			#self.syncmap+= self.adaptation_rate*update 

			#+ noise*self.noise_b
			# self.noise_b *= 0.99
		
			maximum=self.syncmap.max()
			self.syncmap= self.space_size*self.syncmap/maximum
			#print("max",self.syncmap.max())
			
			# self.syncmap = self.syncmap - self.syncmap.mean(axis=0)
			# self.syncmap = self.syncmap / np.abs(self.syncmap).max(axis=0)
			
			# if self.anim == True:
			#	  if self.fps >= 100:
			#		  img = self.ax.scatter(self.syncmap[:,0], self.syncmap[:,1], self.syncmap[:,2], color="blue")
			#		  self.ims.append([img])
			#		  self.fps = 0
			#	  else:
			#		  self.fps += 1
			
			#self.syncmap_history.append(self.syncmap)

	def input(self, x):
	
		self.inputGeneral(x)

		return self.syncmap
	
		print(x.shape)
		plus= x > 0.1
		minus = ~ plus
#	 print(plus)
#	 print(minus)
		
#	 print(plus.shape)
#	 print(type(plus))

#	 print(x.shape)
#	 print("in",x[1,:])
#	 print("map",self.syncmap)
			
		
		sequence_size = x.shape[0]
		for i in range(sequence_size):
			vplus= plus[i,:]
			vminus= minus[i,:]
			plus_mass = vplus.sum()
			minus_mass = vminus.sum()
			#print(self.syncmap)
			#print("plus",vplus)
			if plus_mass <= 1:
				continue
			
			if minus_mass <= 1:
				continue

			#if plus_mass > 0:
			center_plus= np.dot(vplus,self.syncmap)/plus_mass
			#else:
			#	 center_plus= np.dot(vplus,self.syncmap)

			#print(center_plus)
			#exit()
			#if minus_mass > 0:
			center_minus= np.dot(vminus,self.syncmap)/minus_mass
			#else:
			#	 center_minus= np.dot(vminus,self.syncmap)

			
			#print("mass", minus_mass)
			#print(center_plus)
			#print("minus",vminus)
			#print(center_minus/minus_mass)
			#print(self.syncmap)
			#exit()

			#print(vplus)
			#print(self.syncmap.shape)
			#a= np.matmul(np.transpose(vplus),self.syncmap)
			#a= vplus.dot(self.syncmap)
			#a= (vplus*self.syncmap.transpose()).transpose()
			#update_plus= vplus[:,np.newaxis]*self.syncmap
		#	 update_plus= vplus[:,np.newaxis]*(center_plus -center_minus)*plus_mass
			update_plus= vplus[:,np.newaxis]*(center_plus -center_minus)
		#	 update_plus= vplus[:,np.newaxis]*(center_plus -center_minus)/plus_mass
			#update_plus= vplus[:,np.newaxis]*(center_plus -self.syncmap)
		#	 update_minus= vminus[:,np.newaxis]*(center_minus -center_plus)*minus_mass
			update_minus= vminus[:,np.newaxis]*(center_minus -center_plus)
		#	 update_minus= vminus[:,np.newaxis]*(center_minus -center_plus)/minus_mass
			#update_minus= vminus[:,np.newaxis]*(center_minus -self.syncmap)
			#print(self.syncmap)
			#print(center_plus)
			#print(center_plus - self.syncmap)
			#update_minus= vminus[:,np.newaxis]*self.syncmap
			
			#self.plot()

			#ax.scatter(center_plus[0], center_plus[1])
			#ax.scatter(center_minus[0], center_minus[1])
		
			#plt.show()
			
			update= update_plus + update_minus
			self.syncmap+= self.adaptation_rate*update
		
			maximum=self.syncmap.max()
			self.syncmap= self.space_size*self.syncmap/maximum

		
	def organize(self, eps=0.5):
	
		self.organized= True
		self.labels= DBSCAN(eps=eps, min_samples=2).fit_predict(self.syncmap)

		return self.labels
	
	def hierarchical_organize(self, n=2):
	
		self.organized= True
		self.labels= AgglomerativeClustering(distance_threshold=None, n_clusters=n).fit_predict(self.syncmap)

		return self.labels

	def activate(self, x):
		'''
		Return the label of the index with maximum input value
		'''

		if self.organized == False:
			print("Activating a non-organized SyncMap")
			return
		
		#maximum output
		max_index= np.argmax(x)

		return self.labels[max_index]

	def plotSequence(self, input_sequence, input_class,filename="plot.png"):

		input_sequence= input_sequence[500:1000]
		input_class= input_class[500:1000]

		a= np.asarray(input_class)
		t = [i for i,value in enumerate(a)]
		c= [self.activate(x) for x in input_sequence] 
		

		plt.plot(t, a, '-g')
		plt.plot(t, c, '-.k')
		#plt.ylim([-0.01,1.2])


		plt.savefig(filename,quality=1, dpi=300)
		plt.show()
		plt.close()
	

	# def plot(self, label, color=None, save = False, filename= "plot_map.png"):

	#	  if color is None:
	#		  color= label
	
	#	  print(self.syncmap)
	#	  #print(self.syncmap)
	#	  #print(self.syncmap[:,0])
	#	  #print(self.syncmap[:,1])
	#	  if self.dimensions == 2:
	#		  #print(type(color))
	#		  #print(color.shape)
	#		  ax= plt.scatter(self.syncmap[:,0],self.syncmap[:,1], c=color)
		
	#	  if self.dimensions == 3:
	#		  fig = plt.figure()
	#		  ax = plt.axes(projection='3d')

	#		  ax.scatter3D(self.syncmap[:,0],self.syncmap[:,1], self.syncmap[:,2], c=color)
	#		  #ax.plot3D(self.syncmap[:,0],self.syncmap[:,1], self.syncmap[:,2])
	
	#	  if save == True:
	#		  plt.savefig(filename)
	
	#	  plt.show()
	#	  plt.close()
	
	def plot(self, color=None, save = False, filename= "plot_map.png"):

		if color is None:
			color= self.labels
			
		#print(self.syncmap)
		#print(self.syncmap[:,0])
		#print(self.syncmap[:,1])
		if self.dimensions == 2:
			#print(type(color))
			#print(color.shape)
			ax= plt.scatter(self.syncmap[:,0],self.syncmap[:,1], c=color)
			
		if self.dimensions == 3:
			fig = plt.figure()
			ax = plt.axes(projection='3d')

			ax.scatter3D(self.syncmap[:,0],self.syncmap[:,1], self.syncmap[:,2], c=color);
			#ax.plot3D(self.syncmap[:,0],self.syncmap[:,1], self.syncmap[:,2])
		
		if save == True:
			plt.savefig(filename)
		
		plt.show()
		plt.close()
	
	def plot_w_line(self, label, color=None, save = False, filename= "plot_map.png"):

		if color is None:
			color= label
		
		print(self.syncmap)
		#print(self.syncmap)
		#print(self.syncmap[:,0])
		#print(self.syncmap[:,1])
		if self.dimensions == 2:
			#print(type(color))
			#print(color.shape)
			ax= plt.scatter(self.syncmap[:,0],self.syncmap[:,1], c=color)
			
		if self.dimensions == 3:
			fig = plt.figure()
			ax = plt.axes(projection='3d')

			ax.scatter3D(self.syncmap[:,0],self.syncmap[:,1], self.syncmap[:,2], c=color)
		
		if save == True:
			plt.savefig(filename)
		
		plt.show()
		plt.close()
	
	def TP_matrix(self, x):
	
		if len(x) == 0:
			return
		
		output_size = x.shape[1]
		Dtable = np.zeros([output_size, output_size])
		TPMatrix = np.zeros([output_size, output_size])
		
		prev_state = np.argmax(x[0])
		state = None
		for i in x[1:]:
			if np.max(i) != 1:
				continue
			state = np.argmax(i)
			Dtable[prev_state][state]+= 1		 
			prev_state = state
			
		for i, j in enumerate(Dtable):
			state_total = np.sum(j)
			TPMatrix[i] = Dtable[i]/state_total
		
		self.TPMatrix = TPMatrix
		return TPMatrix

	def save(self, filename):
		"""save class as self.name.txt"""
		file = open(filename+'.txt','w')
		file.write(cPickle.dumps(self.__dict__))
		file.close()

	def load(self, filename):
		"""try load self.name.txt"""
		file = open(filename+'.txt','r')
		dataPickle = file.read()
		file.close()

		self.__dict__ = cPickle.loads(dataPickle)
	
	def save_weight(self, path):
		np.save(path, self.syncmap)
	
	def animation_init(self):
		return self.img1, self.anos
	
	def animation_update(self, n):	  
		lab = DBSCAN(eps=3, min_samples=2).fit_predict(self.syncmap_history[n])
		if self.dimensions == 2:
			self.img1.set_offsets(self.syncmap_history[n])
		if self.dimensions == 3:
			self.img1._offsets3d = (self.syncmap_history[n,:,0], self.syncmap_history[n,:,1], self.syncmap_history[n,:,2])
		self.img1.set_array(lab)
		for i, txt in enumerate(self.true_labels):
			self.anos[i].set_x(self.syncmap_history[n,i,0])
			self.anos[i].set_y(self.syncmap_history[n,i,1])
		return self.img1, self.anos
	
	def plot_animation(self, name, true_labels=None):
		fig = plt.figure()
		if self.dimensions == 2:
			self.ax = fig.add_subplot()
		if self.dimensions ==3:
			self.ax = fig.add_subplot(111, projection='3d')
		
		self.syncmap_history = np.array(self.syncmap_history[::50])
		
		self.true_labels = true_labels
		self.anos = []
		for i, txt in enumerate(self.true_labels):
			self.anos.append(self.ax.annotate(txt, (self.syncmap_history[0,i,0], self.syncmap_history[0,i,1])))
		
		ax_lim_max = np.max(self.syncmap_history)
		ax_lim_max = ax_lim_max + ax_lim_max * 0.1
		ax_lim_min = np.min(self.syncmap_history)
		ax_lim_min = ax_lim_min + ax_lim_min * 0.1

		self.ax.set_xlim(ax_lim_min, ax_lim_max)
		self.ax.set_ylim(ax_lim_min, ax_lim_max)
		if self.dimensions == 2:
			self.img1 = self.ax.scatter([],[], cmap="rgb")
		if self.dimensions ==3:
			self.ax.set_zlim(ax_lim_min, ax_lim_max)
			self.img1 = self.ax.scatter3D([],[],[], cmap="rgb")
		
		print("Generating animation...")
		ani = animation.FuncAnimation(fig, self.animation_update, frames=len(self.syncmap_history), init_func=self.animation_init, interval=30, blit=False)
		print("Generating animation completed!")
		print("Saving animation as MP4")
		#animation_path =  "movies_output/movie_" + name + "_" + self.name + datetime.now().strftime('_%d%m%Y_%H%M%S') + ".mp4"
		animation_path =  "movie.mp4"
		ani.save(animation_path)
		print("Saving animation as MP4 completed!")
		
		self.syncmap_history = []
	def evaluation(self,true_label):
		label=self.organize()
		return normalized_mutual_info_score(label,true_label)