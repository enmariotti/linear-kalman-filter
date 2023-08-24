import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter(object):
	def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

		if(F is None or H is None):
			raise ValueError('Setear matrices del sistema.')

		# Matrices del sistema y matriz de medicion
		self.F = F
		self.B = B
		self.H = H
		self.n = F.shape[0]
		self.m = H.shape[0]

		# Matrices de covarianza
		self.Q = Q
		self.R = R
		self.R_u = R

		# Ganancias del filtro
		self.K = np.zeros((self.n, self.m))

		# Matrices de covarianza del filtro y estimacion
		self.P = np.eye(self.n) if P is None else P
		self.x = np.zeros((self.n, 1)) if x0 is None else x0

		# Inicializo rotaciones
		cos_theta = np.cos(self.x[-1])[0]
		sin_theta = np.sin(self.x[-1])[0]
		self.rot = np.array([[cos_theta, sin_theta, 0], [cos_theta, sin_theta, 0],[sin_theta, cos_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])

	def predict(self, u):
		# Actualizo rotaciones
		cos_theta = np.cos(self.x[-1])[0]
		sin_theta = np.sin(self.x[-1])[0]
		self.rot = np.array([[cos_theta, sin_theta, 0], [cos_theta, sin_theta, 0],[sin_theta, cos_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])

		# Rotar matrices
		B = np.multiply(self.B, self.rot)
		Q = np.dot(np.dot(self.B, self.Q), self.B.T)
		
		# Propagacion
		self.x = np.dot(self.F, self.x) + np.dot(B, u) # TODO: "u" esta en los ejes del vehiculo
		self.P = np.dot(np.dot(self.F, self.P), self.F.T) + Q

		return self.x

	def update(self, z, update = 'full', loss = None):
		# Opciones de update
		updateOptions = ['full', 'loss']
		if (update not in updateOptions):
			raise ValueError('Opcion de update no valida.')
		if (update == 'loss' and loss is None):
			raise ValueError('Especificar los indices de los sensores a perder.')

		 # Covarianzas variables p/ simular falla	
		if update == 'full':
			S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
		if update == 'loss':
			self.R_u = self.R.copy()
			self.R_u[loss] = self.R_u[loss]*10e6
			S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R_u

		# Ganacia de Kalman
		self.K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
		
		# Componentes de update
		y = (z - np.dot(self.H, self.x))

		# Update
		self.x = self.x + np.dot(self.K, y)
		self.P = np.dot(np.eye(self.n) - np.dot(self.K, self.H), self.P)

	def run(self, U, Z, update = 'full', loss = None, updateTime = 1):
		# Estimaciones
		estimations = np.zeros(Z.shape)
		covariance = np.zeros((self.P.shape[0],self.P.shape[1],Z.shape[1]))
		kalmanGains = np.zeros((self.K.shape[0],self.K.shape[1],Z.shape[1]))
		for i, (u, z) in enumerate(zip(U.T, Z.T)):
			estimations[:,i] = np.dot(self.H,  self.predict(u.reshape(-1,1))).reshape((-1,))
			covariance[:,:,i] = self.P
			kalmanGains[:,:,i] = self.K
			if not(i%updateTime):
				self.update(z.reshape(-1,1), update = update, loss = loss)

		return estimations, covariance, kalmanGains	
	
def createKF(x0 = None, sigma = (1,1,1,1)):

	# Varianzas
	sigmaGPS, sigmaMag, sigmaAcc, sigmaGyro  = sigma

	# Condiciones iniciales
	x = np.zeros((5, 1)) if x0 is None else x0
	
	# Modelo del sistema
	F = np.array([[1.0, dt_IMU, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, dt_IMU, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
	B = np.array([[0.5*(dt_IMU**2), -0.5*(dt_IMU**2), 0], [dt_IMU, -dt_IMU, 0], [0.5*(dt_IMU**2), 0.5*(dt_IMU**2), 0], [dt_IMU, dt_IMU, 0], [0, 0, dt_IMU]])
	
	# Matriz de Medicion
	H = np.array([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]])

	# Matrices de covarianza
	Q = np.array([[sigmaAcc**2, 0, 0], [0, sigmaAcc**2, 0], [0, 0, sigmaGyro**2]])
	R = np.array([[sigmaGPS**2, 0, 0], [0, sigmaGPS**2, 0], [0, 0, sigmaMag**2]])

	# Inicializo filtro en el origen
	kf = KalmanFilter(F = F, B = B, H = H, Q = Q, R = R, x0 = x)

	return kf

def linear_data(T = 100, speed = 1, updateTime = 10, sigma = (1,1,1,1)):

	# Varianzas
	sigmaGPS, sigmaMag, sigmaAcc, sigmaGyro  = sigma

	# Relacion de tiempos de update
	delta = updateTime

	# Vector de estado de referencia
	x 	  = np.arange(T) 						# posicion en x
	x_dot = np.ones(x.shape)*speed		 		# velocidad en x
	y 	  = np.arange(T) 						# posicion en y
	y_dot = np.ones(y.shape)*speed			 	# velocidad en y
	theta = np.ones(x.shape)*(np.pi/4) 			# rotacion angular

	# Inicializar vectores de medicion.
	sizeIMU = (3,T)
	sizeGPS = (3,T)
	ref = np.vstack((x, x_dot, y, y_dot, theta)) 
	IMU = np.zeros(sizeIMU)
	GPS = np.zeros(sizeGPS)

	IMU[0,:] = np.random.normal(0, sigmaAcc, sizeIMU[1])
	IMU[1,:] = np.random.normal(0, sigmaAcc, sizeIMU[1])
	IMU[2,:] = np.random.normal(0, sigmaGyro, sizeIMU[1])

	for i in range(ref.shape[1]-delta+1):
		if not(i%delta):
			GPS[0,i:i+delta] = ref[0,i] + np.random.normal(0, sigmaGPS)
			GPS[1,i:i+delta] = ref[2,i] + np.random.normal(0, sigmaGPS)
			GPS[2,i:i+delta] = ref[4,i] + np.random.normal(0, sigmaMag)

	return ref, IMU, GPS

def polar_data(T = 100, omega = 1, step = 0.1, updateTime = 10, sigma = (1,1,1,1)):

	# Varianzas
	sigmaGPS, sigmaMag, sigmaAcc, sigmaGyro  = sigma

	# Relacion de tiempos de update
	delta = updateTime

	# Datos de la trayectoria
	r = 1000
	t = np.linspace(0, T*step, T)

	# Vector de estado de referencia
	theta = omega*t 						# rotacion angular
	x 	  = r*np.cos(theta) 				# posicion en x
	x_dot = (r*omega)*np.sin(theta)		 	# velocidad en x
	y 	  = r*np.sin(theta) 				# posicion en y
	y_dot = (r*omega)*np.cos(theta)			# velocidad en y

	# Inicializar vectores de medicion.
	sizeIMU = (3,T)
	sizeGPS = (3,T)
	ref = np.vstack((x, x_dot, y, y_dot, theta)) 
	IMU = np.zeros(sizeIMU)
	GPS = np.zeros(sizeGPS)

	IMU[0,:] = -r*omega**2 + np.random.normal(0, sigmaAcc, sizeIMU[1])
	IMU[1,:] = 				 np.random.normal(0, sigmaAcc, sizeIMU[1])
	IMU[2,:] = 		 omega + np.random.normal(0, sigmaGyro, sizeIMU[1])

	for i in range(ref.shape[1]-delta+1):
		if not(i%delta):
			GPS[0,i:i+delta] = ref[0,i] + np.random.normal(0, sigmaGPS)
			GPS[1,i:i+delta] = ref[2,i] + np.random.normal(0, sigmaGPS)
			GPS[2,i:i+delta] = ref[4,i] + np.random.normal(0, sigmaMag)

	return ref, IMU, GPS

if __name__ == '__main__':

	# Parsers & sub-parsers
	parser = argparse.ArgumentParser()

	# Parser arguments
	parser.add_argument('-d', '--data', dest='data', required=True, type=str, choices={'linear','polar'}, help='Data type: linear or polar')
	parser.add_argument('-s', '--std', dest='std', default='default', type=str, choices={'default','gps','acc'}, help='STD: default or gps or acc')
	parser.add_argument('-l', '--loss', dest='loss', default='default', type=str, choices={'default','gps','mag'}, help='Sensor loss: default or gps or mag')

	# Dictionary of parsers
	args = parser.parse_args()
	
	# Tiempo de sampleo IMU
	dt_IMU = 0.1

	# Tiempo de sampleo GPS/Mag
	dt_GPS = 1.0

	# Varianzas
	if args.std=='default':
		sigmaGPS  = 2.5
		sigmaMag  = 0.00872
		sigmaAcc  = np.sqrt(6.14656e-06)
		sigmaGyro = np.sqrt(2.74155e-06)
	if args.std=='gps':
		sigmaGPS  = 2.5*5
		sigmaMag  = 0.00872
		sigmaAcc  = np.sqrt(6.14656e-06)
		sigmaGyro = np.sqrt(2.74155e-06)
	if args.std=='acc':
		sigmaGPS  = 2.5
		sigmaMag  = 0.00872
		sigmaAcc  = np.sqrt(6.14656e-06)*5
		sigmaGyro = np.sqrt(2.74155e-06)	
	sigma = (sigmaGPS, sigmaMag, sigmaAcc, sigmaGyro)

	if args.data=='linear':

		# Pasos de simulacion
		T = 1000

		# Crear datos de referencia y mediciones, p/ el caso de movimiento rectilineo uniforme.
		ref, IMU, GPS = linear_data(T = T, speed=(np.sqrt(2)/dt_IMU), updateTime = int(dt_GPS/dt_IMU), sigma = sigma)

		# Generar filtro de Kalman
		x0 = np.expand_dims(ref[:,0], axis=1)
		linearKF = createKF(x0=x0, sigma = sigma)

		# Generar estimaciones sobre los datos lineales
		if args.loss=='default':
			linearEstimation, covariance, kalmanGains = linearKF.run(IMU, GPS, update = 'full',  updateTime = int(dt_GPS/dt_IMU))
		if args.loss=='gps':
			linearEstimation, covariance, kalmanGains = linearKF.run(IMU, GPS, update = 'loss', loss = [0,1], updateTime = int(dt_GPS/dt_IMU))
		if args.loss=='mag':
			linearEstimation, covariance, kalmanGains = linearKF.run(IMU, GPS, update = 'loss', loss = [2], updateTime = int(dt_GPS/dt_IMU))

		# Graficos lineales
		plt.figure(1)
		plt.title('Trayectoria')
		plt.xlabel('Posicion x [m]')
		plt.ylabel('Posicion y [m]')
		plt.plot(ref[0,:500], ref[2,:500], label = 'Real trajectory')
		plt.plot(GPS[0,:500], GPS[1,:500], label = 'GPS/Mag measurement')
		plt.plot(linearEstimation[0,:500], linearEstimation[1,:500], label = 'KF estimation')
		plt.axis([0,500,0,500])
		plt.legend()
		
		plt.figure(2)
		plt.title('Rotacion')
		plt.xlabel('Tiempo [ciclos]')
		plt.ylabel('Theta [rad]')
		plt.plot(ref[4,:500], label = 'Real trajectory')
		plt.plot(GPS[2,:500], label = 'GPS/Mag measurement')
		plt.plot(linearEstimation[2,:500], label = 'KF estimation')
		plt.axis([0,500,0.5,1.0])
		plt.legend()

		plt.figure(3)
		plt.title('Covarianza')
		plt.xlabel('Tiempo [ciclos]')
		plt.plot(covariance[0,0,:], label = 'C 11')
		plt.plot(covariance[1,1,:], label = 'C 22')
		plt.plot(covariance[2,2,:], label = 'C 33')
		plt.plot(covariance[3,3,:], label = 'C 44')
		plt.plot(covariance[4,4,:], label = 'C 55')
		plt.legend()

		plt.figure(4)
		plt.title('Kinf')
		plt.xlabel('Tiempo [ciclos]')
		plt.plot(kalmanGains[0,0,:], label = 'K 11')
		plt.plot(kalmanGains[0,1,:], label = 'K 12')
		plt.plot(kalmanGains[0,2,:], label = 'K 13')
		plt.plot(kalmanGains[1,0,:], label = 'K 21')
		plt.plot(kalmanGains[1,1,:], label = 'K 22')
		plt.plot(kalmanGains[1,2,:], label = 'K 23')
		plt.plot(kalmanGains[2,0,:], label = 'K 31')
		plt.plot(kalmanGains[2,1,:], label = 'K 32')
		plt.plot(kalmanGains[2,2,:], label = 'K 33')
		plt.plot(kalmanGains[3,0,:], label = 'K 41')
		plt.plot(kalmanGains[3,1,:], label = 'K 42')
		plt.plot(kalmanGains[3,2,:], label = 'K 43')
		plt.plot(kalmanGains[4,0,:], label = 'K 51')
		plt.plot(kalmanGains[4,1,:], label = 'K 52')
		plt.plot(kalmanGains[4,2,:], label = 'K 53')
		plt.legend()

		plt.show()

	if args.data=='polar':

		# Pasos de simulacion
		T = 100000

		# Crear datos de referencia y mediciones, p/ el caso de movimiento circular uniforme.
		ref, IMU, GPS = polar_data(T = T, omega=(np.pi*2/(dt_IMU*T)), updateTime = int(dt_GPS/dt_IMU), sigma = sigma)

		# Generar filtro de Kalman
		x0 = np.expand_dims(ref[:,0], axis=1)
		polarKF = createKF(x0=x0, sigma = sigma)

		# Generar estimaciones sobre los datos lineales
		if args.loss=='default':
			polarEstimation, covariance, kalmanGains = polarKF.run(IMU, GPS, update = 'full', updateTime = int(dt_GPS/dt_IMU))
		if args.loss=='gps':
			polarEstimation, covariance, kalmanGains = polarKF.run(IMU, GPS, update = 'loss', loss = [0,1], updateTime = int(dt_GPS/dt_IMU))
		if args.loss=='mag':
			polarEstimation, covariance, kalmanGains = polarKF.run(IMU, GPS, update = 'loss', loss = [2], updateTime = int(dt_GPS/dt_IMU))

		# Graficos lineales
		plt.figure(1)
		plt.title('Trayectoria')
		plt.xlabel('Posicion x [m]')
		plt.ylabel('Posicion y [m]')
		plt.plot(ref[0,:], ref[2,:], label = 'Real trajectory')
		plt.plot(GPS[0,:], GPS[1,:], label = 'GPS/Mag measurement')
		plt.plot(polarEstimation[0,:], polarEstimation[1,:], label = 'KF estimation')
		plt.axis('equal')
		plt.legend()
		
		plt.figure(2)
		plt.title('Rotacion')
		plt.xlabel('Tiempo [ciclos]')
		plt.ylabel('Theta [rad]')
		plt.plot(ref[4,:2000], label = 'Real trajectory')
		plt.plot(GPS[2,:2000], label = 'GPS/Mag measurement')
		plt.plot(polarEstimation[2,:2000], label = 'KF estimation')
		plt.legend()

		plt.figure(3)
		plt.title('Covarianza')
		plt.xlabel('Tiempo [ciclos]')
		plt.plot(covariance[0,0,:2000], label = 'C 11')
		plt.plot(covariance[1,1,:2000], label = 'C 22')
		plt.plot(covariance[2,2,:2000], label = 'C 33')
		plt.plot(covariance[3,3,:2000], label = 'C 44')
		plt.plot(covariance[4,4,:2000], label = 'C 55')
		plt.legend()

		plt.figure(4)
		plt.title('Kinf')
		plt.xlabel('Tiempo [ciclos]')
		plt.plot(kalmanGains[0,0,:2000], label = 'K 11')
		plt.plot(kalmanGains[0,1,:2000], label = 'K 12')
		plt.plot(kalmanGains[0,2,:2000], label = 'K 13')
		plt.plot(kalmanGains[1,0,:2000], label = 'K 21')
		plt.plot(kalmanGains[1,1,:2000], label = 'K 22')
		plt.plot(kalmanGains[1,2,:2000], label = 'K 23')
		plt.plot(kalmanGains[2,0,:2000], label = 'K 31')
		plt.plot(kalmanGains[2,1,:2000], label = 'K 32')
		plt.plot(kalmanGains[2,2,:2000], label = 'K 33')
		plt.plot(kalmanGains[3,0,:2000], label = 'K 41')
		plt.plot(kalmanGains[3,1,:2000], label = 'K 42')
		plt.plot(kalmanGains[3,2,:2000], label = 'K 43')
		plt.plot(kalmanGains[4,0,:2000], label = 'K 51')
		plt.plot(kalmanGains[4,1,:2000], label = 'K 52')
		plt.plot(kalmanGains[4,2,:2000], label = 'K 53')
		plt.legend()

		plt.show()
