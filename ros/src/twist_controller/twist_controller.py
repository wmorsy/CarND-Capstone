from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, 
          accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
	
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        kp = 0.3
        ki = 0.1
        kd = 0.0
        mn = 0.0
        mx = 0.2
        self.pid_throttle = PID(kp, ki, kd, mn, mx)

        tau = 0.5
        ts = .02
        self.filter = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capatity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw_enabled:
           self.pid_throttle.reset()
           return 0., 0., 0.

        current_vel = self.filter.filt(current_vel)
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        
        vel_diff = linear_vel - current_vel
        self.last_vel = current_vel

        time = rospy.get_time()
        time_diff = time - self.last_time
        self.last_time = time

        throttle = self.pid_throttle.step(vel_diff, time_diff)
        brake = 0

        if linear_vel == 0. and current_vel < 0.1:
           throttle = 0
           brake = 700
	elif throttle < .1 and vel_diff < 0:
           throttle = 0
           decel = max(vel_diff, self.decel_limit)
           brake = -decel*self.vehicle_mass*self.wheel_radius
   
        return throttle, brake, steering


