"""
How to install the MATLAB Engine API:
https://ch.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

"""

from matlab.engine import start_matlab
from matlab import double
import math
import io


class MatlabWrapper:

    def __init__(self):
        self._out = io.StringIO()
        self._err = io.StringIO()
        self._eng = start_matlab()

    def __del__(self):
        self._eng.quit()
        self._out.close()
        self._err.close()

    def steering_command(self, ref_pose: list, curr_pose: list,
                         velocity: float):
        """
        Computes the steering angle command steerCmd, in degrees, using the  
        Stanley method.

        Parameters
        ----------
            ref_pose : str
                Reference pose, specified as an [x, y, Θ] vector. x and y are in meters, and Θ is in degrees.
                x and y specify the reference point to steer the vehicle toward. 
                Θ specifies the orientation angle of the path at this reference point and is positive in the counterclockwise direction.
            curr_pose : str
                Current pose of the vehicle, specified as an [x, y, Θ] vector. x and y are in meters, and Θ is in degrees.
                x and y specify the location of the vehicle, which is defined as the center of the vehicle's rear axle.
                Θ specifies the orientation angle of the vehicle at location (x,y) and is positive in the counterclockwise direction.
            velocity : float
                Current longitudinal velocity of the vehicle. Units are in meters per second.

        Returns
        ----------
            A steering angle command in degrees.
        """

        ref_pose = double(ref_pose)
        curr_pose = double(curr_pose)
        cmd = self._eng.lateralControllerStanley(
            ref_pose, curr_pose, velocity,
            stdout=self._out, stderr=self._err
        )

        self._log()

        return cmd

    def bicycle_kinematic_model_v0(self, steering_angle=math.pi/4):
        """
        Creates a bicycle vehicle model to simulate simplified car-like vehicle dynamics. 
        This model represents a vehicle with two axles separated by a distance, WheelBase. 
        The state of the vehicle is defined as a three-element vector, [x y theta], with a global xy-position, 
        specified in meters, and a vehicle heading angle, theta, specified in radians. The front wheel can be turned with steering angle psi. 
        The vehicle heading, theta, is defined at the center of the rear axle. 
        To compute the time derivative states of the model, use the derivative function with input commands and the current robot state.

        Parameters
        ----------
            steering_angle : float
                The maximum steering angle, psi, refers to the maximum angle the vehicle can be steered to the right or left, specified in radians. 
                A value of pi/2 provides the vehicle with a minimum turning radius of 0. 
                This property is used to validate the user-provided state input.
                Default value is PI/4 .

        Returns
        ----------
            Kinematic model of a bicycle. 
        """

        bicycle = self._eng.bicycleKinematics(
            "MaxSteeringAngle", steering_angle, stdout=self._out,
            stderr=self._err
        )

        self._log()

        return bicycle

    def _log(self):
        output = self._out.getvalue()
        if output != '':
            print(output)

        error = self._err.getvalue()
        if error != '':
            print(error)

    def bicycle_kinematic_model(self, x: float, y: float, theta: float,
                                v: float, psi: float) -> list:
        """
        Creates a bicycle vehicle model to simulate simplified car-like vehicle dynamics.
        This model represents a vehicle with two axles separated by a distance, WheelBase.
        The state of the vehicle is defined as a three-element vector, [x y theta], with a global xy-position,
        specified in meters, and a vehicle heading angle, theta, specified in radians. The front wheel can be turned with steering angle psi.
        The vehicle heading, theta, is defined at the center of the rear axle.
        To compute the time derivative states of the model, use the derivative function with input commands and the current robot state.

        Parameters
        ----------
            x : float
                Global x coordinate in meters.
            y : float
                Global y coordinate in meters.
            theta : float
                Vehicle heading angle specified in radians.
            v : float
                Velocity of the vehicle in m/s.
            psi : float
                The front wheel steering angle specified in radians.

        Returns
        ----------
            New global state of the vehicle as a list [x, y, theta].
        """

        self._eng.eval('''
        kinematicModel = bicycleKinematics;
        state = [{} {} {}];
        tspan = 0:0.05:0.1;
        inputs = [{} {}];
        [t,y] = ode45(@(t,y)derivative(kinematicModel,y,inputs),tspan,state);
        '''.format(x, y, theta, v, psi),
                       nargout=0,
                       stdout=self._out,
                       stderr=self._err
        )

        self._log()
        new_state = self._eng.workspace['y'][-1]

        return new_state
