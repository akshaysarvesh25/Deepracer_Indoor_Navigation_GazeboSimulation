# Deepracer Manual Control Using Joystick

## Edit the ~/.bashrc
```
startros(){
. /opt/ros/kinetic/setup.bash
. /opt/aws/deepracer/setup.bash
. /opt/aws/intel/dldt/bin/setupvars.sh
export PYTHONPATH=/opt/aws/pyudev/pyudev-0.21.0/src:$PYTHONPATH
}
```

## Dependency Packages -
```
sudo apt-get install jstest-gtk
sudo apt-get install ros-kinetic-joy
sudo apt-get install ros-kinetic-joy-teleop 
```

## To run the DeepRacer with Joystick </br>
```
startros
roslaunch deepracer_joy deepracer_joy.launch
```


## References
The original repository belongs to Allison Thackston.
https://github.com/athackst/deepracer_joy </br>


