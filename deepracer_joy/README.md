# deepracer_joy

This package will allow you to control your DeepRacer via joystick using ROS.

## Quickstart

Run the launch file

```bash
roslaunch deepracer_joy deepracer_joy.launch joystick_device=/dev/input/js0
```

## Detailed directions

### Find your joystick device name

On linux:

```bash
cd /dev/input/
ls
```

My device name is `/dev/input/js0`.  

#### Test your input

You can test your gamepad using jstest

```bash
sudo apt-get install jstest-gtk

sudo jstest --normal /dev/input/js0
```

> Note: You may need to update the permissions of your device to read/write
> `sudo chmod a+rw /dev/input/js0`

### Configure your joystick

I've included an example configuration file that I use with my wired Logitech gamepad.

[logitech_dual_action.yaml](config/logitech_dual_action.yaml)

For the DeepRacer you need to both enable the control mode with the `/enable_state` rosservice and you will need to publish your commands to `/manual_drive`.
