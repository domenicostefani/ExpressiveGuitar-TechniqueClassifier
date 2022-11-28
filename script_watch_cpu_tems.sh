#!/bin/bash

watch -n 1 "sensors | grep -Po 'Core \d:[ ]+\+\d+\.\d+°C'" 
