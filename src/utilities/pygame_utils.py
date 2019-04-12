import sys
sys.path.append("..")
#pylint: disable=import-error

import asyncio
import os
from getpass import getpass
from RCSnailPy.rcsnail import RCSnail, RCSLiveSession
import time
import pygame
from av import VideoFrame
import logging

class Car:
    def __init__(self):
        # units in percentage range 0..1 
        self.steering = 0.0
        self.throttle = 0.0
        self.braking = 0.0
        self.gear = 0
        self.max_steering = 1.0
        self.max_acceleration = 1.0
        self.max_braking = 1.0
        self.braking_k = 2.0            # coef used for virtual speed braking calc 
        self.min_deacceleration = 0.3   # speed reduction when nothing is pressed
        # units of change over one second:
        self.steering_speed = 5.0 
        self.steering_speed_neutral = 3.0 
        self.acceleration_speed = 5.0
        self.deacceleration_speed = 2.0
        self.braking_speed = 5.0
        # virtual speed
        self.virtual_speed = 0.0
        self.max_virtual_speed =  5.0
        # key states
        self.left_down = False
        self.right_down = False
        self.up_down = False
        self.down_down = False

        self.window_width = 960
        self.window_height = 480
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)

    def update(self, dt):
        # calculate steering
        if (not self.left_down) and (not self.right_down):
            # free center positioning
            if self.steering > 0:
                self.steering = max(0.0, self.steering - dt * self.steering_speed_neutral)
            else:
                self.steering = min(0.0, self.steering + dt * self.steering_speed_neutral)
        elif self.left_down and not self.right_down:
            self.steering = max(-1.0, self.steering - dt * self.steering_speed)
        elif not self.left_down and self.right_down:
            self.steering = min(1.0, self.steering + dt * self.steering_speed)

        # calculating gear, throttle, braking
        if self.up_down and not self.down_down:
            if self.gear == 0:
                self.gear = 1
                self.throttle = 0.0
            if self.gear == 1:     # drive accelerating
                self.throttle = min(self.max_acceleration, self.throttle + dt * self.acceleration_speed)
                self.braking = 0.0
            elif self.gear == -1:  # reverse braking
                self.braking = min(self.max_braking, self.braking + dt * self.braking_speed)
                self.throttle = 0.0
        elif not self.up_down and self.down_down:
            if self.gear == 0:
                self.gear = -1
                self.throttle = 0.0
            if self.gear == 1:     # drive braking
                self.braking = min(self.max_braking, self.braking + dt * self.braking_speed)
                self.throttle = 0.0
            elif self.gear == -1:  # reverse accelerating
                self.throttle = min(self.max_acceleration, self.throttle + dt * self.acceleration_speed)
                self.braking = 0.0
        else:  # both down or both up
            self.throttle = max(0.0, self.throttle - dt * self.deacceleration_speed)
            self.braking = max(0.0, self.braking - dt * self.deacceleration_speed)

        # calculate virtual speed
        if self.up_down == self.down_down:
            # nothing or both pressed
            self.virtual_speed = max(0.0, min(self.max_virtual_speed, 
                self.virtual_speed - dt * self.min_deacceleration))
        else:
            self.virtual_speed = max(0.0, min(self.max_virtual_speed, 
                self.virtual_speed + dt * (self.throttle - self.braking_k * self.braking)))
        
        # conditions to change the direction
        if not self.up_down and not self.down_down and self.virtual_speed < 0.01:
            self.gear = 0


    def draw(self, screen):
        # Steering gauge:
        if self.steering < 0:
            R = pygame.Rect((self.steering + 1.0) / 2.0 * self.window_width, 
                self.window_height - 10, 
                -self.steering * self.window_width / 2, 
                10)
        else:
            R = pygame.Rect(self.window_width / 2, 
                self.window_height - 10, 
                self.steering * self.window_width / 2, 
                10)
        pygame.draw.rect(screen, self.green, R)

        # Acceleration/braking gauge:
        if self.gear == 1:
            if self.throttle > 0.0:
                R = pygame.Rect(self.window_width - 20, 
                    0,
                    10,
                    self.window_height / 2 * self.throttle / self.max_acceleration)
                R = R.move(0, self.window_height / 2 - R.height)
                pygame.draw.rect(screen, self.green, R)
            if self.braking > 0.0:
                R = pygame.Rect(self.window_width - 20, 
                    self.window_height / 2,
                    10,
                    self.window_height / 2 * self.braking / self.max_braking)
                pygame.draw.rect(screen, self.red, R)
        elif self.gear == -1:
            if self.throttle > 0.0:
                R = pygame.Rect(self.window_width - 20, 
                    self.window_height / 2,
                    10,
                    self.window_height / 2 * self.throttle / self.max_acceleration)
                pygame.draw.rect(screen, self.green, R)
            if self.braking > 0.0:
                R = pygame.Rect(self.window_width - 20, 
                    0,
                    10,
                    self.window_height / 2 * self.braking / self.max_braking)
                R = R.move(0, self.window_height / 2 - R.height)
                pygame.draw.rect(screen, self.red, R)
        
        # Speed gauge:
        if self.virtual_speed > 0.0:
            R = pygame.Rect(self.window_width - 10,
                0,
                10,
                self.window_height * self.virtual_speed / self.max_virtual_speed)
            if self.gear >= 0:
                R = R.move(0, self.window_height - R.height)
            pygame.draw.rect(screen, self.green, R)


class PygameRenderer:
    def __init__(self): 
        self.window_width = 960
        self.window_height = 480
        self.FPS = 30
        self.black = (0, 0, 0)
        
        self.latest_frame = None

    def pygame_event_loop(self, loop, event_queue):
        while True:
            event = pygame.event.wait()
            asyncio.run_coroutine_threadsafe(event_queue.put(event), loop=loop)

    async def handle_pygame_events(self, event_queue, car):
        while True:
            event = await event_queue.get()
            if event.type == pygame.QUIT:
                print("event", event)
                break
            elif event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    car.left_down = event.type == pygame.KEYDOWN
                elif event.key == pygame.K_RIGHT:
                    car.right_down = event.type == pygame.KEYDOWN
                elif event.key == pygame.K_UP:
                    car.up_down = event.type == pygame.KEYDOWN
                elif event.key == pygame.K_DOWN:
                    car.down_down = event.type == pygame.KEYDOWN
            # print("event", event)
        asyncio.get_event_loop().stop()

    async def render(self, screen, car, rcs):
        current_time = 0
        # overlay is not the nicest but should be most performant way to display frame
        frame_size = (640, 480)
        ovl = pygame.Overlay(pygame.YV12_OVERLAY, frame_size)
        ovl.set_location(pygame.Rect(0, 0, self.window_width - 20, self.window_height - 10))
        while True:
            pygame.event.pump()
            last_time, current_time = current_time, time.time()
            await asyncio.sleep(1 / self.FPS - (current_time - last_time))  # tick
            car.update((current_time - last_time) / 1.0)
            await rcs.updateControl(car.gear, car.steering, car.throttle, car.braking)
            screen.fill(self.black)
            if isinstance(self.latest_frame, VideoFrame):
                if frame_size[0] != self.latest_frame.width or frame_size[1] != self.latest_frame.height:
                    frame_size = (self.latest_frame.width, self.latest_frame.height)
                    ovl = None
                    ovl = pygame.Overlay(pygame.YV12_OVERLAY, frame_size) # (320, 240))
                    ovl.set_location(pygame.Rect(0, 0, self.window_width - 20, self.window_height - 10))
                ovl.display((self.latest_frame.planes[0], self.latest_frame.planes[1], self.latest_frame.planes[2]))
                
                # check different frame formats https://docs.mikeboers.com/pyav/develop/api/video.html
                # PIL or Pillow must be installed:
                #image_pil = latest_frame.to_image()
                #screen.blit(image_pil, (0, 0))
                # Numpy must be installed:
                #image_to_ndarray = latest_frame.to_ndarray()

                #image_rgb = latest_frame.to_rgb()
                #screen.blit(image_rgb, (0, 0))
            car.draw(screen)
            pygame.display.flip()
        asyncio.get_event_loop().stop()

    def handle_new_frame(self, frame):
        #if isinstance(latest_frame, VideoFrame):
        #    VideoFrame(latest_frame).
        self.latest_frame = frame