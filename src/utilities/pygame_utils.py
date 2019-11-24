import asyncio
import time
import pygame

from av import VideoFrame


class Car:
    def __init__(self, configuration, update_override=None):
        # units in percentage range 0..1
        self.steering = 0.0
        self.throttle = 0.0
        self.braking = 0.0
        self.gear = 0
        self.d_steering = 0.0
        self.d_throttle = 0.0
        self.d_braking = 0.0

        self.max_steering = 1.0
        self.max_throttle = 1.0
        self.max_braking = 1.0
        self.braking_k = 5.0            # coefficient used for virtual speed braking calc
        self.min_deceleration = 5       # speed reduction when nothing is pressed
        # units of change over one second:
        self.steering_speed = 5.0
        self.steering_dissipation_speed = 3.0
        self.acceleration_speed = 5.0
        self.dissipation_speed = 2.0
        self.braking_speed = 5.0
        # virtual speed
        self.virtual_speed = 0.0
        self.max_virtual_speed = 5.0
        # key states
        self.left_down = False
        self.right_down = False
        self.up_down = False
        self.down_down = False

        # telemetry
        self.batVoltage_mV = 0

        self.__override_enabled = update_override is not None and configuration.model_override_enabled
        self.__update_override = update_override

    async def update(self, dt):
        self.__update_steering(dt)
        self.__update_linear_movement(dt)
        self.__update_direction()

        if self.__override_enabled:
            await self.__update_override(self)

        # calculate virtual speed
        if self.up_down == self.down_down:
            # nothing or both pressed
            self.virtual_speed = max(0.0, min(self.max_virtual_speed,
                                              self.virtual_speed - dt * self.min_deceleration))
        else:
            self.virtual_speed = max(0.0, min(self.max_virtual_speed,
                                              self.virtual_speed + dt * (self.throttle - self.braking_k * self.braking)))

    def __update_steering(self, dt):
        # calculate steering
        if (not self.left_down) and (not self.right_down):
            self.__passive_steering(dt)
        elif self.left_down and not self.right_down:
            self.d_steering = -dt * self.steering_speed
            if not self.__override_enabled:
                self.steering = max(-1.0, self.steering + self.d_steering)
        elif not self.left_down and self.right_down:
            self.d_steering = dt * self.steering_speed
            if not self.__override_enabled:
                self.steering = min(1.0, self.steering + self.d_steering)

    def __passive_steering(self, dt):
        if self.steering > 0.01:
            self.d_steering = -dt * self.steering_dissipation_speed
            if not self.__override_enabled:
                self.steering = max(0.0, self.steering + self.d_steering)
        elif self.steering < -0.01:
            self.d_steering = dt * self.steering_dissipation_speed
            if not self.__override_enabled:
                self.steering = min(0.0, self.steering + self.d_steering)
        else:
            self.d_steering = 0.0

    def __update_linear_movement(self, dt):
        # calculating gear, throttle, braking
        if self.up_down and not self.down_down:
            if self.gear == 0:
                self.__takeoff(dt, 1)
            if self.gear == 1:  # drive accelerating
                self.__accelerate(dt)
            elif self.gear == -1:  # reverse braking
                self.__decelerate(dt)
        elif not self.up_down and self.down_down:
            if self.gear == 0:
                self.__takeoff(dt, -1)
            if self.gear == 1:  # drive braking
                self.__decelerate(dt)
            elif self.gear == -1:  # reverse accelerating
                self.__accelerate(dt)
        else:  # both down or both up
            self.d_throttle = -dt * self.dissipation_speed
            self.d_braking = -dt * self.dissipation_speed
            if not self.__override_enabled:
                self.throttle = max(0.0, self.throttle + self.d_throttle)
                self.braking = max(0.0, self.braking + self.d_braking)

    def __takeoff(self, dt, gear):
        self.d_throttle = dt * self.acceleration_speed
        self.d_braking = max(-self.braking, -dt * self.braking_speed)
        if not self.__override_enabled:
            self.gear = gear
            self.throttle = 0.0
            self.braking = max(0.0, self.braking + self.d_braking)

    def __decelerate(self, dt):
        self.d_throttle = 0.0
        self.d_braking = dt * self.braking_speed
        if not self.__override_enabled:
            self.throttle = 0.0
            self.braking = min(self.max_braking, self.braking + self.d_braking)

    def __accelerate(self, dt):
        self.d_throttle = dt * self.acceleration_speed
        self.d_braking = max(-self.braking, -dt * self.braking_speed)
        if not self.__override_enabled:
            self.throttle = min(self.max_throttle, self.throttle + self.d_throttle)
            self.braking = max(0.0, self.braking + self.d_braking)

    def __update_direction(self):
        # conditions to change the direction
        if not self.up_down and not self.down_down and self.virtual_speed < 0.01 and not self.__override_enabled:
            self.gear = 0

    def ext_update_linear_movement(self, d_throttle, d_braking):
        self.throttle = min(self.max_throttle, self.throttle + d_throttle)
        self.braking = min(self.max_braking, self.braking + d_braking)

    def ext_update_steering(self, d_steering):
        if d_steering < 0:
            self.steering = max(-1.0, self.steering + d_steering)
        else:
            self.steering = min(1.0, self.steering + d_steering)


class PygameRenderer:
    def __init__(self, screen, car):
        self.window_width = 960
        self.window_height = 480
        self.FPS = 30
        self.latest_frame = None
        self.screen = screen
        self.car = car

        self.black = (0, 0, 0)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)

        self.font = pygame.font.SysFont('Roboto', 12)

    def pygame_event_loop(self, loop, event_queue):
        while True:
            event = pygame.event.wait()
            asyncio.run_coroutine_threadsafe(event_queue.put(event), loop=loop)

    async def register_pygame_events(self, event_queue):
        while True:
            event = await event_queue.get()
            if event.type == pygame.QUIT:
                print("event", event)
                break
            elif event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    self.car.left_down = event.type == pygame.KEYDOWN
                elif event.key == pygame.K_RIGHT:
                    self.car.right_down = event.type == pygame.KEYDOWN
                elif event.key == pygame.K_UP:
                    self.car.up_down = event.type == pygame.KEYDOWN
                elif event.key == pygame.K_DOWN:
                    self.car.down_down = event.type == pygame.KEYDOWN
            # print("event", event)
        asyncio.get_event_loop().stop()

    def draw(self):
        # Steering gauge:
        if self.car.steering < 0:
            R = pygame.Rect((self.car.steering + 1.0) / 2.0 * self.window_width,
                            self.window_height - 10,
                            -self.car.steering * self.window_width / 2,
                            10)
        else:
            R = pygame.Rect(self.window_width / 2,
                            self.window_height - 10,
                            self.car.steering * self.window_width / 2,
                            10)
        pygame.draw.rect(self.screen, self.green, R)

        # Acceleration/braking gauge:
        if self.car.gear == 1:
            if self.car.throttle > 0.0:
                R = pygame.Rect(self.window_width - 20,
                                0,
                                10,
                                self.window_height / 2 * self.car.throttle / self.car.max_throttle)
                R = R.move(0, self.window_height / 2 - R.height)
                pygame.draw.rect(self.screen, self.green, R)
            if self.car.braking > 0.0:
                R = pygame.Rect(self.window_width - 20,
                                self.window_height / 2,
                                10,
                                self.window_height / 2 * self.car.braking / self.car.max_braking)
                pygame.draw.rect(self.screen, self.red, R)
        elif self.car.gear == -1:
            if self.car.throttle > 0.0:
                R = pygame.Rect(self.window_width - 20,
                                self.window_height / 2,
                                10,
                                self.window_height / 2 * self.car.throttle / self.car.max_throttle)
                pygame.draw.rect(self.screen, self.green, R)
            if self.car.braking > 0.0:
                R = pygame.Rect(self.window_width - 20,
                                0,
                                10,
                                self.window_height / 2 * self.car.braking / self.car.max_braking)
                R = R.move(0, self.window_height / 2 - R.height)
                pygame.draw.rect(self.screen, self.red, R)

        # Speed gauge:
        if self.car.virtual_speed > 0.0:
            R = pygame.Rect(self.window_width - 10,
                            0,
                            10,
                            self.window_height * self.car.virtual_speed / self.car.max_virtual_speed)
            if self.car.gear >= 0:
                R = R.move(0, self.window_height - R.height)
            pygame.draw.rect(self.screen, self.green, R)

        if self.car.batVoltage_mV >= 0:
            telemetry_text = "{0} mV".format(self.car.batVoltage_mV)
            telemetry_texture = self.font.render(telemetry_text, True, self.red)
            self.screen.blit(telemetry_texture, (3, self.window_height - 14))

    # overlay is not the nicest but should be most performant way to display frame
    async def render(self, rcs):
        current_time = 0
        frame_size = (640, 480)
        ovl = pygame.Overlay(pygame.YV12_OVERLAY, frame_size)
        ovl.set_location(pygame.Rect(0, 0, self.window_width - 20, self.window_height - 10))
        while True:
            pygame.event.pump()
            last_time, current_time = current_time, time.time()
            await asyncio.sleep(1 / self.FPS - (current_time - last_time))  # tick
            await self.car.update((current_time - last_time) / 1.0)
            await rcs.updateControl(self.car.gear, self.car.steering, self.car.throttle, self.car.braking)
            self.screen.fill(self.black)
            if isinstance(self.latest_frame, VideoFrame):
                self.render_new_frames_on_screen(frame_size)

            self.draw()
            pygame.display.flip()

    def render_new_frames_on_screen(self, frame_size):
        image_to_ndarray = self.latest_frame.to_rgb().to_ndarray()
        surface = pygame.surfarray.make_surface(image_to_ndarray.swapaxes(0, 1))
        height = self.window_height - 10
        width = height * self.latest_frame.width // self.latest_frame.height
        x = (self.window_width - 20 - width) // 2
        y = 0
        scaled_frame = pygame.transform.scale(surface, (width, height))
        self.screen.blit(scaled_frame, (x, y))

    def handle_new_frame(self, frame):
        self.latest_frame = frame

    def handle_new_telemetry(self, telemetry):
        if self.car:
            self.car.batVoltage_mV = telemetry["b"]
